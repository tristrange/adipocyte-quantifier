#!/usr/bin/env python3
"""
Adipocyte cross-section quantification from membrane (wall) detection.

Install:  
  pip install numpy pandas scikit-image matplotlib

Usage:
    python adipocyte_quant2.py \
        --input /path/to/folder \
        --output /path/to/out \
        --pixel-size-um 0.33 \
        --clear-border \
        --min-cell-area-px 600 --hole-fill-area-px 2500 \
        --min-measured-area-px 1500 \
        --overlay-labels --label-font-size 0
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Any, cast
from skimage import io, color, exposure, util, filters, feature, morphology, segmentation, measure
from skimage.morphology import disk, remove_small_holes, remove_small_objects
from skimage.filters import frangi, meijering
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


# ----------------------------
# Core functions
# ----------------------------

def load_gray(path):
    img = io.imread(path)
    if img.ndim == 3:
        gray = color.rgb2gray(img)
    else:
        gray = img.astype(float)
    gray = util.img_as_float(gray)
    eq = exposure.equalize_adapthist(gray, clip_limit=0.02)
    low, high = np.percentile(eq, (1, 99))
    low = float(low)
    high = float(high)
    scale = high - low
    if scale > 1e-6:
        eq = (eq - low) * (1.0 / scale)
        np.clip(eq, 0.0, 1.0, out=eq)
    return eq, img


def detect_membranes(
    gray,
    use_tophat=True, tophat_radius=3, tophat_gain=0.6,
    ridge_hi=0.85, ridge_lo_mul=0.45,
    canny_sigma=1.6, canny_low=0.03, canny_high=0.12,
    mem_close=3, mem_dilate=2):
    inv = 1.0 - gray

    # top-hat preboost (optional) — now robust across skimage versions
    if use_tophat:
        r = max(1, int(tophat_radius))
        se = disk(r)
        try:
            bth = morphology.black_tophat(gray, footprint=se)
        except Exception:
            bth = None

        if bth is None:
            # Fallback: closing(image) - image ≈ black tophat
            closed = morphology.closing(gray, footprint=se) if hasattr(morphology, "closing") else gray
            bth = np.clip(closed - gray, 0, 1)

        bth = np.asarray(bth, dtype=float)
        bmin, bmax = float(np.nanmin(bth)), float(np.nanmax(bth))
        brng = max(1e-8, bmax - bmin)
        bth = (bth - bmin) / brng
        inv = np.clip(inv + float(tophat_gain) * bth, 0, 1)

    # ridge enhancement (bright ridges on the inverted image)
    sigmas = np.linspace(0.8, 5.0, 9).tolist()
    sigmas_param = cast(Any, sigmas)
    ridge_f = frangi(inv, sigmas=sigmas_param, beta=0.5, gamma=15, black_ridges=False)
    ridge_m = meijering(inv, sigmas=sigmas_param, alpha=None, black_ridges=False)
    ridge = np.maximum(
        np.asarray(ridge_f, dtype=float, order=None),
        np.asarray(ridge_m, dtype=float, order=None)
    )
    rmin, rmax = float(np.nanmin(ridge)), float(np.nanmax(ridge))
    rrng = max(1e-8, rmax - rmin)
    ridge = (ridge - rmin) / rrng

    # hysteresis thresholds (tunable)
    hi = float(ridge_hi) * filters.threshold_otsu(ridge)
    lo = float(ridge_lo_mul) * hi
    mem_ridge = filters.apply_hysteresis_threshold(ridge, lo, hi)

    # permissive edges for faint walls
    edges = feature.canny(
        gray,
        sigma=float(canny_sigma),
        low_threshold=float(canny_low),
        high_threshold=float(canny_high),
    )

    # clean + connect membrane network
    membranes = mem_ridge | edges
    membranes = morphology.binary_opening(membranes, footprint=disk(1))
    membranes = morphology.binary_closing(membranes, footprint=disk(int(max(1, mem_close))))
    membranes = remove_small_objects(membranes, min_size=24)
    membranes = morphology.binary_dilation(membranes, footprint=disk(int(max(1, mem_dilate))))

    return membranes, ridge, inv


def segment_cells_from_membranes(membranes, clear_border, min_cell_area_px,
                                 hole_fill_area_px, use_watershed=True,
                                 ws_maxima_footprint=21):
    mask = ~membranes

    # Fill tiny interior holes (gets rid of micro loops inside walls)
    mask = remove_small_holes(mask, area_threshold=hole_fill_area_px)

    if clear_border:
        mask = segmentation.clear_border(mask)

    if not mask.any():
        return np.zeros(mask.shape, dtype=np.int_)

    if use_watershed:
        # Distance from walls; peaks are interior centers
        dist = np.asarray(ndi.distance_transform_edt(mask), dtype=float)

        min_dist = max(1, ws_maxima_footprint // 2)
        coords = peak_local_max(
            dist,
            min_distance=min_dist,
            labels=mask,
            exclude_border=False
        )
        # build marker image from coordinates
        markers = np.zeros(dist.shape, dtype=np.int32)
        if coords.size:
            rows, cols = coords.T
            markers[rows, cols] = np.arange(1, coords.shape[0] + 1, dtype=np.int32)

        lbl = segmentation.watershed(-dist, markers, mask=mask)

    else:
        # Fallback: simple connected components
        lbl = measure.label(mask, connectivity=2)
    lbl = np.asarray(lbl, dtype=np.int_)

    # Remove tiny regions
    props = measure.regionprops(lbl)
    tiny = np.array([p.label for p in props if p.area < min_cell_area_px], dtype=int)
    if tiny.size > 0:
        lbl[np.isin(lbl, tiny)] = 0
        lbl = np.asarray(measure.label(lbl > 0, connectivity=2), dtype=np.int_)

    return lbl


def overlay_boundaries(rgb, lbl):
    base = rgb if rgb.ndim == 3 else color.gray2rgb(rgb)
    base = util.img_as_float(base)
    b = segmentation.find_boundaries(lbl, mode="outer")
    out = base.copy()
    out[b, 0] = 1.0  # red
    out[b, 1] = 0.0
    out[b, 2] = 0.0
    return util.img_as_ubyte(out)


def measure_cells(lbl, pixel_size_um, min_measured_area_px=0):
    props = [p for p in measure.regionprops(lbl) if p.area >= min_measured_area_px]
    px_area = np.array([p.area for p in props])
    df = pd.DataFrame({
        "label": [p.label for p in props],
        "area_px": px_area,
        "equiv_diameter_px": [p.equivalent_diameter for p in props],
        "perimeter_px": [p.perimeter for p in props],
        "centroid_row": [p.centroid[0] for p in props],
        "centroid_col": [p.centroid[1] for p in props],
    })
    if pixel_size_um and pixel_size_um > 0:
        df["area_um2"] = df["area_px"] * (pixel_size_um ** 2)
        df["equiv_diameter_um"] = df["equiv_diameter_px"] * pixel_size_um
        df["perimeter_um"] = df["perimeter_px"] * pixel_size_um
    return df


def save_labeled_overlay(overlay_img, df, out_path, font_size=0):
    """
    Draw cell numbers at centroids on top of the overlay image and save.
    overlay_img: RGB uint8 image (e.g., output of overlay_boundaries)
    df: dataframe from measure_cells (needs 'label', 'centroid_row', 'centroid_col')
    """
    h, w = overlay_img.shape[:2]
    if font_size <= 0:
        # auto size based on image width
        font_size = max(8, min(28, w // 90))

    plt.figure(figsize=(w/100, h/100), dpi=100)
    plt.imshow(overlay_img)
    plt.axis("off")
    for _, r in df.iterrows():
        y, x = float(r["centroid_row"]), float(r["centroid_col"])
        plt.text(
            x, y, str(int(r["label"])),
            color="white", ha="center", va="center",
            fontsize=font_size, weight="bold",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="black")]
        )
    # tight save without borders
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


# ----------------------------
# Batch runner
# ----------------------------

def process_image(path, out_dir, pixel_size_um, clear_border, min_cell_area_px,
                  hole_fill_area_px, min_measured_area_px, save_overlay=True,
                  overlay_labels=False, label_font_size=0,
                  thin_membranes=False, use_watershed=True, ws_maxima_footprint=21):
    gray, rgb = load_gray(path)
    membranes, _ridge, _inv = detect_membranes(gray)
    membranes = np.asarray(membranes, dtype=bool)
    if membranes.ndim > 2:
        membranes = np.squeeze(membranes)
    if membranes.ndim > 2:
        membranes = membranes.any(axis=-1)
    if thin_membranes:
        membranes = morphology.thin(membranes)

    lbl = segment_cells_from_membranes(
        membranes, clear_border, min_cell_area_px, hole_fill_area_px,
        use_watershed=use_watershed, ws_maxima_footprint=ws_maxima_footprint
    )
    df = measure_cells(lbl, pixel_size_um, min_measured_area_px)

    stem = Path(path).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / f"{stem}_measurements.csv", index=False)
    if save_overlay:
        ov = overlay_boundaries(rgb, lbl)
        ov_path = out_dir / f"{stem}_overlay.png"
        io.imsave(ov_path, ov)
        if overlay_labels:
            labeled_path = out_dir / f"{stem}_overlay_labeled.png"
            save_labeled_overlay(ov, df, labeled_path, font_size=label_font_size)

    return df, lbl


def main():
    ap = argparse.ArgumentParser(description="Adipocyte quantification")
    ap.add_argument("--input", required=True, help="Folder with images (jpg/png/tif)")
    ap.add_argument("--output", required=True, help="Output folder")
    ap.add_argument("--pixel-size-um", type=float, default=1.0,
                    help="Micrometers per pixel for area/length units")
    ap.add_argument("--min-cell-area-px", type=int, default=400,
                    help="Discard components smaller than this (pixels)")
    ap.add_argument("--clear-border", action="store_true",
                    help="Exclude cells touching image border")
    ap.add_argument("--no-overlay", action="store_true",
                    help="Skip saving overlay PNGs")

    # segmentation post/measure controls
    ap.add_argument("--hole-fill-area-px", type=int, default=400,
                    help="Fill interior holes up to this area before labeling")
    ap.add_argument("--min-measured-area-px", type=int, default=0,
                    help="Exclude labeled regions smaller than this area in outputs")

    # overlay labeling
    ap.add_argument("--overlay-labels", action="store_true",
                    help="Also save an overlay PNG with cell numbers at centroids")
    ap.add_argument("--label-font-size", type=int, default=0,
                    help="Font size for overlay labels (0 = auto)")

    # membrane detection tuning
    ap.add_argument("--no-top-hat", action="store_true",
                    help="Disable black top-hat pre-enhancement (less aggressive)")
    ap.add_argument("--tophat-radius", type=int, default=3,
                    help="Radius for black top-hat (2–6 typical)")
    ap.add_argument("--tophat-gain", type=float, default=0.6,
                    help="Blend of top-hat into inverted image (0–1)")
    ap.add_argument("--ridge-hi", type=float, default=0.85,
                    help="Hysteresis HIGH threshold multiplier on ridge (lower → more aggressive)")
    ap.add_argument("--ridge-lo-mul", type=float, default=0.45,
                    help="LOW = ridge_hi * ridge_lo_mul (higher → less aggressive)")
    ap.add_argument("--canny-sigma", type=float, default=1.6,
                    help="Canny Gaussian sigma")
    ap.add_argument("--canny-low", type=float, default=0.03,
                    help="Canny low threshold (0–1)")
    ap.add_argument("--canny-high", type=float, default=0.12,
                    help="Canny high threshold (0–1)")
    ap.add_argument("--mem-close", type=int, default=3,
                    help="binary_closing disk size for membranes (2–4 typical)")
    ap.add_argument("--mem-dilate", type=int, default=2,
                    help="binary_dilation disk size for membranes (1–3 typical)")

    # watershed & thinning
    ap.add_argument("--thin-membranes", action="store_true",
                    help="Skeletonize membranes to 1 px before segmentation")
    ap.add_argument("--use-watershed", action="store_true",
                    help="Use distance-transform watershed for interiors")
    ap.add_argument("--ws-maxima-footprint", type=int, default=21,
                    help="Footprint (px) / min spacing for watershed seeds")

    # debug
    ap.add_argument("--debug", action="store_true",
                    help="Save intermediate masks (inv, ridge, membranes, interiors)")

    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    images = sorted([p for p in in_dir.iterdir()
                     if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}])

    all_rows = []
    for p in images:
        print(f"Processing {p.name} ...")
        df, _ = process_image(
            p, out_dir,
            pixel_size_um=args.pixel_size_um,
            clear_border=args.clear_border,
            min_cell_area_px=args.min_cell_area_px,
            hole_fill_area_px=args.hole_fill_area_px,
            min_measured_area_px=args.min_measured_area_px,
            save_overlay=not args.no_overlay,
            overlay_labels=args.overlay_labels,
            label_font_size=args.label_font_size,
            thin_membranes=args.thin_membranes,
            use_watershed=args.use_watershed,
            ws_maxima_footprint=args.ws_maxima_footprint
        )
        df["image"] = p.name
        all_rows.append(df)

    if all_rows:
        full = pd.concat(all_rows, ignore_index=True)
        full.to_csv(out_dir / "all_measurements.csv", index=False)  # batch results
        summary = (full.groupby("image")["area_px"]
                   .agg(count="count", mean="mean", median="median", std="std")
                   .reset_index())
        if "area_um2" in full:
            summary_um = (full.groupby("image")["area_um2"]
                          .agg(mean_um2="mean", median_um2="median", std_um2="std")
                          .reset_index())
            summary = summary.merge(summary_um, on="image", how="left")
        summary.to_csv(out_dir / "summary_by_image.csv", index=False)
        print(f"Saved: {out_dir/'all_measurements.csv'} and {out_dir/'summary_by_image.csv'}")
    else:
        print("No images found.")


if __name__ == "__main__":
    main()
