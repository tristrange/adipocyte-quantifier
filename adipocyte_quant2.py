#!/usr/bin/env python3
"""
Adipocyte cross-section quantification from membrane (wall) detection.
OpenCV-free version using only scikit-image.

Install (no OpenCV needed):  
  pip install numpy pandas scikit-image matplotlib

Usage:
    python adipocyte_quant.py \
        --input /path/to/folder \
        --output /path/to/out \
        --pixel-size-um 0.33 \
        --clear-border \
        --min-cell-area-px 600
    python adipocyte_quant.py \
        --input ./imgs --output ./out \
        --pixel-size-um 0.33 --clear-border \
        --min-cell-area-px 600 --hole-fill-area-px 2500 \
        --min-measured-area-px 1500 \
        --overlay-labels --label-font-size 0
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Any, Tuple, cast
from skimage import io, color, exposure, util, filters, feature, morphology, segmentation, measure
from skimage.morphology import disk, remove_small_holes, remove_small_objects
from skimage.filters import frangi, meijering  # ridge filters for membranes
from skimage.feature import peak_local_max  # for watershed seeding
from numpy.typing import NDArray
from scipy import ndimage as ndi  # NEW: for distance transform
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
    # CHANGED: slightly stronger local equalization and percentile rescale to boost faint walls
    eq = exposure.equalize_adapthist(gray, clip_limit=0.02)  # CHANGED
    low, high = np.percentile(eq, (1, 99))
    in_range: Tuple[float, float] = (float(low), float(high))
    eq = exposure.rescale_intensity(eq, in_range=cast(Any, in_range))  # NEW
    return eq, img


# def detect_membranes(gray):
#     inv = 1.0 - gray  # membranes are dark -> invert so they become bright ridges

#     # NEW: black top-hat on the original to enhance faint dark lines, then add to 'inv'
#     # radius ~ membrane half-thickness; try 2–5
#     TOPHAT_RAD = 3  # NEW
#     bth = morphology.black_tophat(gray, footprint=disk(TOPHAT_RAD))  # NEW
#     bth = np.asarray(bth, dtype=float)
#     bth_range = float(np.ptp(bth))
#     bth = (bth - float(bth.min())) / (bth_range + 1e-8)               # NEW
#     inv = np.clip(inv + 0.6 * bth, 0, 1)                              # NEW

#     # CHANGED: widen scales and tell Frangi we expect BRIGHT ridges on 'inv'
#     sigmas = np.linspace(0.8, 5.0, 9).tolist()                        # CHANGED
#     ridge_f = frangi(inv, sigmas=sigmas, beta=0.5, gamma=15, black_ridges=False)  # CHANGED
#     ridge_m = meijering(inv, sigmas=sigmas, alpha=None, black_ridges=False)
#     if ridge_f is None or ridge_m is None:
#         msg = "Ridge filters returned None; check scikit-image installation."
#         raise RuntimeError(msg)
#     ridge = np.asarray(np.maximum(ridge_f, ridge_m), dtype=float)

#     ridge_min = float(ridge.min())
#     ridge_range = float(np.ptp(ridge))
#     ridge = (ridge - ridge_min) / (ridge_range + 1e-8)

#     # CHANGED: use hysteresis to pull in weak ridges connected to strong ones
#     hi = filters.threshold_otsu(ridge) * 0.85                         # CHANGED
#     lo = hi * 0.45                                                    # CHANGED
#     mem_ridge = filters.apply_hysteresis_threshold(ridge, lo, hi)     # CHANGED

#     # CHANGED: slightly more permissive Canny to catch vague edges
#     edges = feature.canny(gray, sigma=1.6, low_threshold=0.03, high_threshold=0.12)  # CHANGED

#     membranes = mem_ridge | edges

#     # CHANGED: a bit heavier gap-bridging, then a light dilation
#     membranes = morphology.binary_opening(membranes, footprint=disk(1))
#     membranes = morphology.binary_closing(membranes, footprint=disk(3))  # CHANGED
#     # membranes = morphology.binary_closing(membranes, footprint=disk(1))  # NEW (optional micro-bridges)
#     membranes = remove_small_objects(membranes, min_size=24)             # CHANGED
#     membranes = morphology.binary_dilation(membranes, footprint=disk(2)) # CHANGED

#     return membranes

# change signature
# def detect_membranes(gray,
#                      use_tophat=True, tophat_radius=3, tophat_gain=0.6,   # NEW
#                      ridge_hi=0.85, ridge_lo_mul=0.45,                    # NEW
#                      canny_sigma=1.6, canny_low=0.03, canny_high=0.12,    # NEW
#                      mem_close=3, mem_dilate=2):                          # NEW

#     inv = 1.0 - gray

#     # top-hat preboost (optional)
#     if use_tophat:                                                       # NEW
#         TOPHAT_RAD = max(1, int(tophat_radius))                          # NEW
#         bth = morphology.black_tophat(gray, footprint=disk(TOPHAT_RAD))  # NEW
#         bth = (bth - bth.min()) / (bth.ptp() + 1e-8)                     # NEW
#         inv = np.clip(inv + float(tophat_gain) * bth, 0, 1)              # NEW

#     sigmas = np.linspace(0.8, 5.0, 9).tolist()
#     ridge_f = frangi(inv, sigmas=sigmas, beta=0.5, gamma=15, black_ridges=False)
#     ridge_m = meijering(inv, sigmas=sigmas, alpha=None, black_ridges=False)
#     ridge = np.maximum(ridge_f, ridge_m)
#     ridge = (ridge - ridge.min()) / (ridge.ptp() + 1e-8)

#     # hysteresis thresholds (now tunable)
#     hi = float(ridge_hi) * filters.threshold_otsu(ridge)                 # NEW
#     lo = float(ridge_lo_mul) * hi                                        # NEW
#     mem_ridge = filters.apply_hysteresis_threshold(ridge, lo, hi)

#     edges = feature.canny(gray,
#                           sigma=float(canny_sigma),
#                           low_threshold=float(canny_low),
#                           high_threshold=float(canny_high))               # NEW

#     membranes = mem_ridge | edges
#     membranes = morphology.binary_opening(membranes, footprint=disk(1))
#     membranes = morphology.binary_closing(membranes, footprint=disk(int(mem_close)))   # CHANGED
#     membranes = remove_small_objects(membranes, min_size=24)
#     membranes = morphology.binary_dilation(membranes, footprint=disk(int(mem_dilate))) # CHANGED

#     return membranes, ridge, inv                                           # NEW (return ridge, inv for debug)

def detect_membranes(
    gray,
    use_tophat=True, tophat_radius=3, tophat_gain=0.6,                 # NEW
    ridge_hi=0.85, ridge_lo_mul=0.45,                                   # NEW
    canny_sigma=1.6, canny_low=0.03, canny_high=0.12,                   # NEW
    mem_close=3, mem_dilate=2):
    inv = 1.0 - gray

    # top-hat preboost (optional) — now robust across skimage versions
    if use_tophat:
        r = max(1, int(tophat_radius))                                  # CHANGED
        se = disk(r)                                                     # NEW
        try:
            bth = morphology.black_tophat(gray, footprint=se)            # CHANGED
        except Exception:
            bth = None                                                   # NEW

        if bth is None:                                                  # NEW
            # Fallback: closing(image) - image ≈ black tophat
            closed = morphology.closing(gray, footprint=se) if hasattr(morphology, "closing") else gray  # NEW
            bth = np.clip(closed - gray, 0, 1)                           # NEW

        bth = np.asarray(bth, dtype=float)                               # NEW
        bmin, bmax = float(np.nanmin(bth)), float(np.nanmax(bth))        # NEW
        brng = max(1e-8, bmax - bmin)                                    # NEW
        bth = (bth - bmin) / brng                                        # NEW
        inv = np.clip(inv + float(tophat_gain) * bth, 0, 1)              # CHANGED

    # ridge enhancement (bright ridges on the inverted image)
    sigmas = np.linspace(0.8, 5.0, 9).tolist()
    ridge_f = frangi(inv, sigmas=sigmas, beta=0.5, gamma=15, black_ridges=False)
    ridge_m = meijering(inv, sigmas=sigmas, alpha=None, black_ridges=False)
    ridge = np.maximum(np.asarray(ridge_f, float), np.asarray(ridge_m, float))  # CHANGED
    rmin, rmax = float(np.nanmin(ridge)), float(np.nanmax(ridge))               # NEW
    rrng = max(1e-8, rmax - rmin)                                               # NEW
    ridge = (ridge - rmin) / rrng                                               # CHANGED

    # hysteresis thresholds (tunable)
    hi = float(ridge_hi) * filters.threshold_otsu(ridge)                  # CHANGED
    lo = float(ridge_lo_mul) * hi                                         # CHANGED
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
    membranes = morphology.binary_closing(membranes, footprint=disk(int(max(1, mem_close))))   # CHANGED
    membranes = remove_small_objects(membranes, min_size=24)
    membranes = morphology.binary_dilation(membranes, footprint=disk(int(max(1, mem_dilate)))) # CHANGED

    return membranes, ridge, inv


# def segment_cells_from_membranes(membranes, clear_border, min_cell_area_px):
#     interiors = ~membranes
#     interiors = remove_small_holes(interiors, area_threshold=min_cell_area_px)
#     if clear_border:
#         interiors = segmentation.clear_border(interiors)
#     lbl: NDArray[np.int_] = np.asarray(
#         measure.label(interiors, connectivity=2), dtype=np.int_
#     )

#     # Remove tiny regions
#     props = measure.regionprops(lbl)
#     tiny = np.array([p.label for p in props if p.area < min_cell_area_px], dtype=int)
#     if tiny.size > 0:
#         mask = np.isin(lbl, tiny)
#         lbl[mask] = 0
#         lbl = np.asarray(measure.label(lbl > 0, connectivity=2), dtype=np.int_)
#     return lbl

# --- update segment_cells_from_membranes signature & usage -------------------

# def segment_cells_from_membranes(membranes, clear_border, min_cell_area_px, hole_fill_area_px):  # NEW arg
#     interiors = ~membranes
#     # CHANGED: use dedicated hole-fill threshold instead of min_cell_area_px
#     interiors = remove_small_holes(interiors, area_threshold=hole_fill_area_px)  # CHANGED
#     if clear_border:
#         interiors = segmentation.clear_border(interiors)
#     lbl: NDArray[np.int_] = np.asarray(
#         measure.label(interiors, connectivity=2), dtype=np.int_
#     )

#     # Remove tiny labeled specks (true tiny “cells”)
#     props = measure.regionprops(lbl)
#     tiny = np.array([p.label for p in props if p.area < min_cell_area_px], dtype=int)
#     if tiny.size > 0:
#         mask = np.isin(lbl, tiny)
#         lbl[mask] = 0
#         lbl = np.asarray(measure.label(lbl > 0, connectivity=2), dtype=np.int_)
#     return lbl

def segment_cells_from_membranes(membranes, clear_border, min_cell_area_px,
                                 hole_fill_area_px, use_watershed=True,
                                 ws_maxima_footprint=21):  # NEW args
    mask = ~membranes  # interiors candidate

    # Fill tiny interior holes (gets rid of micro loops inside walls)
    mask = remove_small_holes(mask, area_threshold=hole_fill_area_px)

    if clear_border:
        mask = segmentation.clear_border(mask)

    if use_watershed:
        # Distance from walls; peaks are interior centers
        dist = np.asarray(ndi.distance_transform_edt(mask), dtype=float)

        # NEW: pick seed coordinates with peak_local_max (controls min spacing)
        min_dist = max(1, ws_maxima_footprint // 2)  # NEW
        coords = peak_local_max(                      # NEW
            dist,
            min_distance=min_dist,
            labels=mask,
            exclude_border=False
        )
        # build marker image from coordinates
        markers = np.zeros(dist.shape, dtype=np.int32)   # NEW
        for i, (r, c) in enumerate(coords, start=1):     # NEW
            markers[r, c] = i                             # NEW

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


# def measure_cells(lbl, pixel_size_um):
#     props = measure.regionprops(lbl)
#     px_area = np.array([p.area for p in props])
#     df = pd.DataFrame({
#         "label": [p.label for p in props],
#         "area_px": px_area,
#         "equiv_diameter_px": [p.equivalent_diameter for p in props],
#         "perimeter_px": [p.perimeter for p in props],
#         "centroid_row": [p.centroid[0] for p in props],
#         "centroid_col": [p.centroid[1] for p in props],
#     })
#     if pixel_size_um and pixel_size_um > 0:
#         df["area_um2"] = df["area_px"] * (pixel_size_um ** 2)
#         df["equiv_diameter_um"] = df["equiv_diameter_px"] * pixel_size_um
#         df["perimeter_um"] = df["perimeter_px"] * pixel_size_um
#     return df

# --- update measure_cells to support a minimum measured area -----------------

def measure_cells(lbl, pixel_size_um, min_measured_area_px=0):  # NEW arg
    props = [p for p in measure.regionprops(lbl) if p.area >= min_measured_area_px]  # NEW filter
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


# NEW
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
            path_effects=[pe.withStroke(linewidth=2.5, foreground="black")]  # halo for readability
        )
    # tight save without borders
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


# ----------------------------
# Batch runner
# ----------------------------

# def process_image(path, out_dir, pixel_size_um, clear_border, min_cell_area_px, save_overlay=True):
#     gray, rgb = load_gray(path)
#     membranes = detect_membranes(gray)
#     lbl = segment_cells_from_membranes(membranes, clear_border, min_cell_area_px)
#     df = measure_cells(lbl, pixel_size_um)

#     stem = Path(path).stem
#     out_dir.mkdir(parents=True, exist_ok=True)

#     df.to_csv(out_dir / f"{stem}_measurements.csv", index=False)
#     if save_overlay:
#         ov = overlay_boundaries(rgb, lbl)
#         io.imsave(out_dir / f"{stem}_overlay.png", ov)

#     return df, lbl

# --- process_image: pass the new knobs --------------------------------------

# def process_image(path, out_dir, pixel_size_um, clear_border, min_cell_area_px,
#                   hole_fill_area_px, min_measured_area_px, save_overlay=True):  # NEW args
#     gray, rgb = load_gray(path)
#     membranes = detect_membranes(gray)
#     lbl = segment_cells_from_membranes(
#         membranes, clear_border, min_cell_area_px, hole_fill_area_px  # CHANGED
#     )
#     df = measure_cells(lbl, pixel_size_um, min_measured_area_px)  # CHANGED

#     stem = Path(path).stem
#     out_dir.mkdir(parents=True, exist_ok=True)

#     df.to_csv(out_dir / f"{stem}_measurements.csv", index=False)
#     if save_overlay:
#         ov = overlay_boundaries(rgb, lbl)
#         io.imsave(out_dir / f"{stem}_overlay.png", ov)

#     return df, lbl

def process_image(path, out_dir, pixel_size_um, clear_border, min_cell_area_px,
                  hole_fill_area_px, min_measured_area_px, save_overlay=True,
                  overlay_labels=False, label_font_size=0,
                  thin_membranes=False, use_watershed=True, ws_maxima_footprint=21):  # NEW
    gray, rgb = load_gray(path)
    membranes = detect_membranes(gray)
    membranes = np.asarray(membranes)
    if membranes.ndim == 3:
        # collapse any trailing channel axis (e.g., HxWx1 or HxWx3)
        membranes = membranes.any(axis=-1)  # NEW
    elif membranes.ndim > 3:
        membranes = np.squeeze(membranes)   # NEW
    membranes = membranes.astype(bool)
    if thin_membranes:  # NEW
        membranes = morphology.thin(membranes)

    lbl = segment_cells_from_membranes(
        membranes, clear_border, min_cell_area_px, hole_fill_area_px,
        use_watershed=use_watershed, ws_maxima_footprint=ws_maxima_footprint  # NEW
    )
    df = measure_cells(lbl, pixel_size_um, min_measured_area_px)

    stem = Path(path).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / f"{stem}_measurements.csv", index=False)
    if save_overlay:
        ov = overlay_boundaries(rgb, lbl)
        ov_path = out_dir / f"{stem}_overlay.png"
        io.imsave(ov_path, ov)
        if overlay_labels:  # NEW
            labeled_path = out_dir / f"{stem}_overlay_labeled.png"  # NEW
            save_labeled_overlay(ov, df, labeled_path, font_size=label_font_size)  # NEW

    return df, lbl


def main():
    ap = argparse.ArgumentParser(description="Adipocyte quantification")
    ap.add_argument("--input", required=True, help="Folder with images (jpg/png/tif)")  # RESTORED
    ap.add_argument("--output", required=True, help="Output folder")                    # RESTORED
    ap.add_argument("--pixel-size-um", type=float, default=1.0,
                    help="Micrometers per pixel for area/length units")                 # RESTORED
    ap.add_argument("--min-cell-area-px", type=int, default=400,
                    help="Discard components smaller than this (pixels)")               # RESTORED
    ap.add_argument("--clear-border", action="store_true",
                    help="Exclude cells touching image border")                         # RESTORED
    ap.add_argument("--no-overlay", action="store_true",
                    help="Skip saving overlay PNGs")                                    # RESTORED

    # segmentation post/measure controls
    ap.add_argument("--hole-fill-area-px", type=int, default=400,
                    help="Fill interior holes up to this area before labeling")         # RESTORED
    ap.add_argument("--min-measured-area-px", type=int, default=0,
                    help="Exclude labeled regions smaller than this area in outputs")   # RESTORED

    # overlay labeling
    ap.add_argument("--overlay-labels", action="store_true",
                    help="Also save an overlay PNG with cell numbers at centroids")     # RESTORED
    ap.add_argument("--label-font-size", type=int, default=0,
                    help="Font size for overlay labels (0 = auto)")                     # RESTORED

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
                    help="Skeletonize membranes to 1 px before segmentation")           # RESTORED
    ap.add_argument("--use-watershed", action="store_true",
                    help="Use distance-transform watershed for interiors")              # CHANGED (flag only)
    ap.add_argument("--ws-maxima-footprint", type=int, default=21,
                    help="Footprint (px) / min spacing for watershed seeds")            # RESTORED

    # debug
    ap.add_argument("--debug", action="store_true",
                    help="Save intermediate masks (inv, ridge, membranes, interiors)")  # RESTORED

    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    images = sorted([p for p in in_dir.iterdir()
                     if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}])

    all_rows = []
    # for p in images:
    #     print(f"Processing {p.name} ...")
    #     df, _ = process_image(
    #         p, out_dir,
    #         pixel_size_um=args.pixel_size_um,
    #         clear_border=args.clear_border,
    #         min_cell_area_px=args.min_cell_area_px,
    #         save_overlay=not args.no_overlay
    #     )
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
            thin_membranes=args.thin_membranes,                # NEW
            use_watershed=args.use_watershed,                  # NEW
            ws_maxima_footprint=args.ws_maxima_footprint       # NEW
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
