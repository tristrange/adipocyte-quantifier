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
from skimage import io, color, exposure, util, filters, feature, morphology, segmentation, measure, draw
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


# NEW / FIXED: include ALL labels in the adjacency dict so isolated cells get unique colors
def _label_adjacency(lbl: np.ndarray) -> dict[int, set[int]]:
    L = np.asarray(lbl, dtype=np.int32)
    if L.ndim != 2:
        L = np.squeeze(L)
    H, W = L.shape
    nbrs: dict[int, set[int]] = {}

    def add_edge(a: int, b: int):
        if a == 0 or b == 0 or a == b:
            return
        nbrs.setdefault(a, set()).add(b)
        nbrs.setdefault(b, set()).add(a)

    # 4-neighborhood edges (up/down/left/right)
    A, B = L[1:, :], L[:-1, :]
    mask = (A != B)
    for a, b in zip(A[mask].ravel(), B[mask].ravel()):
        add_edge(int(a), int(b))

    A, B = L[:, 1:], L[:, :-1]
    mask = (A != B)
    for a, b in zip(A[mask].ravel(), B[mask].ravel()):
        add_edge(int(a), int(b))

    # Ensure every label > 0 exists as a node (isolated regions get empty neighbor sets)  # FIXED
    labs = np.unique(L)
    for lab in labs[labs > 0]:
        nbrs.setdefault(int(lab), set())

    return nbrs


def _greedy_coloring(nbrs: dict[int, set[int]], max_colors: int = 4) -> dict[int, int]:
    # order by degree (highest first)
    order = sorted(nbrs.keys(), key=lambda k: len(nbrs[k]), reverse=True)
    color_of: dict[int, int] = {}
    for n in order:
        used = {color_of[m] for m in nbrs[n] if m in color_of}
        c = 0
        while c in used and c < max_colors - 1:
            c += 1
        color_of[n] = c
    return color_of


def colorize_labels_overlay(lbl: np.ndarray, base_img: np.ndarray | None, alpha: float = 0.45) -> np.ndarray:
    from skimage import color as skcolor, util as skut
    L = np.asarray(lbl)
    if L.ndim != 2:
        L = np.squeeze(L)
        if L.ndim != 2:
            raise ValueError("label image must be 2D after squeeze")

    # Build adjacency including isolated labels (fix) and color them
    nbrs = _label_adjacency(L)                                 # FIXED
    color_idx = _greedy_coloring(nbrs, max_colors=8)

    palette = np.array([
        [0.894, 0.102, 0.110],  # red
        [0.216, 0.494, 0.722],  # blue
        [0.302, 0.686, 0.290],  # green
        [0.596, 0.306, 0.639],  # purple
        [1.000, 0.498, 0.000],  # orange
        [1.000, 1.000, 0.200],  # yellow
        [0.651, 0.337, 0.157],  # brown
        [0.969, 0.506, 0.749],  # pink
    ])

    H, W = L.shape
    color_img = np.zeros((H, W, 3), dtype=float)
    for lab, cidx in color_idx.items():
        mask = (L == lab)
        color_img[mask] = palette[cidx % len(palette)]

    # Blend onto base image (or return solid colored map if base is None)
    if base_img is None:
        return (np.clip(color_img, 0, 1) * 255).astype(np.uint8)

    base = np.asarray(base_img, dtype=float)
    if base.ndim == 2:
        base = skcolor.gray2rgb(base)
    if base.shape[-1] != 3:
        base = base[..., :3]
    base = base / 255.0 if base.max() > 1.0 else base
    a = float(np.clip(alpha, 0.0, 1.0))
    out = np.clip(a * color_img + (1 - a) * base, 0, 1)
    return (out * 255).astype(np.uint8)


# NEW: simple label→color mapper (cycles a fixed palette)
def colorize_labels_flat(lbl: np.ndarray, base_img: np.ndarray | None = None, alpha: float = 0.45) -> np.ndarray:
    from skimage import color as skcolor, util as skut
    L = np.asarray(lbl)
    if L.ndim != 2:
        L = np.squeeze(L)
        if L.ndim != 2:
            raise ValueError("label image must be 2D after squeeze")

    # fixed palette (8 distinct colors); labels cycle 0..7  # NEW
    palette = np.array([
        [0.894, 0.102, 0.110],  # red
        [0.216, 0.494, 0.722],  # blue
        [0.302, 0.686, 0.290],  # green
        [0.596, 0.306, 0.639],  # purple
        [1.000, 0.498, 0.000],  # orange
        [1.000, 1.000, 0.200],  # yellow
        [0.651, 0.337, 0.157],  # brown
        [0.969, 0.506, 0.749],  # pink
    ], dtype=float)

    H, W = L.shape
    color_img = np.zeros((H, W, 3), dtype=float)

    # assign color by (label % len(palette)); skip background 0  # NEW
    labs = np.unique(L)
    labs = labs[labs > 0]
    for lab in labs:
        color_img[L == lab] = palette[int(lab) % len(palette)]

    if base_img is None:
        return (np.clip(color_img, 0, 1) * 255).astype(np.uint8)

    base = np.asarray(base_img, dtype=float)
    if base.ndim == 2:
        base = skcolor.gray2rgb(base)
    if base.shape[-1] != 3:
        base = base[..., :3]
    base = base / 255.0 if base.max() > 1.0 else base

    a = float(np.clip(alpha, 0.0, 1.0))
    out = np.clip(a * color_img + (1 - a) * base, 0, 1)
    return (out * 255).astype(np.uint8)


# NEW
def _bridge_membrane_gaps(
    membranes: np.ndarray,
    ridge: np.ndarray,
    max_dist: int = 12,
    min_ridge: float = 0.18,
    iterations: int = 1,
    dilate_radius: int = 1,
) -> np.ndarray:
    """
    Connect close skeleton endpoints if the straight-line path has sufficient ridge support.
    membranes: bool HxW
    ridge: float HxW in [0,1] (from detect_membranes)
    """
    m = membranes.astype(bool)
    ridge = np.asarray(ridge, dtype=float)
    iterations = max(1, int(iterations))
    dilate_radius = max(0, int(dilate_radius))
    dilate_fp = disk(dilate_radius) if dilate_radius > 0 else None

    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    max_d2 = max_dist * max_dist
    min_ridge = float(min_ridge)

    for _ in range(iterations):
        sk = morphology.skeletonize(m)

        # count 8-neighborhood degree; endpoints have exactly 1 neighbor
        deg = ndi.convolve(sk.astype(np.uint8), kernel, mode="constant")
        ends = np.column_stack(np.nonzero(sk & (deg == 1)))
        if ends.size == 0:
            break

        H, W = m.shape
        added = np.zeros_like(m, dtype=bool)

        # simple O(N^2) search within radius (images are modest)
        for i in range(len(ends)):
            r1, c1 = ends[i]
            for j in range(i + 1, len(ends)):
                r2, c2 = ends[j]
                dr, dc = r2 - r1, c2 - c1
                if dr * dr + dc * dc > max_d2:
                    continue

                rr, cc = draw.line(int(r1), int(c1), int(r2), int(c2))
                # keep line in bounds
                ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
                rr, cc = rr[ok], cc[ok]
                if rr.size == 0:
                    continue

                mean_ridge = float(np.nanmean(ridge[rr, cc]))
                if min_ridge <= 0 or mean_ridge >= min_ridge:
                    added[rr, cc] = True

        if not added.any():
            break

        if dilate_fp is not None:
            added = morphology.binary_dilation(added, footprint=dilate_fp)
        m |= added

    return m


    return m


# ----------------------------
# Batch runner
# ----------------------------

def process_image(path, out_dir, pixel_size_um, clear_border, min_cell_area_px,
                  hole_fill_area_px, min_measured_area_px, save_overlay=True,
                  overlay_labels=False, label_font_size=0,
                  thin_membranes=False, use_watershed=True, ws_maxima_footprint=21,
                  overlay_colored=False, color_alpha=0.45,
                  bridge_gaps_px=12, bridge_min_ridge=0.18,
                  bridge_iters=1, bridge_dilate_radius=1,
                  debug=False, debug_dir=None):  # NEW
    gray, rgb = load_gray(path)
    membranes, ridge, inv = detect_membranes(gray)
    mem_initial = np.asarray(membranes, dtype=bool)

    # proactively bridge small gaps between wall fragments
    membranes = _bridge_membrane_gaps(
        mem_initial,
        np.asarray(ridge, dtype=float),
        max_dist=int(bridge_gaps_px),          # NEW
        min_ridge=float(bridge_min_ridge),     # NEW
        iterations=int(max(1, bridge_iters)),
        dilate_radius=int(max(0, bridge_dilate_radius)),
    )

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

    if debug:
        dbg_dir = debug_dir or (out_dir / "debug")
        dbg_dir.mkdir(parents=True, exist_ok=True)

        def _save_debug(name: str, arr: Any) -> None:
            path = dbg_dir / f"{stem}_{name}.png"
            if arr.dtype == bool:
                img = (arr.astype(np.uint8) * 255)
            else:
                data = np.asarray(arr, dtype=float)
                data = np.nan_to_num(data, nan=0.0)
                data = np.clip(data, 0.0, 1.0)
                img = util.img_as_ubyte(data)
            io.imsave(path, img)

        _save_debug("gray_eq", gray)
        _save_debug("inv", inv)
        _save_debug("ridge", ridge)
        _save_debug("membranes_prebridge", mem_initial)
        _save_debug("membranes_postbridge", membranes)

    df.to_csv(out_dir / f"{stem}_measurements.csv", index=False)
    if save_overlay:
        ov = overlay_boundaries(rgb, lbl)
        ov_path = out_dir / f"{stem}_overlay.png"
        io.imsave(ov_path, ov)
        if overlay_labels:
            labeled_path = out_dir / f"{stem}_overlay_labeled.png"
            save_labeled_overlay(ov, df, labeled_path, font_size=label_font_size)
        if overlay_colored:  # NEW
            # color fill where adjacent cells get different colors, blended over the base image  # NEW
            # colored = colorize_labels_overlay(lbl, rgb, alpha=color_alpha)  # NEW
            colored = colorize_labels_flat(lbl, rgb, alpha=color_alpha)
            io.imsave(out_dir / f"{stem}_overlay_colored.png", colored)     # NEW

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
    
    ap.add_argument("--overlay-colored", action="store_true",
                    help="Also save a colored overlay where adjacent cells have different colors  # NEW")
    ap.add_argument("--color-alpha", type=float, default=0.45,
                    help="Alpha for colored overlay blending into the image (0–1)  # NEW")


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
    # bridge gap
    ap.add_argument("--bridge-gaps-px", type=int, default=12,
                    help="Max pixel distance to bridge between membrane endpoints  # NEW")
    ap.add_argument("--bridge-min-ridge", type=float, default=0.18,
                    help="Min mean ridge value along the bridge to accept [0..1]  # NEW")
    ap.add_argument("--bridge-iters", type=int, default=1,
                    help="How many bridging passes to run (higher = more aggressive)")
    ap.add_argument("--bridge-dilate-radius", type=int, default=1,
                    help="Radius (in px) to dilate accepted bridges for robustness")


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
            ws_maxima_footprint=args.ws_maxima_footprint,
            overlay_colored=getattr(args, "overlay_colored", False),
            color_alpha=getattr(args, "color_alpha", 0.45),
            bridge_gaps_px=getattr(args, "bridge_gaps_px", 12),
            bridge_min_ridge=getattr(args, "bridge_min_ridge", 0.18),
            bridge_iters=getattr(args, "bridge_iters", 1),
            bridge_dilate_radius=getattr(args, "bridge_dilate_radius", 1),
            debug=getattr(args, "debug", False),
            debug_dir=out_dir / "debug" if getattr(args, "debug", False) else None,
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
