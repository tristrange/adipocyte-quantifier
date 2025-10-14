# Adipocyte Cross-Section Quantification

`adipocyte_quant.py` is a command-line tool that batches membrane-based segmentation and morphometry for adipocyte cross-sections. It enhances faint cell walls, repairs gaps, segments interiors, and collects measurements while saving visual QA overlays.

## Features
- **Robust membrane detection** using adaptive histogram equalisation, ridge filters, and Canny edges.
- **Gap bridging** to connect broken walls via distance-constrained skeleton linking.
- **Watershed-based segmentation** with configurable seed spacing for crowded samples.
- **Comprehensive outputs**: per-image CSV, project-wide summary, boundary overlays, optional coloured + labelled PNGs.
- **Debug mode** that writes intermediate masks for inspection.

## Requirements
- Python 3.9+
- Packages: `numpy`, `pandas`, `scikit-image`, `scipy`, `matplotlib`

Install dependencies into your environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-image scipy matplotlib
```

## Usage

```bash
python adipocyte_quant.py \
  --input /path/to/images \
  --output /path/to/output
```

Only `--input` and `--output` are required. The script walks the input directory for `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff` files.

### Key options

| Option | Default | Purpose |
| --- | --- | --- |
| `--pixel-size-um` | `0.33` | Microns per pixel for physical measurements. |
| `--min-cell-area-px` | `800` | Drop labeled regions smaller than this area. |
| `--hole-fill-area-px` | `2500` | Fill interior voids up to this size before labeling. |
| `--min-measured-area-px` | `1500` | Filter small regions from the measurement CSVs. |
| `--ridge-hi`, `--ridge-lo-mul` | `0.95`, `0.60` | Ridge hysteresis thresholds (higher = conservative). |
| `--canny-sigma`, `--canny-low`, `--canny-high` | `1.6`, `0.05`, `0.15` | Edge detection controls. |
| `--mem-close`, `--mem-dilate` | `4`, `2` | Morphological cleanup on membranes. |
| `--use-watershed` / `--ws-maxima-footprint` | enabled, `10` | Enable watershed and control seed spacing. |
| `--bridge-gaps-px` | `55` | Maximum gap distance (pixels) to bridge between endpoints. |
| `--bridge-min-ridge` | `0.0` | Minimum mean ridge strength required for a bridge. |
| `--bridge-iters`, `--bridge-dilate-radius` | `5`, `3` | Number of bridging passes and dilation radius. |
| `--overlay-labels`, `--overlay-colored` | enabled | Save label number PNG and coloured overlay. |
| `--color-alpha` | `0.45` | Blend ratio for coloured overlays. |
| `--debug` | disabled | Write intermediate arrays (ridge, membranes, etc.) to `output/debug/`. |

All options can be overridden on the command line. To disable overlay labels or coloured overlays, pass `--no-overlay-labels` or `--no-overlay-colored`. Use `--no-watershed` to fall back to connected components.

### Example tuned run

```bash
python adipocyte_quant.py \
  --input ./test \
  --output ./out \
  --ridge-hi 0.95 --ridge-lo-mul 0.60 \
  --canny-low 0.05 --canny-high 0.15 \
  --mem-close 4 --mem-dilate 2 \
  --bridge-gaps-px 55 --bridge-iters 5 --bridge-dilate-radius 3 \
  --ws-maxima-footprint 10 \
  --overlay-colored --overlay-labels
```

### Outputs

For each image (`IMG_0161` as an example):
- `out/IMG_0161_measurements.csv` – per-cell region properties.
- `out/IMG_0161_overlay.png` – boundary overlay.
- `out/IMG_0161_overlay_labeled.png` – centroid labels (if enabled).
- `out/IMG_0161_overlay_colored.png` – coloured cell fill (if enabled).

Batch CSV files:
- `out/all_measurements.csv` – concatenated per-region data.
- `out/summary_by_image.csv` – statistics grouped by image.

With `--debug`, additional PNGs (equalised grayscale, inverted image, ridge response, membranes before/after bridging) are written to `out/debug/`.

## Workflow Tips
- Start with defaults, then adjust `--bridge-*` and membrane thresholds until overlays show closed walls.
- Reduce `--ws-maxima-footprint` when neighbouring cells remain merged.
- Tweak `--ridge-hi`/`--ridge-lo-mul` with caution; lower values accept more ridge noise but help weak membranes.
- Inspect outputs after each tuning pass; the overlays are intended for quick QA.

## License

Add your preferred license here (e.g. MIT) before publishing.
