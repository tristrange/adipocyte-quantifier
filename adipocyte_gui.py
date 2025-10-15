# adipocyte_gui.py
# Cross-platform GUI wrapper around your adipocyte quantification pipeline.
# - Imports process_image from adipocyte_quant.py
# - Folder pickers + checkbox list of images in input folder
# - Tabs for all parameters (with defaults/typical ranges/tooltips)
# - Live progress bar and log window
# - Produces the same CSV + labeled + colored overlays as your CLI

import os
import sys
import traceback
from pathlib import Path
from typing import List, Tuple

import PySimpleGUI as sg

# --- Import your backend ---
# Ensure this file sits next to adipocyte_quant.py
try:
    from adipocyte_quant import process_image
except Exception as e:
    sg.popup_error("Failed to import process_image from adipocyte_quant.py\n\n" + repr(e))
    raise


# ---- Helpers ---------------------------------------------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def list_images(folder: Path) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])


def build_file_checklist(files: List[Path]) -> Tuple[List[List[sg.Element]], List[str]]:
    """
    Return (layout_rows, keys) so we can read which boxes are ticked.
    """
    rows = []
    keys = []
    for i, p in enumerate(files):
        key = f"-FILE-{i}-"
        keys.append(key)
        rows.append([sg.Checkbox(p.name, key=key, default=True, pad=(0, 2))])
    if not rows:
        rows = [[sg.Text("No images found in this folder.", text_color="yellow")]]
    return rows, keys


def int_field(label, key, default, rng, tooltip):
    lo, hi = rng
    return [
        sg.Text(label, size=(28, 1)),
        sg.Spin(
            [i for i in range(lo, hi + 1)],
            initial_value=int(default),
            key=key,
            tooltip=f"{tooltip}  (Typical: {lo}–{hi})",
            size=(8, 1),
        ),
    ]


def float_field(label, key, default, rng, step, tooltip):
    lo, hi = rng
    # Using a Slider for better feel (with a paired input to show exact value)
    return [
        sg.Text(label, size=(28, 1)),
        sg.Slider(
            range=(lo, hi),
            resolution=step,
            default_value=float(default),
            orientation="h",
            key=key,
            size=(30, 15),
            tooltip=f"{tooltip}  (Typical: {lo}–{hi})",
        ),
        sg.Input(
            str(default),
            key=f"{key}-IN",
            size=(6, 1),
            tooltip="Exact value (press Enter to apply)",
            enable_events=True,
        ),
    ]


def checkbox(label, key, default, tooltip):
    return [sg.Checkbox(label, key=key, default=default, tooltip=tooltip)]


def sync_slider_and_input(values, window, slider_key):
    """Mirror a text input into its slider (and vice versa) when Enter is pressed in the input."""
    in_key = f"{slider_key}-IN"
    if in_key in values and values[in_key] is not None:
        try:
            val = float(values[in_key])
            window[slider_key].update(value=val)
        except Exception:
            pass  # ignore invalid user text


# ---- Layout ----------------------------------------------------------------

sg.theme("SystemDefault")

# General tab
general_col = [
    [sg.Text("Input folder:"), sg.Input(key="-IN-FOLDER-", size=(45, 1), enable_events=True),
     sg.FolderBrowse()],
    [sg.Text("Output folder:"), sg.Input(key="-OUT-FOLDER-", size=(45, 1)),
     sg.FolderBrowse()],
    [sg.Button("Refresh file list"), sg.Push(), sg.Button("Select all"), sg.Button("Select none")],
    [sg.Text("Images in input folder:")],
    [sg.Column([[sg.Column([[]], key="-FILE-LIST-COL-", scrollable=True, vertical_scroll_only=True, size=(520, 220))]])],
    [sg.HorizontalSeparator()],
    float_field("Pixel size (µm/pixel)", "-PIXEL-SIZE-", 0.33, (0.01, 5.0), 0.01,
                "Scale for converting pixels to physical units. ↑ = larger reported areas (µm²)."),
    int_field("Min cell area (px)", "-MIN-CELL-AREA-", 800, (0, 20000),
              "Filter out tiny labeled regions (noise). ↑ = fewer, larger cells kept."),
    checkbox("Clear border cells", "-CLEAR-BORDER-", False,
             "Exclude cells touching the image border."),
]

# Membrane detection tab
mem_col = [
    checkbox("Disable top-hat pre-enhancement", "-NO-TOPHAT-", False,
             "Turn off the dark-line pre-boost. Off = more aggressive wall detection."),
    int_field("Top-hat radius (px)", "-TOPHAT-R-", 3, (1, 12),
              "Structural radius for dark-line boost. ↑ = emphasize thicker walls."),
    float_field("Top-hat gain", "-TOPHAT-GAIN-", 0.6, (0.0, 2.0), 0.05,
                "How strongly to blend the top-hat boost. ↑ = more aggressive."),
    float_field("Ridge high (×Otsu)", "-RIDGE-HI-", 0.95, (0.5, 1.5), 0.01,
                "Hysteresis HIGH multiplier on ridge map. ↓ = accept weaker walls (more aggressive)."),
    float_field("Ridge low mul", "-RIDGE-LO-MUL-", 0.60, (0.2, 1.0), 0.01,
                "LOW threshold = HIGH × this. ↑ = less aggressive; ↓ = more aggressive."),
    float_field("Canny sigma", "-CANNY-SIGMA-", 1.6, (0.5, 5.0), 0.1,
                "Gaussian sigma. ↑ = smoother edges, fewer fine details."),
    float_field("Canny low", "-CANNY-LOW-", 0.05, (0.0, 0.5), 0.01,
                "Lower Canny threshold. ↓ = more edges."),
    float_field("Canny high", "-CANNY-HIGH-", 0.15, (0.01, 0.8), 0.01,
                "Upper Canny threshold. ↑ = stricter edges."),
    int_field("Membrane closing (disk)", "-MEM-CLOSE-", 4, (1, 8),
              "binary_closing size. ↑ = bridge more gaps (thicker walls)."),
    int_field("Membrane dilation (disk)", "-MEM-DILATE-", 2, (0, 6),
              "binary_dilation size. ↑ = thicker walls."),
]

# Gap bridging tab
bridge_col = [
    int_field("Bridge gaps up to (px)", "-BRIDGE-GAPS-PX-", 55, (0, 200),
              "Max distance between skeleton endpoints to auto-connect."),
    float_field("Bridge min ridge", "-BRIDGE-MIN-RIDGE-", 0.0, (0.0, 1.0), 0.01,
                "Mean ridge value along bridge. ↑ = stricter, ↓ = more bridges."),
    int_field("Bridge iterations", "-BRIDGE-ITERS-", 5, (0, 25),
              "How many bridging passes. ↑ = more aggressive."),
    int_field("Bridge dilate radius (px)", "-BRIDGE-DILATE-R-", 3, (0, 8),
              "Dilate radius for accepted bridges. ↑ = thicker, robust bridges."),
]

# Watershed & thinning tab
ws_col = [
    checkbox("Use watershed (on = DT watershed, off = connected components)", "-USE-WS-", True,
             "Distance-transform watershed assigns one cell per interior basin."),
    int_field("WS maxima footprint (px)", "-WS-FOOTPRINT-", 10, (3, 61),
              "Minimum spacing between seeds. ↑ = fewer seeds (merge); ↓ = more seeds (split)."),
    checkbox("Thin membranes to 1 px", "-THIN-MEM-", False,
             "Skeletonize walls before segmentation to avoid small enclosed islands."),
    int_field("Hole-fill area (px)", "-HOLE-FILL-", 2500, (0, 20000),
              "Fill small holes in interiors before labeling. ↑ = fewer tiny holes."),
]

# Overlays & output tab
overlay_col = [
    checkbox("Save label overlay (numbers)", "-OVERLAY-LABELS-", True,
             "Overlay cell IDs at centroids on saved image."),
    int_field("Label font size", "-LABEL-FONT-", 0, (0, 64),
              "0 = auto size; ↑ = larger numbers on overlay."),
    checkbox("Save colored overlay", "-OVERLAY-COLORED-", True,
             "Save a colored overlay where each label is colored (cycling palette)."),
    float_field("Colored overlay alpha", "-COLOR-ALPHA-", 0.45, (0.0, 1.0), 0.05,
                "Blend of colors over the base image. 1.0 = solid color map."),
    checkbox("Skip red-line overlay image", "-NO-OVERLAY-", False,
             "If checked, the red boundary overlay image is NOT saved."),
    checkbox("Debug intermediates", "-DEBUG-", False,
             "Save intermediate masks (inverted, ridge, membranes, interiors)."),
    int_field("Min measured area (px)", "-MIN-MEAS-AREA-", 1500, (0, 20000),
              "Exclude labeled regions smaller than this in CSV/overlay."),
]

tabs = [
    [sg.TabGroup([[
        sg.Tab("General", general_col, key="-TAB-GEN-"),
        sg.Tab("Membrane Detection", mem_col, key="-TAB-MEM-"),
        sg.Tab("Gap Bridging", bridge_col, key="-TAB-BRIDGE-"),
        sg.Tab("Watershed & Thinning", ws_col, key="-TAB-WS-"),
        sg.Tab("Overlays & Output", overlay_col, key="-TAB-OVR-"),
    ]], expand_x=True, expand_y=False)]
]

bottom = [
    [sg.HorizontalSeparator()],
    [sg.ProgressBar(100, orientation="h", size=(50, 20), key="-PROG-"),
     sg.Text("Ready.", key="-STATUS-", size=(40, 1))],
    [sg.Multiline(size=(95, 12), key="-LOG-", autoscroll=True, reroute_cprint=True, disabled=True)],
    [sg.Button("Run", bind_return_key=True), sg.Button("Exit")]
]

layout = tabs + bottom

window = sg.Window("Adipocyte Quantification — GUI",
                   layout,
                   finalize=True,
                   resizable=False)


# ---- Dynamic file checklist population -------------------------------------

file_keys: List[str] = []
files: List[Path] = []


def refresh_files():
    global files, file_keys
    in_folder = Path(window["-IN-FOLDER-"].get().strip() or ".")
    files = list_images(in_folder)
    col_layout, file_keys = build_file_checklist(files)
    window["-FILE-LIST-COL-"].update(visible=False)
    window["-FILE-LIST-COL-"].Widget.canvas.master.destroy()  # remove old inner canvas
    window.extend_layout(window["-FILE-LIST-COL-"].ParentRowFrame, col_layout)
    window.refresh()


# ---- Event loop -------------------------------------------------------------

while True:
    event, values = window.read(timeout=200)

    if event in (sg.WIN_CLOSED, "Exit"):
        break

    # Keep slider <-> input text in sync (when user hits Enter in the small input boxes)
    for k in ("-PIXEL-SIZE-", "-TOPHAT-GAIN-", "-RIDGE-HI-", "-RIDGE-LO-MUL-",
              "-CANNY-SIGMA-", "-CANNY-LOW-", "-CANNY-HIGH-", "-BRIDGE-MIN-RIDGE-",
              "-COLOR-ALPHA-"):
        sync_slider_and_input(values, window, k)

    if event in ("-IN-FOLDER-", "Refresh file list"):
        refresh_files()

    if event == "Select all":
        for k in file_keys:
            if k in window.AllKeysDict:
                window[k].update(True)
    if event == "Select none":
        for k in file_keys:
            if k in window.AllKeysDict:
                window[k].update(False)

    if event == "Run":
        # Collect selections
        in_dir = Path(values["-IN-FOLDER-"]).expanduser().resolve()
        out_dir = Path(values["-OUT-FOLDER-"]).expanduser().resolve()
        if not in_dir.exists() or not in_dir.is_dir():
            sg.popup_error("Please choose a valid input folder.")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)

        selected: List[Path] = []
        for i, p in enumerate(files):
            k = f"-FILE-{i}-"
            if values.get(k, False):
                selected.append(p)
        if not selected:
            sg.popup_error("No images selected.")
            continue

        # Read parameters
        def fval(k, default):
            try:
                return float(values[k])
            except Exception:
                try:
                    return float(window[k].Widget.get())
                except Exception:
                    return default

        def ival(k, default):
            try:
                return int(values[k])
            except Exception:
                return default

        params = dict(
            pixel_size_um=fval("-PIXEL-SIZE-", 0.33),
            min_cell_area_px=ival("-MIN-CELL-AREA-", 800),
            clear_border=values["-CLEAR-BORDER-"],
            no_overlay=values["-NO-OVERLAY-"],  # note: we invert when calling
            hole_fill_area_px=ival("-HOLE-FILL-", 2500),
            min_measured_area_px=ival("-MIN-MEAS-AREA-", 1500),
            overlay_labels=values["-OVERLAY-LABELS-"],
            label_font_size=ival("-LABEL-FONT-", 0),
            overlay_colored=values["-OVERLAY-COLORED-"],
            color_alpha=fval("-COLOR-ALPHA-", 0.45),
            no_tophat=values["-NO-TOPHAT-"],
            tophat_radius=ival("-TOPHAT-R-", 3),
            tophat_gain=fval("-TOPHAT-GAIN-", 0.6),
            ridge_hi=fval("-RIDGE-HI-", 0.95),
            ridge_lo_mul=fval("-RIDGE-LO-MUL-", 0.60),
            canny_sigma=fval("-CANNY-SIGMA-", 1.6),
            canny_low=fval("-CANNY-LOW-", 0.05),
            canny_high=fval("-CANNY-HIGH-", 0.15),
            mem_close=ival("-MEM-CLOSE-", 4),
            mem_dilate=ival("-MEM-DILATE-", 2),
            bridge_gaps_px=ival("-BRIDGE-GAPS-PX-", 55),
            bridge_min_ridge=fval("-BRIDGE-MIN-RIDGE-", 0.0),
            bridge_iters=ival("-BRIDGE-ITERS-", 5),
            bridge_dilate_radius=ival("-BRIDGE-DILATE-R-", 3),
            use_watershed=values["-USE-WS-"],
            ws_maxima_footprint=ival("-WS-FOOTPRINT-", 10),
            thin_membranes=values["-THIN-MEM-"],
            debug=values["-DEBUG-"],
        )

        # Log parameters
        sg.cprint("Parameters:")
        for k, v in params.items():
            sg.cprint(f"  {k}: {v}")
        sg.cprint("")

        total = len(selected)
        window["-PROG-"].update_bar(0, total)
        window["-STATUS-"].update(f"Processing 0 / {total} …")
        window["-LOG-"].update(disabled=False)

        # Process
        done = 0
        for idx, img_path in enumerate(selected, start=1):
            try:
                window["-STATUS-"].update(f"Processing {img_path.name} ({idx} / {total}) …")
                sg.cprint(f"→ {img_path.name}")

                # Call your backend for a single image
                _df, _lbl = process_image(
                    path=str(img_path),
                    out_dir=out_dir,
                    pixel_size_um=params["pixel_size_um"],
                    clear_border=params["clear_border"],
                    min_cell_area_px=params["min_cell_area_px"],
                    hole_fill_area_px=params["hole_fill_area_px"],
                    min_measured_area_px=params["min_measured_area_px"],
                    save_overlay=not params["no_overlay"],
                    overlay_labels=params["overlay_labels"],
                    label_font_size=params["label_font_size"],
                    thin_membranes=params["thin_membranes"],
                    use_watershed=params["use_watershed"],
                    ws_maxima_footprint=params["ws_maxima_footprint"],
                    overlay_colored=params["overlay_colored"],
                    color_alpha=params["color_alpha"],
                    bridge_gaps_px=params["bridge_gaps_px"],
                    bridge_min_ridge=params["bridge_min_ridge"],
                )

                done += 1
                window["-PROG-"].update_bar(done)
                window.refresh()

            except Exception as e:
                sg.cprint(f"!! Error on {img_path.name}: {e}", text_color="red")
                sg.cprint(traceback.format_exc(), text_color="red")

        window["-STATUS-"].update(f"Done. Processed {done} / {total} images.")
        window["-LOG-"].update(disabled=True)

window.close()
