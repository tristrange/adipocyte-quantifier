"""Tests for adipocyte_quant pipeline functions.

Usage:
    python3 -m pip install pytest
    python3 -m pytest tests/adipocyte_quant_tests.py

These tests cover core units of the workflow in isolation. They rely on
pytest and numpy/scikit-image being installed in the test environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from skimage import measure, morphology

import adipocyte_quant as aq


def test_load_gray_returns_normalised_float_and_original(monkeypatch: pytest.MonkeyPatch) -> None:
    img = np.linspace(0, 1, 16).reshape(4, 4).astype(np.float32)

    monkeypatch.setattr(aq.io, "imread", lambda _path: img)

    gray, original = aq.load_gray("dummy.png")

    assert gray.dtype == np.float64
    assert original.dtype in (np.float32, np.uint8)
    assert gray.min() >= 0.0 and gray.max() <= 1.0


def test_as_bool_mask_reduces_channels_and_squeezes() -> None:
    arr = np.zeros((2, 2, 3), dtype=bool)
    arr[0, 0, 1] = True
    mask = aq.as_bool_mask(arr)
    assert mask.shape == (2, 2)
    assert mask.dtype == bool
    assert mask[0, 0]


def test_ensure_labels_2d_raises_on_invalid_shape() -> None:
    with pytest.raises(ValueError):
        aq.ensure_labels_2d(np.zeros((2, 2, 2), dtype=int))


def test_segment_cells_from_membranes_watershed() -> None:
    membranes = np.zeros((20, 20), dtype=bool)
    membranes[5:15, 5] = True
    membranes[5:15, 14] = True
    membranes[5, 5:15] = True
    membranes[14, 5:15] = True
    lbl = aq.segment_cells_from_membranes(
        membranes=membranes,
        clear_border=False,
        min_cell_area_px=5,
        hole_fill_area_px=5,
        use_watershed=True,
        ws_maxima_footprint=5,
    )
    assert lbl.max() >= 1


def test_overlay_boundaries_colorises() -> None:
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    lbl = np.zeros((4, 4), dtype=int)
    lbl[1:3, 1:3] = 1
    out = aq.overlay_boundaries(rgb, lbl)
    assert out.dtype == np.uint8
    assert (out[..., 0] >= out[..., 1]).all()


def test_measure_cells_filters_small_regions() -> None:
    lbl = np.zeros((5, 5), dtype=int)
    lbl[0:2, 0:2] = 1  # area 4
    lbl[3:4, 3:4] = 2  # area 1
    df = aq.measure_cells(lbl, pixel_size_um=1.0, min_measured_area_px=3)
    assert set(df["label"]) == {1}


def test_bridge_membrane_gaps_connects_close_endpoints() -> None:
    membranes = np.zeros((12, 12), dtype=bool)
    membranes[2:6, 6] = True
    membranes[7:11, 6] = True
    ridge = np.ones_like(membranes, dtype=float)
    bridged = aq._bridge_membrane_gaps(membranes, ridge, max_dist=5, min_ridge=0.0)
    assert bridged.sum() > membranes.sum()


def test_colorize_labels_flat_blends_with_base() -> None:
    lbl = np.zeros((4, 4), dtype=int)
    lbl[1:3, 1:3] = 1
    base = np.zeros((4, 4, 3), dtype=np.uint8)
    out = aq.colorize_labels_flat(lbl, base_img=base, alpha=0.5)
    assert out.shape == (4, 4, 3)
    assert out.dtype == np.uint8


def test_process_image_creates_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gray = np.zeros((32, 32), dtype=np.float64)
    gray[8:24, 8:24] = 1.0
    rgb = np.stack([gray] * 3, axis=-1)
    in_dir = tmp_path / "input"
    out_dir = tmp_path / "output"
    in_dir.mkdir()
    out_dir.mkdir()
    path = in_dir / "sample.png"
    monkeypatch.setattr(aq, "load_gray", lambda _path: (gray, rgb))
    monkeypatch.setattr(aq.io, "imsave", lambda *args, **kwargs: None)

    df, lbl = aq.process_image(
        path=path,
        out_dir=out_dir,
        pixel_size_um=0.33,
        clear_border=False,
        min_cell_area_px=50,
        hole_fill_area_px=200,
        min_measured_area_px=0,
        save_overlay=False,
        overlay_colored=False,
        bridge_gaps_px=10,
        bridge_min_ridge=0.0,
        bridge_iters=1,
    )

    assert not df.empty
    assert lbl.max() >= 1
    assert (out_dir / "sample_measurements.csv").exists()
