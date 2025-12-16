import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metai.dataset import MetSample
from metai.utils import MetNwp, get_config


SAMPLES_JSONL = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "samples.jsonl")


def _read_first_n_samples(jsonl_path: str, n: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if len(records) >= n:
                break
    return records


def _expected_input_channels(sample: MetSample) -> int:
    """
    MetSample.load_input_data():
    - input_base: (T=in_seq_length, C=len(channels), H, W)
    - if future NWP present and out_seq_length % in_seq_length == 0:
        append folded future NWP: (T=in_seq_length, fold_factor*C_nwp, H, W)
    """
    base_c = len(sample.channels)
    c_nwp = sum(1 for c in sample.channels if isinstance(c, MetNwp))
    if c_nwp <= 0:
        return base_c
    if sample.out_seq_length % sample.in_seq_length != 0:
        return base_c
    # Only appended if timestamps has full future window (checked in test via len(timestamps)==30)
    fold_factor = sample.out_seq_length // sample.in_seq_length
    return base_c + fold_factor * c_nwp


@pytest.mark.parametrize("n", [5])
def test_met_sample_first5_mocked_np_load_shapes_and_types(monkeypatch: pytest.MonkeyPatch, n: int) -> None:
    """
    Fast unit-test: does NOT require the real dataset to exist.
    It validates:
    - MetSample can be built from samples.jsonl records
    - to_numpy(is_train=True) runs end-to-end
    - tensor shapes / dtypes are consistent with the design (10 past, 20 future)
    """
    assert os.path.exists(SAMPLES_JSONL), f"Missing {SAMPLES_JSONL}"
    records = _read_first_n_samples(SAMPLES_JSONL, n)
    assert len(records) == n

    # Make np.load deterministic + cheap, and exercise resize paths:
    # - label/radar: return (301,301)
    # - nwp: return (50,50) so _load_nwp_frame's resize runs
    # - gis: return (301,301)
    rng = np.random.default_rng(123)

    def fake_np_load(path: str, *args: Any, **kwargs: Any) -> np.ndarray:
        basename = os.path.basename(path)
        if "_RRA_" in basename or "_RRA" in basename:
            # NWP files
            return rng.random((50, 50), dtype=np.float32)
        return rng.random((301, 301), dtype=np.float32)

    monkeypatch.setattr(np, "load", fake_np_load, raising=True)

    config = get_config()
    for rec in records:
        sample = MetSample.create(
            sample_id=rec["sample_id"],
            timestamps=rec["timestamps"],
            config=config,
            is_train=True,
        )

        # samples.jsonl appears to be 30 timestamps = 10 past + 20 future
        assert len(sample.timestamps) == sample.in_seq_length + sample.out_seq_length

        metadata, x, y, xmask, ymask = sample.to_numpy()
        assert isinstance(metadata, dict)

        expected_c = _expected_input_channels(sample)
        assert x.shape == (sample.in_seq_length, expected_c, 301, 301)
        assert x.dtype == np.float32

        assert isinstance(xmask, np.ndarray)
        assert xmask.shape == x.shape
        assert xmask.dtype == np.bool_

        assert y is not None
        assert ymask is not None
        assert isinstance(y, np.ndarray)
        assert isinstance(ymask, np.ndarray)
        assert y.shape == (sample.out_seq_length, 1, 301, 301)
        assert y.dtype == np.float32
        assert ymask.shape == y.shape
        assert ymask.dtype == np.bool_

        # Normalization should not produce NaNs.
        assert np.isfinite(x).all()
        assert np.isfinite(y).all()


def test_met_sample_first5_integration_real_files_if_enabled() -> None:
    """
    Integration test (opt-in):
    - Reads the first 5 records from data/samples.jsonl
    - Executes MetSample.to_numpy against real .npy files under MetConfig.root_path

    Enable by setting:
      METAI_INTEGRATION=1
    """
    if os.getenv("METAI_INTEGRATION", "0") != "1":
        pytest.skip("Set METAI_INTEGRATION=1 to run real-file integration test.")

    if not os.path.exists(SAMPLES_JSONL):
        pytest.skip(f"Missing {SAMPLES_JSONL}")

    config = get_config()
    if not os.path.exists(config.root_path):
        pytest.skip(f"Dataset root_path not found: {config.root_path}")

    records = _read_first_n_samples(SAMPLES_JSONL, 5)
    assert len(records) == 5

    # Quick existence check using the first sample's first timestamp label file.
    first = records[0]
    sample0 = MetSample.create(first["sample_id"], first["timestamps"], config=config, is_train=True)
    label_probe = os.path.join(
        sample0.base_path,
        "LABEL",
        "RA",
        f"{sample0.task_id}_Label_RA_{sample0.station_id}_{sample0.timestamps[0]}.npy",
    )
    if not os.path.exists(label_probe):
        pytest.skip(f"Probe file not found (dataset layout mismatch or missing data): {label_probe}")

    for rec in records:
        sample = MetSample.create(rec["sample_id"], rec["timestamps"], config=config, is_train=True)
        metadata, x, y, xmask, ymask = sample.to_numpy()

        expected_c = _expected_input_channels(sample)
        assert x.shape == (sample.in_seq_length, expected_c, 301, 301)
        assert x.dtype == np.float32
        assert isinstance(xmask, np.ndarray)
        assert xmask.shape == x.shape
        assert xmask.dtype == np.bool_

        assert y is not None
        assert ymask is not None
        assert isinstance(y, np.ndarray)
        assert isinstance(ymask, np.ndarray)
        assert y.shape == (sample.out_seq_length, 1, 301, 301)
        assert y.dtype == np.float32
        assert ymask.shape == y.shape
        assert ymask.dtype == np.bool_


