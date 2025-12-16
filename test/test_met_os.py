from __future__ import annotations

from pathlib import Path

import pytest

import metai.utils.met_config as met_config
import metai.utils.met_os as met_os


def _mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def test_scan_directory_level_le_0_returns_self_or_basename(tmp_path: Path) -> None:
    root = _mkdir(tmp_path / "root")

    # return_full_path=True: returns a list containing the input path itself
    assert met_os.scan_directory(str(root), level=0, return_full_path=True) == [str(root)]
    assert met_os.scan_directory(str(root), level=-1, return_full_path=True) == [str(root)]

    # return_full_path=False: returns basename string (not a list)
    assert met_os.scan_directory(str(root), level=0, return_full_path=False) == "root"
    assert met_os.scan_directory(str(root), level=-1, return_full_path=False) == "root"


def test_scan_directory_level_1_returns_sorted_direct_subdirs(tmp_path: Path) -> None:
    root = _mkdir(tmp_path / "root")
    _mkdir(root / "b")
    _mkdir(root / "a")
    (root / "not_a_dir.txt").write_text("x", encoding="utf-8")

    # names
    assert met_os.scan_directory(str(root), level=1, return_full_path=False) == ["a", "b"]

    # full paths
    assert met_os.scan_directory(str(root), level=1, return_full_path=True) == [
        str(root / "a"),
        str(root / "b"),
    ]


def test_scan_directory_level_2_returns_second_level_only_flattened(tmp_path: Path) -> None:
    """
    level=2 should return "grandchildren" only (exactly depth 2), not include depth 1.
    """
    root = _mkdir(tmp_path / "root")
    a = _mkdir(root / "a")
    b = _mkdir(root / "b")
    _mkdir(a / "a2")
    _mkdir(a / "a1")
    _mkdir(b / "b1")

    # names (sorted, flattened across first-level dirs)
    assert met_os.scan_directory(str(root), level=2, return_full_path=False) == ["a1", "a2", "b1"]

    # full paths (sorted, flattened)
    assert met_os.scan_directory(str(root), level=2, return_full_path=True) == [
        str(a / "a1"),
        str(a / "a2"),
        str(b / "b1"),
    ]


def test_scan_directory_missing_or_unreadable_returns_empty_list(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    assert met_os.scan_directory(str(missing), level=1, return_full_path=False) == []
    assert met_os.scan_directory(str(missing), level=2, return_full_path=True) == []


def test_scan_directory_repo_config_root_path_folders_if_present() -> None:
    """
    Integration test: scan folders under configured root_path.

    This test is skipped when root_path doesn't exist on the machine running tests.
    """
    repo_root = Path(__file__).resolve().parents[1]
    cfg = met_config.MetConfig.load(repo_root / "metai" / "config.yaml")
    root = Path(cfg.root_path)

    if not root.exists() or not root.is_dir():
        pytest.skip(f"root_path does not exist or is not a directory: {root}")

    full = met_os.scan_directory(str(root), level=1, return_full_path=True)
    assert isinstance(full, list)

    # If there are no subdirs, that's OK; but if there are, they must all be directories.
    for p in full:
        pp = Path(p)
        assert pp.is_dir()
        # ensure returned paths are under root
        assert str(pp).startswith(str(root))

    names = met_os.scan_directory(str(root), level=1, return_full_path=False)
    assert isinstance(names, list)
    assert sorted(names) == names


