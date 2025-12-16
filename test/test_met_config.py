import yaml

import metai.utils.met_config as met_config


def _write_yaml(path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True)


def test_from_file_missing_returns_default(tmp_path):
    missing = tmp_path / "missing.yaml"
    cfg = met_config.MetConfig.from_file(missing)
    assert cfg.root_path == met_config.MetConfig().root_path
    assert cfg.gis_data_path == met_config.MetConfig().gis_data_path
    assert cfg.file_date_format == met_config.MetConfig().file_date_format
    assert cfg.nwp_prefix == met_config.MetConfig().nwp_prefix


def test_from_file_applies_known_keys_and_ignores_unknown(tmp_path, capsys):
    p = tmp_path / "config.yaml"
    _write_yaml(
        p,
        {
            "root_path": "/tmp/root",
            "nwp_prefix": "XYZ",
            "unknown_key": 123,
        },
    )

    cfg = met_config.MetConfig.from_file(p)
    out = capsys.readouterr().out

    assert cfg.root_path == "/tmp/root"
    assert cfg.nwp_prefix == "XYZ"
    assert "未知配置项" in out


def test_save_then_from_file_roundtrip(tmp_path):
    p = tmp_path / "config.yaml"
    src = met_config.MetConfig(
        root_path="/tmp/root",
        gis_data_path="/tmp/gis",
        file_date_format="%Y%m%d-%H%M",
        nwp_prefix="ABC",
    )
    assert src.save(p) is True

    dst = met_config.MetConfig.from_file(p)
    assert dst.root_path == src.root_path
    assert dst.gis_data_path == src.gis_data_path
    assert dst.file_date_format == src.file_date_format
    assert dst.nwp_prefix == src.nwp_prefix


def test_load_defaults_to_repo_metai_config_yaml(monkeypatch):
    """
    仓库默认配置文件在 `metai/config.yaml`，`MetConfig.load()` 在 repo root 下应能自动发现它。
    同时做“内容监测”：逐字段与 YAML 内容一致，并且 validate() 通过。
    """
    from pathlib import Path
    from unittest import SkipTest

    repo_root = Path(__file__).resolve().parents[1]
    root_level = repo_root / "config.yaml"
    if root_level.exists():
        raise SkipTest("repo root 存在 config.yaml，会覆盖 metai/config.yaml 的默认查找优先级")

    config_path = repo_root / "metai" / "config.yaml"
    assert config_path.exists(), f"missing repo config: {config_path}"

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    monkeypatch.chdir(repo_root)
    cfg = met_config.MetConfig.load()

    # 内容监测：字段一致（只比对 MetConfig 已知字段，避免 YAML 未来加新字段导致测试脆弱）
    for k in ("root_path", "gis_data_path", "file_date_format", "nwp_prefix"):
        assert getattr(cfg, k) == data.get(k)

    assert cfg.validate() is True


def test_get_config_caches_global_instance(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_yaml(tmp_path / "config.yaml", {"root_path": "/tmp/v1"})

    met_config._config = None
    cfg1 = met_config.get_config()
    assert cfg1.root_path == "/tmp/v1"

    # Modify config.yaml, but get_config() without config_path should return cached instance.
    _write_yaml(tmp_path / "config.yaml", {"root_path": "/tmp/v2"})
    cfg2 = met_config.get_config()
    assert cfg2 is cfg1
    assert cfg2.root_path == "/tmp/v1"


def test_get_config_invalid_config_falls_back_to_default(tmp_path):
    p = tmp_path / "bad.yaml"
    _write_yaml(p, {"root_path": ""})

    met_config._config = None
    cfg = met_config.get_config(p)
    assert cfg.root_path == met_config.MetConfig().root_path


def test_reload_config_forces_refresh(tmp_path):
    p1 = tmp_path / "c1.yaml"
    p2 = tmp_path / "c2.yaml"
    _write_yaml(p1, {"root_path": "/tmp/one"})
    _write_yaml(p2, {"root_path": "/tmp/two"})

    met_config._config = None
    c1 = met_config.reload_config(p1)
    assert c1.root_path == "/tmp/one"

    c2 = met_config.reload_config(p2)
    assert c2.root_path == "/tmp/two"


def test_create_default_config_writes_yaml(tmp_path):
    p = tmp_path / "config.yaml"
    assert met_config.create_default_config(p) is True
    assert p.exists()

    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert isinstance(data, dict)
    assert "root_path" in data
    assert "gis_data_path" in data
    assert "file_date_format" in data
    assert "nwp_prefix" in data


