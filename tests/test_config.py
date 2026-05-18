"""Tests for config module."""

from pathlib import Path

from alpha_quat.config import Config


def test_config_reads_token_and_data_dir(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("""
tushare:
  token: "test_token_123"
data:
  dir: "/tmp/test_data"
""")
    config = Config.from_yaml(str(yaml_path))

    assert config.token == "test_token_123"
    assert config.data_dir == Path("/tmp/test_data")


def test_config_default_data_dir(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("""
tushare:
  token: "abc"
""")
    config = Config.from_yaml(str(yaml_path))

    assert config.token == "abc"
    assert config.data_dir == Path("data")


def test_config_missing_token_raises(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("""
data:
  dir: "./data"
""")
    try:
        Config.from_yaml(str(yaml_path))
        assert False, "Expected KeyError"
    except KeyError:
        pass
