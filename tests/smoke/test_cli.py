import json
import subprocess
import sys
from pathlib import Path

from edge_triage.cli import main

ROOT = Path(__file__).resolve().parents[2]


def test_module_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "edge_triage", "--help"], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert all(command in result.stdout for command in ("doctor", "show-config", "sample-run"))


def test_show_config(config, capsys):
    assert main(["show-config", "--config", str(ROOT / "configs/default.yaml")]) == 0
    assert config.config_hash in capsys.readouterr().out


def test_doctor(config, capsys):
    assert main(["doctor", "--config", str(ROOT / "configs/default.yaml")]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_invalid_config_exit_code(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("unknown: true", encoding="utf-8")
    assert main(["show-config", "--config", str(path)]) == 1


def test_malformed_yaml_exit_code(tmp_path):
    path = tmp_path / "malformed.yaml"
    path.write_text("training: [", encoding="utf-8")
    assert main(["doctor", "--config", str(path)]) == 1


def test_sample_run(isolated_config, tmp_path, capsys):
    path = tmp_path / "config.yaml"
    path.write_text(isolated_config.resolved_yaml(), encoding="utf-8")
    assert main(["sample-run", "--config", str(path)]) == 0
    directory = Path(capsys.readouterr().out.strip())
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
