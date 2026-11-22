import json
from pathlib import Path

from qecbench.cli import main


def test_toml_cli_override_and_plot(tmp_path):
    config = tmp_path / "quick.toml"
    config.write_text('distances = [3]\nnoise = [0.0]\nshots = 100\ndecoders = ["zero"]\n')
    output = tmp_path / "run"
    assert main(["run", "--config", str(config), "--shots", "17", "--output", str(output)]) == 0
    rows = json.loads((output / "results.json").read_text())
    assert rows[0]["shots"] == 17
    assert (
        main(["run", "--config", str(config), "--shots", "17", "--output", str(output), "--resume"])
        == 0
    )
    figure = tmp_path / "curves.png"
    assert main(["plot", str(output), "--output", str(figure)]) == 0
    assert figure.read_bytes().startswith(b"\x89PNG")


def test_cli_rejects_bad_config_and_reports_cause(tmp_path, capsys):
    assert main(["run", "--shots", "0", "--output", str(tmp_path)]) == 1
    assert "shots must be" in capsys.readouterr().err


def test_shipped_configs_parse():
    import tomllib

    from qecbench import SweepConfig

    for path in (Path(__file__).parents[1] / "configs").glob("*.toml"):
        SweepConfig.from_mapping(tomllib.loads(path.read_text()))
