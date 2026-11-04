import pytest

from yolo11.cli.main import build_parser


@pytest.mark.parametrize("cmd", ["train", "val", "predict", "export", "convert-coco"])
def test_every_command_has_help(cmd, capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([cmd, "--help"])
    assert exc.value.code == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_unknown_command_fails():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["ocr"])
