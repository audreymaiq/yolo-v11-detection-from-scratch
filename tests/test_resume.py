"""A run stopped after an epoch and resumed must end exactly where an uninterrupted run ends.

The tiny dataset has 8 images, so batch=2 gives 4 batches per epoch, and nbs=6 makes accumulate=3: optimizer steps
fall at batches 2, 5, 8, 11 (counted over the whole run), so epochs end with gradients still accumulating and the
first step after a resume depends on when the last one was. Each scenario resumes on a different side of the epoch
where mosaic closes, as each takes a different code path.
"""

import json
from pathlib import Path

import pytest
import torch
import yaml

from yolo11.cli.main import main
from yolo11.data.dataloader import EpochShuffleSampler
from yolo11.training.trainer import DetectionTrainer
from yolo11.utils.torch_utils import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
EPOCHS = 3
SCENARIOS = {  # name: (index of the last epoch before stopping, close_mosaic)
    "before_close_mosaic": (0, 1),  # resume at epoch 1; the resumed run closes mosaic at epoch 2
    "after_close_mosaic": (1, 2),  # mosaic closed at epoch 1; resume at epoch 2 must close it again
}


class Interrupted(Exception):
    pass


def _config(tiny_dataset, out: Path, name: str, close_mosaic: int) -> Path:
    cfg = yaml.safe_load((ROOT / "configs" / "training.yaml").read_text())
    cfg.update(
        data=str(tiny_dataset), save_dir=str(out), name=name, epochs=EPOCHS, batch=2, nbs=6, imgsz=64, workers=0,
        device="cpu", amp=False, cache=False, plots=False, close_mosaic=close_mosaic, save_period=0,
        model="yolov11n", warmup_epochs=0, resume=False, compile=False, seed=7, deterministic=True,
    )
    path = out / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def _final_state(run: Path) -> dict[str, torch.Tensor]:
    model = load_checkpoint(run / "weights" / "last.pt")[0]
    return {k: v.float() for k, v in model.state_dict().items()}


def _results(run: Path) -> list[dict[str, str]]:
    lines = (run / "results.csv").read_text().splitlines()
    keys = lines[0].split(",")
    return [{k: v for k, v in zip(keys, line.split(",")) if k != "time"} for line in lines[1:]]


@pytest.fixture(scope="module", params=SCENARIOS, ids=list(SCENARIOS))
def runs(request, tiny_dataset, tmp_path_factory):
    stop_after, close_mosaic = SCENARIOS[request.param]
    out = tmp_path_factory.mktemp(request.param)
    assert main(["train", "--config", str(_config(tiny_dataset, out, "full", close_mosaic))]) == 0

    save_model = DetectionTrainer.save_model

    def save_then_stop(self):
        save_model(self)
        if self.epoch == stop_after:
            raise Interrupted

    cfg = _config(tiny_dataset, out, "stopped", close_mosaic)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(DetectionTrainer, "save_model", save_then_stop)
        with pytest.raises(Interrupted):
            main(["train", "--config", str(cfg)])
    last = out / "stopped" / "weights" / "last.pt"
    interrupted = torch.load(last, map_location="cpu", weights_only=False)

    assert main(["train", "--config", str(cfg), "--resume", str(last)]) == 0
    return out / "full", out / "stopped", interrupted, stop_after


def test_interrupted_checkpoint_holds_training_state(runs):
    _, _, ckpt, stop_after = runs
    assert ckpt["epoch"] == stop_after
    state = ckpt["train_state"]
    assert state["model"] and state["rng"] and state["stopper"]["best_epoch"] >= 1
    assert state["last_opt_step"] == {0: 2, 1: 5}[stop_after]
    assert state["grads"], "the epoch ended mid-accumulation, so pending gradients must be saved"
    assert all(v.dtype == torch.float32 for v in ckpt["ema"].state_dict().values() if v.is_floating_point())


def test_resumed_run_matches_uninterrupted_run(runs):
    full, resumed, _, _ = runs
    a, b = _final_state(full), _final_state(resumed)
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), f"{k} differs after resuming"
    assert _results(full) == _results(resumed)  # losses, mAP and learning rates of every epoch


def test_resume_restores_trainer_state(runs, tmp_path, monkeypatch):
    """State that may not change this short run's results (patience is never reached) must still come back."""
    closed = []
    close = DetectionTrainer._close_dataloader_mosaic
    monkeypatch.setattr(DetectionTrainer, "_close_dataloader_mosaic", lambda self: closed.append(1) or close(self))
    full, _, ckpt, stop_after = runs
    last = tmp_path / "weights" / "last.pt"
    last.parent.mkdir()
    torch.save(ckpt, last)
    cfg = yaml.safe_load((full.parent / "stopped.yaml").read_text())
    trainer = DetectionTrainer({**cfg, "resume": str(last)})
    trainer._setup_train()

    state = ckpt["train_state"]
    assert trainer.start_epoch == stop_after + 1
    assert (trainer.stopper.best_fitness, trainer.stopper.best_epoch) == (
        state["stopper"]["best_fitness"], state["stopper"]["best_epoch"]
    )
    assert trainer.best_fitness == ckpt["best_fitness"]
    assert trainer.last_opt_step == state["last_opt_step"]
    assert trainer.ema.updates == ckpt["updates"]
    for k, v in trainer.model.state_dict().items():
        assert torch.equal(v, state["model"][k]), k
    assert trainer.optimizer.state_dict()["state"].keys() == ckpt["optimizer"]["state"].keys()
    assert bool(closed) == (stop_after + 1 > EPOCHS - cfg["close_mosaic"])  # == is left to the training loop
    assert trainer.train_loader.sampler.epoch == stop_after + 1


def test_finished_checkpoint_is_stripped(runs):
    full, _, _, _ = runs
    ckpt = torch.load(full / "weights" / "last.pt", map_location="cpu", weights_only=False)
    assert ckpt["train_state"] is None and ckpt["optimizer"] is None and ckpt["epoch"] == -1
    assert next(ckpt["model"].parameters()).dtype == torch.float16


def test_resume_keeps_original_provenance(runs):
    full, resumed, _, _ = runs
    prov = json.loads((resumed / "provenance.json").read_text())
    assert len(prov["resumes"]) == 1
    assert prov["data"]["train"] == prov["resumes"][0]["data"]["train"]
    ckpt = torch.load(resumed / "weights" / "last.pt", map_location="cpu", weights_only=False)
    assert ckpt["provenance"] == prov
    # same data, same fingerprint, whichever run computed it
    assert json.loads((full / "provenance.json").read_text())["data"]["train"] == prov["data"]["train"]


def test_shuffle_order_depends_on_seed_and_epoch_only():
    a, b = EpochShuffleSampler(50, seed=1), EpochShuffleSampler(50, seed=1)
    epochs = [list(a) for _ in range(3)]
    assert all(sorted(e) == list(range(50)) for e in epochs)  # each epoch is a permutation
    assert epochs[0] != epochs[1] != epochs[2]  # reshuffled every epoch
    b.set_epoch(2)
    assert list(b) == epochs[2]  # a fresh sampler set to an epoch replays it
    assert list(EpochShuffleSampler(50, seed=2)) != epochs[0]  # the seed changes the order
