"""Stop a COCO YOLO run when the result is good enough, or the session is almost over.

Ultralytics already stops on ``patience`` when fitness does not improve. In
Ultralytics 8.4 that fitness is validation mAP50-95 (weight 1 on mAP50-95).
This module adds two more rules, both decided by :func:`decide_stop`:

* validation mAP50-95 reached a target
* total validation loss (box + cls + dfl) improved by less than ``loss_min_delta``
  for ``loss_patience`` epochs in a row

It also stops when the wall-clock budget is used up, so a 12-hour Kaggle
session can save ``best.pt`` and ``last.pt`` instead of being killed.

The decision itself does not touch a trainer. :meth:`AcceptableResultStop.on_fit_epoch_end`
applies it by setting ``trainer.stop`` and capping ``trainer.epochs`` after
that epoch's validation and checkpoint write.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class StopConfig:
    """Thresholds for the extra stop rules. ``patience`` stays a train() argument."""

    loss_min_delta: float = 0.01
    loss_patience: int = 15
    map_target: float | None = 0.55
    time_budget_hours: float | None = 11.0

    def __post_init__(self) -> None:
        if self.loss_min_delta < 0:
            raise ValueError("loss_min_delta must be >= 0")
        if self.loss_patience < 0:
            raise ValueError("loss_patience must be >= 0")
        if self.map_target is not None and not 0.0 <= float(self.map_target) <= 1.0:
            raise ValueError("map_target must be between 0 and 1, or None to disable it")
        if self.time_budget_hours is not None and self.time_budget_hours < 0:
            raise ValueError("time_budget_hours must be >= 0, or None to disable it")


@dataclass(frozen=True)
class PlateauState:
    """Best total validation loss seen so far, and how many epochs failed to beat it."""

    best_loss: float | None = None
    stall_epochs: int = 0


@dataclass(frozen=True)
class EpochObservation:
    """One epoch's validation numbers. Missing values mean that rule is skipped."""

    map5095: float | None
    val_box_loss: float | None
    val_cls_loss: float | None
    val_dfl_loss: float | None
    elapsed_seconds: float


@dataclass(frozen=True)
class StopDecision:
    stop: bool
    reason: str
    state: PlateauState


def finite_number(value: object) -> float | None:
    """Return a finite float, or None when the value is missing or not a number."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def total_val_loss(box: float, cls: float, dfl: float) -> float:
    """Sum the three detection validation losses."""
    return float(box) + float(cls) + float(dfl)


def advance_plateau(state: PlateauState, loss: float | None, min_delta: float) -> PlateauState:
    """Record whether this epoch improved total validation loss.

    The first finite loss is the baseline and is not a stall. A later epoch
    improves only when the loss falls by at least ``min_delta``. A smaller
    drop, a rise, or an equal loss counts as one stall. An unknown loss
    (validation skipped) leaves the counter unchanged.
    """
    if loss is None:
        return state
    if state.best_loss is None:
        return PlateauState(best_loss=loss, stall_epochs=0)
    if state.best_loss - loss >= min_delta:
        return PlateauState(best_loss=loss, stall_epochs=0)
    return PlateauState(best_loss=state.best_loss, stall_epochs=state.stall_epochs + 1)


def decide_stop(state: PlateauState, observation: EpochObservation, config: StopConfig) -> StopDecision:
    """Decide whether this epoch is the last one.

    Order: session time budget, then mAP50-95 target, then the loss plateau.
    The returned state always includes this epoch's plateau update.
    """
    losses = (observation.val_box_loss, observation.val_cls_loss, observation.val_dfl_loss)
    loss = total_val_loss(*losses) if all(item is not None for item in losses) else None
    new_state = advance_plateau(state, loss, config.loss_min_delta)

    budget = config.time_budget_hours
    if budget is not None and budget > 0 and observation.elapsed_seconds >= budget * 3600:
        return StopDecision(True, "time", new_state)

    target = config.map_target
    if target is not None and observation.map5095 is not None and observation.map5095 >= target:
        return StopDecision(True, "map", new_state)

    if config.loss_patience > 0 and new_state.stall_epochs >= config.loss_patience:
        return StopDecision(True, "loss_plateau", new_state)

    return StopDecision(False, "", new_state)


def _lookup(mapping: dict, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in mapping:
            number = finite_number(mapping[key])
            if number is not None:
                return number
    return None


def read_val_losses(metrics: dict | None, validator_loss: dict | None) -> tuple[float | None, float | None, float | None]:
    """Read box, cls, and dfl validation losses from Ultralytics metric dicts."""
    names = ("box_loss", "cls_loss", "dfl_loss")
    sources = (
        (metrics or {}, ("val/{name}", "{name}")),
        (validator_loss or {}, ("{name}", "val/{name}")),
    )
    for source, patterns in sources:
        found = []
        for name in names:
            found.append(_lookup(source, tuple(pattern.format(name=name) for pattern in patterns)))
        if all(item is not None for item in found):
            return found[0], found[1], found[2]
    return None, None, None


def read_map5095(metrics: dict | None) -> float | None:
    """Read validation mAP50-95. Ultralytics names it ``metrics/mAP50-95(B)``."""
    return _lookup(
        metrics or {},
        ("metrics/mAP50-95(B)", "metrics/mAP50-95", "map50-95", "mAP50-95"),
    )


def current_rank() -> int:
    """Ultralytics sets RANK to -1 for one GPU and 0 for the lead DDP process."""
    raw = os.environ.get("RANK", "-1")
    try:
        return int(raw)
    except ValueError:
        return -1


def observation_from_trainer(trainer: object, started_at: float, now: float | None = None) -> EpochObservation:
    """Pull the latest validation metrics off an Ultralytics trainer."""
    metrics = getattr(trainer, "metrics", None)
    validator = getattr(trainer, "validator", None)
    validator_loss = getattr(validator, "loss", None) if validator is not None else None
    box, cls, dfl = read_val_losses(
        metrics if isinstance(metrics, dict) else None,
        validator_loss if isinstance(validator_loss, dict) else None,
    )
    clock = time.time() if now is None else now
    return EpochObservation(
        map5095=read_map5095(metrics if isinstance(metrics, dict) else None),
        val_box_loss=box,
        val_cls_loss=cls,
        val_dfl_loss=dfl,
        elapsed_seconds=clock - started_at,
    )


def apply_stop(trainer: object, decision: StopDecision) -> None:
    """Stop after the current epoch. Validation and ``save_model`` already ran.

    Ultralytics calls ``on_fit_epoch_end`` after that epoch's validation and
    checkpoint write, then breaks the loop when ``trainer.stop`` is true and
    runs a final eval on ``best.pt``. Setting ``epochs`` to the epoch that
    just finished keeps a later check from scheduling another epoch.
    """
    if not decision.stop:
        return
    trainer.stop = True  # type: ignore[attr-defined]
    epoch = int(getattr(trainer, "epoch", 0))
    trainer.epochs = epoch + 1  # type: ignore[attr-defined]
    args = getattr(trainer, "args", None)
    if args is not None and hasattr(args, "epochs"):
        args.epochs = trainer.epochs


class AcceptableResultStop:
    """``on_fit_epoch_end`` callback. Keep one instance for the whole run.

    Multi-GPU: Ultralytics 8.4 pickles callbacks into each DDP worker and,
    after ``on_fit_epoch_end``, broadcasts ``trainer.stop`` from rank 0.
    This callback therefore decides only on rank -1 (one GPU) and rank 0.
    If a future Ultralytics build stops broadcasting that flag, train with
    ``device=0`` (one GPU) or rely on ``patience`` alone.
    """

    def __init__(self, config: StopConfig, started_at: float | None = None) -> None:
        self.config = config
        self.started_at = time.time() if started_at is None else float(started_at)
        self.state = PlateauState()
        self.last_decision: StopDecision | None = None

    def on_fit_epoch_end(self, trainer: object) -> None:
        if current_rank() not in (-1, 0):
            return
        observation = observation_from_trainer(trainer, self.started_at)
        decision = decide_stop(self.state, observation, self.config)
        self.state = decision.state
        self.last_decision = decision
        if not decision.stop:
            return
        apply_stop(trainer, decision)
        print(
            f"Stopping: {decision.reason}. "
            f"mAP50-95={observation.map5095} "
            f"stall_epochs={decision.state.stall_epochs} "
            f"elapsed_hours={observation.elapsed_seconds / 3600:.2f}. "
            "This epoch was already validated and last.pt/best.pt were saved.",
            flush=True,
        )
