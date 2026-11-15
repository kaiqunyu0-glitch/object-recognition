"""Stop rules for the Kaggle COCO notebook. No training and no downloads."""

from __future__ import annotations

import os
from types import SimpleNamespace

from src.detect.coco_early_stop import (
    AcceptableResultStop,
    EpochObservation,
    PlateauState,
    StopConfig,
    advance_plateau,
    apply_stop,
    decide_stop,
    read_map5095,
    read_val_losses,
    total_val_loss,
)


def _config(**kwargs) -> StopConfig:
    base = dict(loss_min_delta=0.01, loss_patience=2, map_target=None, time_budget_hours=None)
    base.update(kwargs)
    return StopConfig(**base)


def _obs(**kwargs) -> EpochObservation:
    base = dict(map5095=0.10, val_box_loss=1.0, val_cls_loss=1.0, val_dfl_loss=1.0, elapsed_seconds=60.0)
    base.update(kwargs)
    return EpochObservation(**base)


def test_total_val_loss_is_the_sum_of_box_cls_and_dfl():
    assert total_val_loss(1.5, 0.25, 0.25) == 2.0


def test_first_loss_is_a_baseline_and_not_a_stall():
    state = advance_plateau(PlateauState(), 3.0, min_delta=0.01)
    assert state.best_loss == 3.0
    assert state.stall_epochs == 0


def test_a_real_drop_resets_the_stall_count():
    state = PlateauState(best_loss=3.0, stall_epochs=4)
    state = advance_plateau(state, 2.9, min_delta=0.01)
    assert state.best_loss == 2.9
    assert state.stall_epochs == 0


def test_a_drop_smaller_than_the_delta_counts_as_a_stall():
    state = advance_plateau(PlateauState(best_loss=3.0, stall_epochs=0), 2.995, min_delta=0.01)
    assert state.best_loss == 3.0
    assert state.stall_epochs == 1


def test_unknown_loss_does_not_move_the_plateau():
    start = PlateauState(best_loss=3.0, stall_epochs=1)
    assert advance_plateau(start, None, min_delta=0.01) == start


def test_loss_plateau_stops_after_n_consecutive_epochs():
    config = _config(loss_patience=2, map_target=None, time_budget_hours=None)
    state = PlateauState()
    state = decide_stop(state, _obs(val_box_loss=2.0, val_cls_loss=0.0, val_dfl_loss=0.0), config).state
    first = decide_stop(state, _obs(val_box_loss=1.995, val_cls_loss=0.0, val_dfl_loss=0.0), config)
    assert first.stop is False
    second = decide_stop(first.state, _obs(val_box_loss=1.994, val_cls_loss=0.0, val_dfl_loss=0.0), config)
    assert second.stop is True
    assert second.reason == "loss_plateau"
    assert second.state.stall_epochs == 2


def test_default_map_target_is_off():
    assert StopConfig().map_target is None
    decision = decide_stop(
        PlateauState(),
        _obs(map5095=0.99),
        StopConfig(time_budget_hours=None),
    )
    assert decision.stop is False


def test_reported_map_05219_stops_only_when_the_target_is_at_or_below_it():
    """Epoch 5 on Kaggle logged mAP50-95=0.5219 and reason 'map'.

    The chess notebook had MAP_TARGET = 0.50, and this check runs before the
    loss plateau. 0.5219 is below 0.55 and is not a one-epoch loss stall.
    """
    observation = _obs(map5095=0.5219, val_box_loss=2.1, val_cls_loss=0.0, val_dfl_loss=0.0)
    baseline = decide_stop(
        PlateauState(),
        _obs(map5095=0.40, val_box_loss=2.0, val_cls_loss=0.0, val_dfl_loss=0.0),
        StopConfig(map_target=0.50, loss_patience=15, time_budget_hours=None),
    )
    fired = decide_stop(
        baseline.state,
        observation,
        StopConfig(map_target=0.50, loss_patience=15, time_budget_hours=None),
    )
    assert fired.state.stall_epochs == 1
    assert fired.stop is True
    assert fired.reason == "map"

    for target in (None, 0.55):
        quiet = decide_stop(
            baseline.state,
            observation,
            StopConfig(map_target=target, loss_patience=15, time_budget_hours=None),
        )
        assert quiet.stop is False
        assert quiet.state.stall_epochs == 1


def test_map_target_stops_only_at_or_above_the_number():
    config = _config(map_target=0.55)
    assert decide_stop(PlateauState(), _obs(map5095=0.5499), config).stop is False
    decision = decide_stop(PlateauState(), _obs(map5095=0.55), config)
    assert decision.stop is True
    assert decision.reason == "map"


def test_missing_map_and_loss_do_not_stop():
    config = _config(time_budget_hours=None)
    observation = EpochObservation(None, None, None, None, elapsed_seconds=10.0)
    decision = decide_stop(PlateauState(), observation, config)
    assert decision.stop is False
    assert decision.state == PlateauState()


def test_one_epoch_loss_spike_and_map_dip_do_not_stop():
    config = StopConfig(loss_min_delta=0.01, loss_patience=15, map_target=None, time_budget_hours=None)
    state = decide_stop(
        PlateauState(),
        _obs(map5095=0.40, val_box_loss=2.0, val_cls_loss=0.0, val_dfl_loss=0.0),
        config,
    ).state
    dipped = decide_stop(
        state,
        _obs(map5095=0.001, val_box_loss=5.0, val_cls_loss=0.0, val_dfl_loss=0.0),
        config,
    )
    assert dipped.stop is False
    assert dipped.state.stall_epochs == 1
    recovered = decide_stop(
        dipped.state,
        _obs(map5095=0.52, val_box_loss=1.5, val_cls_loss=0.0, val_dfl_loss=0.0),
        config,
    )
    assert recovered.stop is False
    assert recovered.state.stall_epochs == 0
    assert recovered.state.best_loss == 1.5


def test_fifteen_non_improving_epochs_stop_on_the_loss():
    config = StopConfig(loss_min_delta=0.01, loss_patience=15, map_target=None, time_budget_hours=None)
    state = decide_stop(
        PlateauState(),
        _obs(val_box_loss=2.0, val_cls_loss=0.0, val_dfl_loss=0.0, map5095=0.2),
        config,
    ).state
    decision = None
    for epoch in range(15):
        decision = decide_stop(
            state,
            _obs(val_box_loss=2.0, val_cls_loss=0.0, val_dfl_loss=0.0, map5095=0.9),
            config,
        )
        state = decision.state
        if epoch < 14:
            assert decision.stop is False
    assert decision is not None
    assert decision.stop is True
    assert decision.reason == "loss_plateau"
    assert decision.state.stall_epochs == 15


def test_time_budget_stops_before_the_other_rules():
    config = _config(time_budget_hours=11.0, map_target=0.99)
    decision = decide_stop(PlateauState(), _obs(elapsed_seconds=11 * 3600), config)
    assert decision.stop is True
    assert decision.reason == "time"


def test_disabled_rules_stay_off():
    config = _config(loss_patience=0, map_target=None, time_budget_hours=None)
    state = PlateauState(best_loss=1.0, stall_epochs=100)
    decision = decide_stop(state, _obs(map5095=0.99, elapsed_seconds=100 * 3600), config)
    assert decision.stop is False


def test_metric_readers_accept_ultralytics_names():
    metrics = {
        "metrics/mAP50-95(B)": 0.42,
        "val/box_loss": 1.0,
        "val/cls_loss": 0.5,
        "val/dfl_loss": 0.25,
    }
    assert read_map5095(metrics) == 0.42
    assert read_val_losses(metrics, None) == (1.0, 0.5, 0.25)
    assert read_val_losses(None, {"box_loss": 1, "cls_loss": 2, "dfl_loss": 3}) == (1.0, 2.0, 3.0)
    assert read_map5095({"metrics/mAP50-95(B)": float("nan")}) is None


def test_apply_stop_caps_epochs_without_clearing_a_previous_stop():
    trainer = SimpleNamespace(stop=False, epoch=4, epochs=300, args=SimpleNamespace(epochs=300))
    apply_stop(trainer, decide_stop(PlateauState(), _obs(map5095=0.2), _config(map_target=0.9, loss_patience=0, time_budget_hours=None)))
    assert trainer.stop is False
    assert trainer.epochs == 300

    decision = decide_stop(PlateauState(), _obs(map5095=0.6), _config(map_target=0.55, time_budget_hours=None))
    apply_stop(trainer, decision)
    assert trainer.stop is True
    assert trainer.epochs == 5
    assert trainer.args.epochs == 5


def test_callback_stops_on_rank_zero_and_ignores_other_ranks(monkeypatch):
    config = StopConfig(loss_min_delta=0.01, loss_patience=15, map_target=None, time_budget_hours=1.0)
    trainer = SimpleNamespace(
        stop=False,
        epoch=0,
        epochs=100,
        args=SimpleNamespace(epochs=100),
        metrics={"metrics/mAP50-95(B)": 0.5219, "val/box_loss": 1.0, "val/cls_loss": 1.0, "val/dfl_loss": 1.0},
        validator=SimpleNamespace(loss={}),
    )
    monkeypatch.setenv("RANK", "1")
    follower = AcceptableResultStop(config, started_at=0.0)
    follower.on_fit_epoch_end(trainer)
    assert trainer.stop is False

    monkeypatch.setenv("RANK", "0")
    leader = AcceptableResultStop(config, started_at=0.0)
    leader.on_fit_epoch_end(trainer)
    assert trainer.stop is True
    assert trainer.epochs == 1
    assert leader.last_decision is not None
    assert leader.last_decision.reason == "time"
    monkeypatch.delenv("RANK", raising=False)
    assert os.environ.get("RANK") is None
