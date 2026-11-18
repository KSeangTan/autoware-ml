"""Unit tests for optimizer helper utilities."""

from __future__ import annotations

from functools import partial

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import OneCycleLR

from autoware_ml.utils.schedulers.cyclic_cosine_annealing import StepCyclicCosineAnnealingLR
from autoware_ml.utils.optimizer import (
    build_lightning_optimizer_config,
    build_optimizer_param_groups,
    materialize_partial_kwargs,
)


class _ToyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(4, 4)
        self.block = nn.Linear(4, 4)

    def build_optimizer_groups(self) -> dict[str, list[nn.Parameter]]:
        return {
            "default": list(self.stem.parameters()),
            "block": list(self.block.parameters()),
        }


def test_build_optimizer_param_groups_assigns_named_groups() -> None:
    model = _ToyModule()

    param_groups = build_optimizer_param_groups(
        model,
        {"block": {"lr": 1e-4}},
    )

    assert isinstance(param_groups, list)
    assert len(param_groups) == 2
    assert "lr" not in param_groups[0]
    assert param_groups[1]["lr"] == 1e-4
    assert len(param_groups[0]["params"]) == 2
    assert len(param_groups[1]["params"]) == 2


def test_build_optimizer_param_groups_scales_the_base_lr_of_a_group() -> None:
    param_groups = build_optimizer_param_groups(
        _ToyModule(), {"block": {"lr_scale": 0.1, "weight_decay": 0.0}}, base_lr=2e-4
    )

    assert "lr" not in param_groups[0]
    assert param_groups[1]["lr"] == pytest.approx(2e-5)
    assert param_groups[1]["weight_decay"] == 0.0
    assert "lr_scale" not in param_groups[1]


def test_build_optimizer_param_groups_rejects_an_lr_scale_it_cannot_resolve() -> None:
    with pytest.raises(ValueError, match="both lr and lr_scale"):
        build_optimizer_param_groups(
            _ToyModule(), {"block": {"lr": 1e-4, "lr_scale": 0.1}}, base_lr=1e-3
        )
    with pytest.raises(ValueError, match="binds no lr to scale"):
        build_optimizer_param_groups(_ToyModule(), {"block": {"lr_scale": 0.1}})


def test_build_lightning_optimizer_config_scales_a_group_through_the_schedule() -> None:
    optimizers = build_lightning_optimizer_config(
        _ToyModule(),
        optimizer_factory=partial(torch.optim.AdamW, lr=1e-3),
        scheduler_factory=partial(
            StepCyclicCosineAnnealingLR, warmup_epochs=1, decay_epochs=1, max_lr_factor=2.0
        ),
        optimizer_group_overrides=OmegaConf.create({"block": {"lr_scale": 0.1}}),
        estimated_stepping_batches=8,
    )
    optimizer, scheduler = optimizers["optimizer"], optimizers["lr_scheduler"]["scheduler"]

    for _ in range(8):
        default_lr, block_lr = (group["lr"] for group in optimizer.param_groups)
        assert block_lr == pytest.approx(0.1 * default_lr)
        optimizer.step()
        scheduler.step()


def test_build_optimizer_param_groups_rejects_unknown_group_overrides() -> None:
    model = _ToyModule()

    with pytest.raises(ValueError, match="Unknown optimizer group override"):
        build_optimizer_param_groups(model, {"missing": {"lr": 1e-4}})


def test_build_optimizer_param_groups_returns_flat_params_without_overrides() -> None:
    class _DefaultOnlyModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def build_optimizer_groups(self) -> dict[str, list[nn.Parameter]]:
            return {"default": list(self.linear.parameters())}

    model = _DefaultOnlyModule()
    params = build_optimizer_param_groups(model)

    assert isinstance(params, list)
    assert all(isinstance(parameter, nn.Parameter) for parameter in params)
    assert len(params) == 2


def test_materialize_partial_kwargs_converts_omegaconf_containers() -> None:
    value = OmegaConf.create({"max_lr": [1e-3, 1e-4], "betas": [0.9, 0.999]})

    materialized = materialize_partial_kwargs(value)

    assert materialized == {"max_lr": [1e-3, 1e-4], "betas": [0.9, 0.999]}


def test_build_lightning_optimizer_config_infers_total_steps_for_scheduler() -> None:
    model = _ToyModule()

    optimizers = build_lightning_optimizer_config(
        model,
        optimizer_factory=lambda params: torch.optim.AdamW(params, lr=1e-3),
        scheduler_factory=partial(OneCycleLR, max_lr=[1e-3, 1e-4]),
        optimizer_group_overrides={"block": {"lr": 1e-4}},
        scheduler_config={"interval": "step"},
        estimated_stepping_batches=10,
    )

    optimizer = optimizers["optimizer"]
    scheduler = optimizers["lr_scheduler"]["scheduler"]

    assert [group["max_lr"] for group in optimizer.param_groups] == [1e-3, 1e-4]
    assert optimizers["lr_scheduler"]["interval"] == "step"
    assert scheduler.total_steps == 10


def test_build_lightning_optimizer_config_materializes_partial_omegaconf_kwargs() -> None:
    model = _ToyModule()

    optimizers = build_lightning_optimizer_config(
        model,
        optimizer_factory=partial(torch.optim.AdamW, lr=1e-3),
        scheduler_factory=partial(
            OneCycleLR,
            max_lr=OmegaConf.create([1e-3, 1e-4]),
        ),
        optimizer_group_overrides={"block": {"lr": 1e-4}},
        scheduler_config={"interval": "step"},
        estimated_stepping_batches=10,
    )

    optimizer = optimizers["optimizer"]
    scheduler = optimizers["lr_scheduler"]["scheduler"]

    assert [group["max_lr"] for group in optimizer.param_groups] == [1e-3, 1e-4]
    assert scheduler.total_steps == 10


def test_build_lightning_optimizer_config_keeps_scheduler_metadata() -> None:
    model = _ToyModule()

    optimizers = build_lightning_optimizer_config(
        model,
        optimizer_factory=lambda params: torch.optim.AdamW(params, lr=1e-3),
        scheduler_factory=lambda optimizer: OneCycleLR(optimizer, max_lr=1e-3, total_steps=10),
        scheduler_config={"interval": "step", "frequency": 2, "monitor": "val/loss"},
    )

    assert optimizers["lr_scheduler"]["interval"] == "step"
    assert optimizers["lr_scheduler"]["frequency"] == 2
    assert optimizers["lr_scheduler"]["monitor"] == "val/loss"
