"""Tests for the IO processing timing callback: renamed step metric, epoch totals,
sanity-check suppression, and console logging at a fixed batch interval."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from autoware_ml.callbacks.io_processing_timer import IOProcessingTimer


def _module() -> MagicMock:
    module = MagicMock()
    module.log = MagicMock()
    return module


def _trainer(sanity_checking: bool = False) -> MagicMock:
    trainer = MagicMock()
    trainer.sanity_checking = sanity_checking
    return trainer


def _batch(io_processing_time: float) -> MagicMock:
    """Build a stand-in for MultiTaskBatchInputs carrying an IO timing."""
    batch = MagicMock(spec=["multi_task_gt_batch"])
    batch.multi_task_gt_batch.io_processing_time = io_processing_time
    return batch


def _logged(module: MagicMock) -> dict[str, float]:
    return {call.args[0]: call.args[1] for call in module.log.call_args_list}


class TestStepMetric:
    def test_batch_io_time_is_logged_under_the_renamed_key(self) -> None:
        callback = IOProcessingTimer()
        module = _module()

        callback.on_train_batch_start(_trainer(), module, batch=_batch(0.5), batch_idx=0)

        logged = _logged(module)
        assert logged["train/io_processing_total_time"] == pytest.approx(0.5)
        assert "train/io_processing_seconds" not in logged

    @pytest.mark.parametrize("stage", ["train", "val", "test"])
    def test_every_stage_is_recorded(self, stage: str) -> None:
        callback = IOProcessingTimer()
        module = _module()
        hook = {
            "train": callback.on_train_batch_start,
            "val": callback.on_validation_batch_start,
            "test": callback.on_test_batch_start,
        }[stage]

        hook(_trainer(), module, batch=_batch(0.25), batch_idx=0)

        assert _logged(module)[f"{stage}/io_processing_total_time"] == pytest.approx(0.25)


class TestEpochSummary:
    def test_epoch_total_mean_and_max(self) -> None:
        callback = IOProcessingTimer()
        module = _module()
        trainer = _trainer()

        for io_time in (1.0, 2.0, 3.0):
            callback.on_train_batch_start(trainer, module, batch=_batch(io_time), batch_idx=0)
        callback.on_train_epoch_end(trainer, module)

        logged = _logged(module)
        assert logged["train/io_processing_total_time_sum"] == pytest.approx(6.0)
        assert logged["train/io_processing_total_time_mean"] == pytest.approx(2.0)
        assert logged["train/io_processing_total_time_max"] == pytest.approx(3.0)

    def test_epoch_end_resets_the_accumulator(self) -> None:
        callback = IOProcessingTimer()
        module = _module()
        trainer = _trainer()

        callback.on_train_batch_start(trainer, module, batch=_batch(10.0), batch_idx=0)
        callback.on_train_epoch_end(trainer, module)
        module.log.reset_mock()

        callback.on_train_batch_start(trainer, module, batch=_batch(1.0), batch_idx=0)
        callback.on_train_epoch_end(trainer, module)

        assert _logged(module)["train/io_processing_total_time_sum"] == pytest.approx(1.0)

    def test_stages_accumulate_independently(self) -> None:
        callback = IOProcessingTimer()
        module = _module()
        trainer = _trainer()

        callback.on_train_batch_start(trainer, module, batch=_batch(4.0), batch_idx=0)
        callback.on_validation_batch_start(trainer, module, batch=_batch(1.0), batch_idx=0)
        callback.on_validation_epoch_end(trainer, module)
        callback.on_train_epoch_end(trainer, module)

        logged = _logged(module)
        assert logged["val/io_processing_total_time_sum"] == pytest.approx(1.0)
        assert logged["train/io_processing_total_time_sum"] == pytest.approx(4.0)

    def test_summary_is_skipped_without_batches(self) -> None:
        callback = IOProcessingTimer()
        module = _module()

        callback.on_test_epoch_end(_trainer(), module)

        assert module.log.call_args_list == []


class TestSanityCheck:
    def test_sanity_check_batches_are_not_recorded(self) -> None:
        callback = IOProcessingTimer()
        module = _module()
        sanity = _trainer(sanity_checking=True)

        callback.on_validation_batch_start(sanity, module, batch=_batch(1.0), batch_idx=0)
        callback.on_validation_epoch_end(sanity, module)

        assert module.log.call_args_list == []


class TestLogInterval:
    def _run_batches(self, callback: IOProcessingTimer, count: int) -> None:
        trainer, module = _trainer(), _module()
        for batch_idx in range(count):
            callback.on_train_batch_start(trainer, module, batch=_batch(0.5), batch_idx=batch_idx)

    def test_every_nth_batch_is_logged(self, caplog) -> None:
        callback = IOProcessingTimer(log_interval=2)

        with caplog.at_level("INFO"):
            self._run_batches(callback, count=5)

        # Batches 2 and 4 of 5 land on the interval.
        lines = [
            record.message for record in caplog.records if "IO processing batch" in record.message
        ]
        assert len(lines) == 2
        assert "batch 2" in lines[0]
        assert "batch 4" in lines[1]

    def test_zero_disables_console_logging(self, caplog) -> None:
        callback = IOProcessingTimer(log_interval=0)

        with caplog.at_level("INFO"):
            self._run_batches(callback, count=5)

        assert [
            record for record in caplog.records if "IO processing batch" in record.message
        ] == []

    def test_interval_counts_per_stage(self, caplog) -> None:
        callback = IOProcessingTimer(log_interval=2)
        trainer, module = _trainer(), _module()

        with caplog.at_level("INFO"):
            # One train batch and one val batch must not add up to the interval.
            callback.on_train_batch_start(trainer, module, batch=_batch(0.5), batch_idx=0)
            callback.on_validation_batch_start(trainer, module, batch=_batch(0.5), batch_idx=0)

        assert [
            record for record in caplog.records if "IO processing batch" in record.message
        ] == []

    def test_negative_interval_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="log_interval"):
            IOProcessingTimer(log_interval=-1)
