# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the augmentation a pipeline draws once per stream."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from autoware_ml.transforms.base import BaseTransform, TransformsCompose, stream_augmentation


class _Draw(BaseTransform):
    """Append ``count`` uniform draws to the sample, as an augmentation reads its parameters."""

    def __init__(self, count: int = 1, probability: float | None = None) -> None:
        super().__init__(probability=probability)
        self.count = count

    def transform(self, model_gt_sample: SimpleNamespace) -> SimpleNamespace:
        model_gt_sample.draws.append(np.random.uniform(size=self.count).tolist())
        return model_gt_sample


def _draws(pipeline: TransformsCompose, seed: int | None) -> list[list[float]]:
    sample = SimpleNamespace(draws=[])
    if seed is None:
        return pipeline(sample).draws
    with stream_augmentation(seed):
        return pipeline(sample).draws


def test_frames_of_one_stream_draw_the_same_augmentation_and_streams_differ() -> None:
    pipeline = TransformsCompose([_Draw(3), _Draw(2)])

    first, second, other = _draws(pipeline, 7), _draws(pipeline, 7), _draws(pipeline, 8)

    assert first == second
    assert first != other
    # The two transforms do not repeat the draws of one another
    assert first[0][:2] != first[1]


def test_frames_outside_a_stream_draw_on_their_own() -> None:
    pipeline = TransformsCompose([_Draw(3)])
    np.random.seed(0)

    assert _draws(pipeline, None) != _draws(pipeline, None)


def test_a_transform_draws_the_same_whatever_the_transforms_before_it_drew() -> None:
    short = TransformsCompose([_Draw(1), _Draw(2)])
    long = TransformsCompose([_Draw(5), _Draw(2)])

    assert _draws(short, 3)[1] == _draws(long, 3)[1]


def test_a_transform_applied_with_a_probability_is_applied_to_the_whole_stream() -> None:
    pipeline = TransformsCompose([_Draw(1, probability=0.5)])

    applied = {seed: len(_draws(pipeline, seed)) for seed in range(40)}

    assert set(applied.values()) == {0, 1}
    assert all(len(_draws(pipeline, seed)) == count for seed, count in applied.items())


def test_the_random_state_outside_the_stream_is_left_as_it_was() -> None:
    pipeline = TransformsCompose([_Draw(4)])
    np.random.seed(11)
    expected = np.random.uniform(size=3).tolist()

    np.random.seed(11)
    _draws(pipeline, 5)

    assert np.random.uniform(size=3).tolist() == expected
