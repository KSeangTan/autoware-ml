import numpy as np
import pytest

from autoware_ml.transforms.point_cloud.crop import (
    CenterShift,
    CropBoxInner,
    CropBoxOuter,
    PointsRangeFilter,
    SphereCrop,
)
from autoware_ml.transforms.point_cloud.geometry import RandomRotateTargetAngle
from autoware_ml.transforms.point_cloud.perturbation import RandomShift, RandomStrengthJitter
from autoware_ml.transforms.point_cloud.sampling import (
    ElasticDistortion,
    GridSample,
    PointShuffle,
    RandomDropout,
)


class TestPointCloudTransforms:
    @pytest.fixture
    def point_cloud(self):
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
                [-2.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        return {"points": points}

    def test_crop_box_inner(self, point_cloud):
        transform = CropBoxInner(crop_box=[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
        output = transform(point_cloud)

        expected = np.array(
            [
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
                [-2.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        assert len(output["points"]) == 4
        assert np.allclose(output["points"], expected)

    def test_crop_box_outer(self, point_cloud):
        transform = CropBoxOuter(crop_box=[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
        output = transform(point_cloud)

        expected = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]], dtype=np.float32)
        assert len(output["points"]) == 3
        assert np.allclose(output["points"], expected)

    def test_point_shuffle_keeps_aligned_arrays(self):
        sample = {
            "points": np.arange(12, dtype=np.float32).reshape(4, 3),
            "labels": np.arange(4, dtype=np.int64),
        }

        output = PointShuffle()(sample)

        assert sorted(output["labels"].tolist()) == [0, 1, 2, 3]
        assert output["points"].shape == (4, 3)

    def test_points_range_filter(self):
        sample = {
            "points": np.array(
                [
                    [-1.0, -1.0, -1.0],
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 2.0],
                    [5.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                ],
                dtype=np.float32,
            ),
            "intensity": np.arange(7, dtype=np.float32),
        }

        output = PointsRangeFilter(point_cloud_range=[-1.0, -1.0, -1.0, 2.0, 2.0, 2.0])(sample)

        expected_points = np.array(
            [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32
        )
        assert np.allclose(output["points"], expected_points)
        assert output["intensity"].tolist() == [0.0, 1.0, 6.0]

    def test_points_range_filter_prevents_max_bound_grid_coords(self):
        point_cloud_range = [0.0, 0.0, 0.0, 2.0, 2.0, 2.0]
        sample = {
            "coord": np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.999, 1.999, 1.999],
                    [2.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 2.0],
                ],
                dtype=np.float32,
            )
        }

        ranged = PointsRangeFilter(point_cloud_range=point_cloud_range)(sample)
        output = GridSample(
            grid_size=1.0,
            mode="test",
            keys=("coord",),
            return_grid_coord=True,
            point_cloud_range=point_cloud_range,
        )(ranged)

        assert output["grid_coord"].shape == (2, 3)
        assert np.all(output["grid_coord"] >= 0)
        assert np.all(output["grid_coord"] < 2)

    def test_random_dropout_keeps_point_arrays_aligned(self):
        np.random.seed(0)
        sample = {
            "coord": np.arange(12, dtype=np.float32).reshape(4, 3),
            "strength": np.arange(4, dtype=np.float32).reshape(4, 1),
            "segment": np.arange(4, dtype=np.int64),
        }

        output = RandomDropout(dropout_ratio=0.5, p=1.0)(sample)

        assert output["coord"].shape[0] == 2
        assert output["strength"].shape[0] == 2
        assert output["segment"].shape[0] == 2

    def test_random_dropout_respects_application_probability(self):
        sample = {
            "coord": np.arange(12, dtype=np.float32).reshape(4, 3),
            "strength": np.arange(4, dtype=np.float32).reshape(4, 1),
        }

        output = RandomDropout(dropout_ratio=0.5, p=0.0)(sample)

        assert output["coord"].shape[0] == 4
        assert output["strength"].shape[0] == 4

    def test_random_rotate_target_angle_rotates_boxes_with_points(self):
        sample = {
            "coord": np.array([[2.0, 0.0, 1.0]], dtype=np.float32),
            "gt_boxes": np.array([[2.0, 0.0, 1.0, 4.0, 2.0, 1.5, 0.1, 3.0, 0.0]], dtype=np.float32),
        }

        output = RandomRotateTargetAngle(angle=[0.5], center=[0.0, 0.0, 0.0], p=1.0)(sample)

        box = output["gt_boxes"][0]
        assert np.allclose(output["coord"], [[0.0, 2.0, 1.0]], atol=1e-6)
        assert np.allclose(box[:3], [0.0, 2.0, 1.0], atol=1e-6)
        assert np.allclose(box[3:6], [4.0, 2.0, 1.5])
        assert np.isclose(box[6], 0.1 + 0.5 * np.pi)
        assert np.allclose(box[7:9], [0.0, 3.0], atol=1e-6)

    def test_random_rotate_target_angle_rejects_boxes_off_z_axis(self):
        sample = {
            "coord": np.zeros((1, 3), dtype=np.float32),
            "gt_boxes": np.zeros((1, 9), dtype=np.float32),
        }

        with pytest.raises(ValueError, match="axis='z'"):
            RandomRotateTargetAngle(angle=[0.5], axis="x", p=1.0)(sample)

    def test_random_strength_jitter_stays_normalized_and_monotonic(self):
        np.random.seed(0)
        sample = {"strength": np.linspace(0.0, 1.0, 5, dtype=np.float32).reshape(5, 1)}

        output = RandomStrengthJitter(
            gamma_range=[0.8, 1.25], scale_range=[0.9, 1.1], shift_range=[-0.02, 0.02]
        )(sample)

        strength = output["strength"]
        assert strength.shape == (5, 1)
        assert strength.dtype == np.float32
        assert strength.min() >= 0.0
        assert strength.max() <= 1.0
        assert np.all(np.diff(strength[:, 0]) >= 0.0)

    def test_grid_sample_keeps_arrays_aligned(self):
        sample = {
            "coord": np.array(
                [[0.0, 0.0, 0.0], [0.01, 0.01, 0.01], [1.0, 1.0, 1.0]],
                dtype=np.float32,
            ),
            "strength": np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
            "segment": np.array([10, 11, 12], dtype=np.int64),
        }

        output = GridSample(
            grid_size=0.05,
            mode="train",
            keys=("coord", "strength", "segment"),
            return_grid_coord=True,
        )(sample)

        assert output["coord"].shape[0] == output["strength"].shape[0] == output["segment"].shape[0]
        assert output["grid_coord"].shape[0] == output["coord"].shape[0]

    def test_grid_sample_test_mode_returns_voxels_and_inverse(self):
        sample = {
            "coord": np.array(
                [[0.0, 0.0, 0.0], [0.01, 0.01, 0.01], [1.0, 1.0, 1.0]],
                dtype=np.float32,
            ),
            "strength": np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
            "segment": np.array([10, 11, 12], dtype=np.int64),
        }

        output = GridSample(
            grid_size=0.05,
            hash_type="fnv",
            mode="test",
            keys=("coord", "strength"),
            return_grid_coord=True,
            return_inverse=True,
        )(sample)

        assert isinstance(output, dict)
        assert output["coord"].shape == (2, 3)
        assert output["strength"].shape == (2, 1)
        assert output["grid_coord"].shape == (2, 3)
        assert output["inverse"].shape == (3,)
        assert output["inverse"].max() < output["coord"].shape[0]
        assert output["inverse"][0] == output["inverse"][1]
        assert output["inverse"][0] != output["inverse"][2]

    def test_grid_sample_rejects_unknown_hash_type(self):
        with pytest.raises(ValueError, match="hash_type"):
            GridSample(grid_size=0.05, hash_type="typo", mode="test", keys=("coord",))

    def test_grid_sample_train_mode_selects_representatives_per_voxel(self, monkeypatch):
        sample = {
            "coord": np.array(
                [[0.0, 0.0, 0.0], [0.01, 0.01, 0.01], [1.0, 1.0, 1.0]],
                dtype=np.float32,
            ),
            "segment": np.array([10, 11, 12], dtype=np.int64),
        }
        monkeypatch.setattr(np.random, "random", lambda size: np.array([0.0, 0.75]))

        output = GridSample(
            grid_size=0.05,
            hash_type="fnv",
            mode="train",
            keys=("coord", "segment"),
        )(sample)

        assert sorted(output["segment"].tolist()) == [11, 12]

    def test_sphere_crop_crops_all_point_arrays_consistently(self):
        sample = {
            "coord": np.arange(30, dtype=np.float32).reshape(10, 3),
            "strength": np.arange(10, dtype=np.float32).reshape(10, 1),
            "segment": np.arange(10, dtype=np.int64),
            "grid_coord": np.arange(30, dtype=np.int32).reshape(10, 3),
        }

        output = SphereCrop(point_max=4)(sample)

        assert output["coord"].shape[0] == 4
        assert output["strength"].shape[0] == 4
        assert output["segment"].shape[0] == 4
        assert output["grid_coord"].shape[0] == 4

    def test_sphere_crop_center_mode_is_deterministic(self):
        sample = {
            "coord": np.array(
                [
                    [0.0, 0.0, 0.0],
                    [10.0, 0.0, 0.0],
                    [11.0, 0.0, 0.0],
                    [12.0, 0.0, 0.0],
                    [50.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            "segment": np.arange(5, dtype=np.int64),
        }

        output = SphereCrop(point_max=3, mode="center")(sample)

        assert np.array_equal(output["segment"], np.array([1, 2, 3], dtype=np.int64))

    def test_random_rotate_target_angle_rotates_by_selected_angle(self):
        sample = {"coord": np.array([[1.0, 0.0, 0.0]], dtype=np.float32)}

        np.random.seed(0)
        output = RandomRotateTargetAngle(angle=(0.5,), center=[0.0, 0.0, 0.0], p=1.0)(sample)

        assert np.allclose(
            output["coord"], np.array([[0.0, 1.0, 0.0]], dtype=np.float32), atol=1e-5
        )

    def test_random_rotate_target_angle_respects_probability(self, monkeypatch):
        sample = {"coord": np.array([[1.0, 0.0, 0.0]], dtype=np.float32)}
        monkeypatch.setattr(np.random, "rand", lambda: 0.75)

        output = RandomRotateTargetAngle(angle=(0.5,), center=[0.0, 0.0, 0.0], p=0.5)(sample)

        assert np.allclose(output["coord"], np.array([[1.0, 0.0, 0.0]], dtype=np.float32))

    def test_random_shift_translates_all_points(self):
        sample = {"coord": np.zeros((2, 3), dtype=np.float32)}

        np.random.seed(0)
        output = RandomShift(shift=[0.5, 0.5, 0.5])(sample)

        assert np.allclose(output["coord"][0], output["coord"][1])
        assert not np.allclose(output["coord"][0], np.zeros(3, dtype=np.float32))

    def test_center_shift_can_keep_z_unchanged(self):
        sample = {"coord": np.array([[0.0, 0.0, 1.0], [2.0, 2.0, 3.0]], dtype=np.float32)}

        output = CenterShift(apply_z=False)(sample)

        assert np.allclose(output["coord"][:, 2], np.array([1.0, 3.0], dtype=np.float32))
        assert np.allclose(output["coord"][:, :2].mean(axis=0), np.zeros(2, dtype=np.float32))

    def test_elastic_distortion_preserves_shape(self):
        sample = {"coord": np.random.rand(8, 3).astype(np.float32)}

        output = ElasticDistortion(distortion_params=[[0.2, 0.4]])(sample)

        assert output["coord"].shape == (8, 3)
