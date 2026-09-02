"""Camera image masking transforms."""

from __future__ import annotations

from jaxtyping import Float32
import numpy as np
import torch
from torch import Tensor
from torchvision.transforms import v2

from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.transforms.multi_task.base import MultiTaskBaseTransform
from autoware_ml.datamodule.multi_task.dataclasses.multi_task_samples import MultiTaskGTSample


class GridMask(MultiTaskBaseTransform):
    """Apply grid masking augmentation to every camera image of a sample.

    A random grid of stripes is zeroed out independently for each camera, so the
    masked region differs from camera to camera. Only pixel values are modified,
    every geometric attribute of `camera_image_data` is carried over unchanged.
    """

    _required_keys = ["camera_image_data"]

    def __init__(
        self,
        probability: float = 0.7,
        ratio: float = 0.5,
        rotate: int = 1,
    ) -> None:
        """Initialize the GridMask transform.

        Args:
            probability: Probability of applying the transform, None to always run.
            ratio: Fraction of each grid period that is masked out.
            rotate: Maximum absolute rotation in degrees applied to the mask.
        """
        super().__init__(probability=probability)
        self.ratio = ratio
        self.rotate = rotate

    def transform(self, multi_task_gt_sample: MultiTaskGTSample) -> MultiTaskGTSample:
        """Mask every camera image with a regular grid pattern.

        Args:
            multi_task_gt_sample: MultiTaskGTSample instance containing `camera_image_data`.

        Returns:
            Updated MultiTaskGTSample instance with a masked `camera_image_data`.
        """
        assert multi_task_gt_sample.camera_image_data is not None
        camera_image_data = multi_task_gt_sample.camera_image_data

        images = camera_image_data.images
        masked_images = torch.stack([image * self._grid_mask(image) for image in images], dim=0)

        # Only the pixel values change, every other field is carried over unchanged.
        # model_copy does not validate what it is given, so the copy is validated explicitly.
        masked_camera_image_data = BaseImages.model_validate(
            camera_image_data.model_copy(update={"images": masked_images})
        )
        return multi_task_gt_sample._replace(camera_image_data=masked_camera_image_data)

    def _grid_mask(
        self, image: Float32[Tensor, "num_channels height width"]
    ) -> Float32[Tensor, "1 height width"]:
        """Sample a grid mask matching the given image.

        Args:
            image: Single camera image the mask is generated for.

        Returns:
            Mask with a leading singleton channel dimension broadcastable over the image,
            holding ``0.0`` on masked pixels and ``1.0`` elsewhere.
        """
        height, width = image.shape[-2:]
        period = int(np.random.randint(32, max(33, min(height, width))))
        cut = max(1, int(period * self.ratio))
        # (1, height, width), the leading dimension broadcasts over the image channels.
        mask = torch.ones((1, height, width), dtype=image.dtype, device=image.device)

        offset_x = int(np.random.randint(period))
        offset_y = int(np.random.randint(period))
        for x in range(offset_x, width, period):
            mask[:, :, x : x + cut] = 0.0
        for y in range(offset_y, height, period):
            mask[:, y : y + cut, :] = 0.0

        if self.rotate > 0:
            angle = float(np.random.uniform(-self.rotate, self.rotate))
            # Corners rotated out of the image are filled with 0.0, hence masked out.
            mask = v2.functional.rotate(mask, angle, interpolation=v2.InterpolationMode.BILINEAR)
        return mask
