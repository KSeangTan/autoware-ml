"""
It is a base class for all image data structures, providing common attributes and methods that can be used by derived classes.
"""

from __future__ import annotations

from abc import ABC
from typing import Sequence

from jaxtyping import Float32
import torch


class BaseImages(ABC):
    """
    Abstract base class for images that defines the common interface.
    Note that it only supports tensor data type for now.
    """

    def __init__(
        self,
        images: Float32[torch.Tensor, "num_cameras num_channels height width"],
        timestamps: Float32[torch.Tensor, " num_cameras"],
        camera_intrinsics: Float32[torch.Tensor, "num_cameras 3 3"],
        camera_names: Sequence[str],
        lidar2images: Float32[torch.Tensor, "num_cameras 4 4"],
        lidar2cams: Float32[torch.Tensor, "num_cameras 4 4"],
        distortion_models: Sequence[str],
        distortion_coefficients: Sequence[Float32[torch.Tensor, " num_coefficients"]],
        noises: Float32[torch.Tensor, " num_cameras"] | None = None,
        augmented_camera_intrinsics: Float32[torch.Tensor, "num_cameras 3 3"] | None = None,
        image_augmentation_matrices: Float32[torch.Tensor, "num_cameras 3 3"] | None = None,
    ) -> None:
        """
        Initialize the BasePoints instance.

        Args:
            images: Images in Tensor to represent images for a sample.
            timestamps: Tensor represents the timestamps for each images.
            camera_intrinstic: Tensor represents camera intristincs for each camera.
            camera_names: Sequence for camera names to represent each image.
            lidar2images: Transformation matrix for lidar to each image.
            lidar2cams: Transformation matrix for lidar to each camera.
            distortion_models: Distortion model name per camera following ROS convention
            (e.g. ``"plumb_bob"``, ``"rational_polynomial"``). Empty string for
            pre-undistorted images.
            distortion_coefficients: Distortion coefficients per camera following the OpenCV
            convention ``(k1, k2, p1, p2[, k3[, ...]])``. The length varies by distortion
            model (4, 5, 8, 12 or 14), hence a sequence of 1D tensors rather than a single
            stacked tensor. An empty tensor for pre-undistorted images.
            noises: Optional homogeneous perturbation transform applied by calibration
            augmentation.
            augmented_camera_intrinsics: Updated intrinsic matrix after image-space transforms.
            When omitted, a copy of ``camera_intrinsics`` is used.
            image_augmentation_matrices: Composition of the 2D affines applied to the pixels
            by the image-space augmentations, mapping a pixel of the raw loaded image onto
            its location in ``images``. When omitted, the identity is used. Note that a
            non-affine image transform, such as the undistortion applied by
            ``UndistortImage``, is not composed into it, so it describes the augmentations
            alone rather than the full raw-image-to-augmented-image mapping.
        """
        self._images = images
        self._timestamps = timestamps
        self._camera_intrinsics = camera_intrinsics
        self._camera_names = camera_names
        self._lidar2images = lidar2images
        self._lidar2cams = lidar2cams
        self._distortion_models = distortion_models
        self._distortion_coefficients = distortion_coefficients
        self._noises = noises
        if augmented_camera_intrinsics is None:
            self._augmented_camera_intrinsics = self._camera_intrinsics.clone()
        else:
            self._augmented_camera_intrinsics = augmented_camera_intrinsics
        if image_augmentation_matrices is None:
            self._image_augmentation_matrices = torch.eye(
                3,
                dtype=self._augmented_camera_intrinsics.dtype,
                device=self._augmented_camera_intrinsics.device,
            ).repeat(self._camera_intrinsics.shape[0], 1, 1)
        else:
            self._image_augmentation_matrices = image_augmentation_matrices

    @property
    def images(self) -> Float32[torch.Tensor, "num_cameras num_channels height width"]:
        """Return images."""
        return self._images

    @property
    def timestamps(self) -> Float32[torch.Tensor, " num_cameras"]:
        """Return timestamps for each images."""
        return self._timestamps

    @property
    def camera_intrinsics(self) -> Float32[torch.Tensor, "num_cameras 3 3"]:
        """Return camera_intrinsics."""
        return self._camera_intrinsics

    @property
    def camera_names(self) -> Sequence[str]:
        """Return camera names."""
        return self._camera_names

    @property
    def lidar2images(self) -> Float32[torch.Tensor, "num_cameras 4 4"]:
        """Return Lidar to each image transformation matrix."""
        return self._lidar2images

    @property
    def lidar2cams(self) -> Float32[torch.Tensor, "num_cameras 4 4"]:
        """Return Lidar to each camera transformation matrix."""
        return self._lidar2cams

    @property
    def distortion_models(self) -> Sequence[str]:
        """Return distortion model for each camera."""
        return self._distortion_models

    @property
    def distortion_coefficients(self) -> Sequence[Float32[torch.Tensor, " num_coefficients"]]:
        """Return distortion coefficients for each camera."""
        return self._distortion_coefficients

    @property
    def noises(self) -> Float32[torch.Tensor, " num_cameras"] | None:
        """Return calibration augmentation noises."""
        return self._noises

    @property
    def augmented_camera_intrinsics(self) -> Float32[torch.Tensor, "num_cameras 3 3"]:
        """Return augmented camera intrinsics after series of image-space transformations."""
        return self._augmented_camera_intrinsics

    @property
    def image_augmentation_matrices(self) -> Float32[torch.Tensor, "num_cameras 3 3"]:
        """Return the composed 2D affine the image-space augmentations applied to the pixels."""
        return self._image_augmentation_matrices

    def update_lidar_transformation_matrices(
        self, augmentation_inverse: Float32[torch.Tensor, "4 4"]
    ) -> None:
        """
        Keep lidar-camera projection consistent after a lidar-space transform.

        Applies ``augmentation_inverse`` (inverse of the 4x4 point-space augmentation) to
        ``lidar2cam`` and recomputes ``lidar2img`` from ``camera_intrinsics``.
        """
        # (num_cameras, 4, 4) @ (4, 4)
        augmented_lidar2cam = self.lidar2cams @ augmentation_inverse
        self._lidar2cams = augmented_lidar2cam

        # (num_cameras, 4, 4), the intrinsics padded to homogeneous coordinates.
        camera_intrinsics = torch.eye(
            4,
            device=self.augmented_camera_intrinsics.device,
            dtype=self.augmented_camera_intrinsics.dtype,
        ).repeat(self.augmented_camera_intrinsics.shape[0], 1, 1)
        camera_intrinsics[:, :3, :3] = self.augmented_camera_intrinsics
        # augmented_intrinsic @ augmented_lidar2cam -> lidar2img
        self._lidar2images = camera_intrinsics @ augmented_lidar2cam
