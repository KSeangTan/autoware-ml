import numpy as np
import numpy.typing as npt
from pyquaternion import Quaternion


def convert_quaternion_to_matrix(
    rotation_quaternion: Quaternion,
    translation: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:  # (4, 4)
    """
    Convert a translation and quaternion to a 4x4 transformation matrix.
    Args:
        rotation: Quaternion to represent the rotation.
        translation (3x1 or None): Translation to represent the translation.
    Returns:
        npt.NDArray[np.float64]: 4x4 transformation matrix.
    """
    result = np.eye(4)
    result[:3, :3] = rotation_quaternion.rotation_matrix
    if translation is not None:
        if not isinstance(translation, np.ndarray):
            translation = np.array(translation)

        if translation.shape != (3, 1) or translation.shape != (3,):
            raise ValueError(f"Translation must be a 3x1 array, got shape {translation.shape}")

        result[:3, 3] = translation

    return result.astype(np.float32)
