from time import time
from typing import List, Tuple
import numpy as np
import cv2


def chrono(fn):
    def wrapper(*args):
        before = time()
        result = fn(*args)
        print(fn.__name__, ":", time() - before, "s")
        return result

    return wrapper


def parse_source(sources: List[str]):
    return list(map(lambda source: int(source) if source.isnumeric() else source, sources))


def extract_image_patch(image, bbox, patch_shape):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """
    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int64)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    image = cv2.resize(image, tuple(patch_shape[::-1]))
    return image


def euclidean_distance(emb1: np.ndarray, emb2: np.ndarray):
    return np.linalg.norm(emb1 - emb2)


def draw_person_box(f: np.ndarray, bbox: Tuple[int,int,int,int], label: str, color: Tuple[int, int, int]) -> np.ndarray:
    """
    Draw the rectangle and the ids in the image.
    :param bbox:
    :param color: Color of the person
    :param f: Frame to draw onto.
    :param label: Label of the person (ID and nb embeddings)
    :return:
    """
    l, t, r, b = bbox
    l, t, r, b = map(int, [l, t, r, b])

    cv2.rectangle(f, (l, t), (r, b), color, 2)

    (text_width, text_height), _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        2
    )

    cv2.rectangle(
        f,
        (l, t - text_height - 10),
        (l + text_width, t),
        color,
        -1
    )

    cv2.putText(
        f,
        label,
        (l, t - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    return f
