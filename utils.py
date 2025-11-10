from time import time
from typing import Tuple
import numpy as np
import cv2


def chrono(fn):
    def wrapper(*args):
        before = time()
        result = fn(*args)
        print(fn.__name__, ":", time() - before, "s")
        return result

    return wrapper


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
