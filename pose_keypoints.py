"""Front/back selection shared by raw prediction and tracking entry points."""

import numpy as np

KEYPOINT_NAMES = ("front_with_hp", "back_with_hp", "front_without_hp", "back_without_hp")


def validate_pose_model(model):
    """Accept legacy front/back models or mouse_g3's two mutually exclusive pairs."""
    shape = tuple(getattr(model.model, "kpt_shape", ()))
    if model.task != "pose" or shape not in ((2, 2), (2, 3), (4, 3)):
        raise ValueError("Use a front/back [2,2]/[2,3] or mouse_g3 [4,3] pose model")
    return KEYPOINT_NAMES if shape[0] == 4 else ("front", "back")


def select_pose_pair(keypoints, min_conf=0.5):
    """Choose the valid pair with highest minimum endpoint confidence; ties prefer with_hp."""
    points = np.asarray(keypoints)
    groups = ("with_hp", "without_hp") if len(points) == 4 else ("legacy",)
    scores = (
        {group: float(np.min(points[2 * i : 2 * i + 2, 2])) for i, group in enumerate(groups)}
        if len(points) == 4
        else {}
    )
    candidates = []
    for i, group in enumerate(groups):
        pair = points[2 * i : 2 * i + 2]
        xy = pair[:, :2]
        valid = np.isfinite(xy).all() and (xy >= 0).all() and np.any(xy != 0, axis=1).all()
        valid = valid and np.any(xy[0] != xy[1])
        score = scores.get(group, 1.0)
        if valid and np.isfinite(score) and score >= min_conf:
            candidates.append((score, group, pair))
    if not candidates:
        return None, None, scores
    _, group, pair = max(candidates, key=lambda candidate: candidate[0])
    return group, pair, scores
