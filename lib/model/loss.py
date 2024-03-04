import torch
import torch.nn as nn
import numpy as np
from numpy import ndarray
import torch.nn.functional as F
from torch import Tensor


# Numpy-based errors

def mpjpe(predicted: ndarray, target: ndarray) -> ndarray:
    """Mean per-joint position error (i.e. mean Euclidean distance), often referred to as "Protocol #1" in many papers.
    :param predicted: The predicted pose. Shape: (V?, B?, S?, J, D).
    :param target: The target pose. Shape: (V?, B?, S?, J, D).
    :return: The MPJPE. Shape: (1,).
    """
    assert predicted.shape == target.shape
    return np.linalg.norm(predicted - target, axis=-1).mean()


def p_mpjpe(predicted_batch: ndarray, target_batch: ndarray) -> ndarray:
    """Pose error: MPJPE after rigid alignment (scale, rotation, and translation), often referred to as "Protocol #2" in
    many papers.
    :param predicted_batch: The predicted pose. Shape: (B?, V?, S, J, D).
    :param target_batch: The target pose. Shape: (B?, V?, S, J, D).
    :return: The pose error. Shape: (1,).
    """
    assert predicted_batch.shape == target_batch.shape

    target_frame_centroid = target_batch.mean(axis=-2, keepdims=True)
    prediction_frame_centroid = predicted_batch.mean(axis=-2, keepdims=True)

    translated_target = target_batch - target_frame_centroid
    translated_prediction = predicted_batch - prediction_frame_centroid
    norm_target = np.linalg.norm(translated_target, axis=(-1, -2), keepdims=True)
    norm_prediction = np.linalg.norm(translated_prediction, axis=(-1, -2), keepdims=True)
    translated_target /= norm_target
    translated_prediction /= norm_prediction

    H = translated_target.swapaxes(-1, -2) @ translated_prediction
    U, s, Vt = np.linalg.svd(H)
    V = Vt.swapaxes(-1, -2)
    R = V @ U.swapaxes(-1, -2)
    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.linalg.det(R))
    V[..., -1] *= sign_detR[..., None]
    s[..., -1] *= sign_detR
    R = V @ U.swapaxes(-1, -2)

    tr = s.sum(axis=-1, keepdims=True)[..., None, :]
    a = tr * norm_target / norm_prediction  # Scale
    t = target_frame_centroid - a * (prediction_frame_centroid @ R)  # Translation
    # Perform rigid transformation on the input
    predicted_aligned = a * (predicted_batch @ R) + t

    return mpjpe(predicted_aligned, target_batch)


# PyTorch-based errors (for losses)

def loss_mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return (predicted - target).norm(dim=-1).mean()


def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape)-1))


def loss_2d_weighted(predicted, target, conf):
    assert predicted.shape == target.shape
    predicted_2d = predicted[:,:,:,:2]
    target_2d = target[:,:,:,:2]
    diff = (predicted_2d - target_2d) * conf
    return torch.mean(torch.norm(diff, dim=-1))


def n_mpjpe(predicted: Tensor, target: Tensor) -> Tensor:
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py

    :param predicted: The predicted pose. Shape: (B?, V?, S?, J, D).
    :param target: The target pose. Shape: (B?, V?, S?, J, D).
    :return: The normalized MPJPE. Shape: (B?, V?, S?, J).
    """
    assert predicted.shape == target.shape
    norm_predicted = predicted.square().sum(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
    norm_target = (target * predicted).sum(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
    scale = norm_target / norm_predicted
    return loss_mpjpe(scale * predicted, target)


def get_limb_lengths(pose: Tensor) -> Tensor:
    """Computes the lengths of the limbs of a 3D pose in Human3.6M format.

    :param pose: The h36m pose to compute the limb lengths for. Shape: (B?, V?, S?, 17, 3)
    :return: The limb lengths. Shape: (B?, V?, S?, 16)
    """
    joint_connections = [  # TODO: This is also used below. Use it again and it should be extracted.
        (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9),
        (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16),
    ]
    limb_endpoints = pose[..., joint_connections, :]
    limb_vectors = limb_endpoints.diff(dim=-2).squeeze(-2)
    limb_lengths = limb_vectors.norm(dim=-1)
    return limb_lengths


def loss_limb_var(x: Tensor) -> Tensor:
    """Calculate the variance of limb lengths

    :param x: The 3D H3.6M skeleton, shape (B?, V?, S?, 17, 3)
    :return: The variance of the limb lengths, shape (B?, V?, S?, 16)
    """
    if x.shape[-3] <= 1:
        return torch.FloatTensor(1).fill_(0.0)[0].to(x.device)
    limb_lens = get_limb_lengths(x)
    return limb_lens.var(dim=-2).mean()


def loss_limb_gt(x: Tensor, gt: Tensor) -> Tensor:
    """
    Input: (N, T, 17, 3), (N, T, 17, 3)
    """
    limb_lens_x = get_limb_lengths(x)
    limb_lens_gt = get_limb_lengths(gt)  # (N, T, 16)
    return nn.functional.l1_loss(limb_lens_x, limb_lens_gt)


def loss_velocity(predicted: Tensor, target: Tensor) -> Tensor:
    """Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative (i.e. the difference between
    consecutive frames)).
    :param predicted: The predicted pose. Shape: (B?, V?, S, J, D).
    :param target: The target pose. Shape: (B?, V?, S, J, D).
    :return: The mean per-joint velocity error. Shape: (1,).
    """
    assert predicted.shape == target.shape
    if predicted.shape[-3] <= 1:
        return torch.FloatTensor(1).fill_(0.0)[0].to(predicted.device)
    return (predicted.diff(dim=-3) - target.diff(dim=-3)).norm(dim=-1).mean()


def loss_angle(x: Tensor, gt: Tensor) -> Tensor:
    """Calculates the l1 loss of the limb angles of two poses.

    :param x: The predicted pose. Shape: (B?, V?, S?, 17, 3).
    :param gt: The target pose. Shape: (B?, V?, S?, 17, 3).
    :return: The l1 loss of the limb angles. Shape: (1,).
    """
    limb_angles_x = limb_angles(x)
    limb_angles_gt = limb_angles(gt)
    return nn.functional.l1_loss(limb_angles_x, limb_angles_gt)


def loss_angle_velocity(x: Tensor, gt: Tensor) -> Tensor:
    """Mean per-angle velocity error (i.e. mean Euclidean distance of the 1st derivative)

    :param x: The predicted pose. Shape: (V?, B?, S, 17, 3).
    :param gt: The target pose. Shape: (V?, B?, S, 17, 3).
    :return: The mean per-angle velocity error. Shape: (1,).
    """
    assert x.shape == gt.shape
    if x.shape[-3] <= 1:
        return torch.FloatTensor(1).fill_(0.0)[0].to(x.device)
    x_a = limb_angles(x)
    gt_a = limb_angles(gt)
    x_av = x_a.diff(dim=-2)
    gt_av = gt_a.diff(dim=-2)
    return nn.functional.l1_loss(x_av, gt_av)


def limb_angles(pose: Tensor) -> Tensor:
    """Computes the angles between the limbs of a 3D pose in Human3.6M format.

    :param pose: The h36m pose to compute the limb angles for. Shape: (V?, B?, S?, 17, 3)
    :return: The limb angles. Shape: (V?, B?, S?, 16)
    """
    joint_connections = [
        (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9),
        (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16),
    ]
    limb_connections = [
        (0, 3), (0, 6), (3, 6), (0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 10), (7, 13),
        (8, 13), (10, 13), (7, 8), (8, 9), (10, 11), (11, 12), (13, 14), (14, 15),
    ]
    limb_endpoints = pose[..., joint_connections, :]
    limb_vectors = limb_endpoints.diff(dim=-2).squeeze(-2)
    limb_pairs = limb_vectors[..., limb_connections, :]
    limb_angle_cos = F.cosine_similarity(limb_pairs[..., 0, :], limb_pairs[..., 1, :], dim=-1)
    eps = 1e-7  # 1e-9 is too small to prevent crash on back-propagation
    return limb_angle_cos.clamp(-1 + eps, 1 - eps).acos()
