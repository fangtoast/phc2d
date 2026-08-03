"""Objective-driven inverse-design losses.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from typing import Any, Mapping


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    return float(value)


def band_relative_error_loss(
    predictions: Any,
    target: Any,
    *,
    torch: Any,
    n_bands: int | None = None,
    band_start_one_based: int = 2,
    epsilon: float = 1.0e-8,
) -> Any:
    """Return a differentiable mean relative error, defaulting to bands 2..N."""

    pred = predictions.reshape(int(predictions.shape[0]), -1)
    raw_target = target.reshape(int(target.shape[0]), -1) if int(target.ndim) > 1 else target.reshape(1, -1)
    if int(raw_target.shape[0]) == int(pred.shape[0]):
        tgt = raw_target
    else:
        tgt = raw_target.reshape(1, -1).expand_as(pred)
    nb = int(n_bands or 0)
    start = max(int(band_start_one_based) - 1, 0)
    if nb > start and int(pred.shape[1]) % nb == 0:
        pred = pred.reshape(int(pred.shape[0]), -1, nb)[:, :, start:]
        tgt = tgt.reshape(int(tgt.shape[0]), -1, nb)[:, :, start:]
    return (torch.abs(pred - tgt) / torch.clamp(torch.abs(tgt), min=float(epsilon))).mean()


def _reshape_band_tensor(
    values: Any,
    *,
    torch: Any,
    n_bands: int | None,
) -> Any:
    flat = values.reshape(int(values.shape[0]), -1)
    nb = int(n_bands or 0)
    if nb <= 0 or int(flat.shape[1]) % nb != 0:
        raise ValueError("n_bands must divide the flattened prediction width")
    _ = torch
    return flat.reshape(int(flat.shape[0]), -1, nb)


def cross_surrogate_consistency_loss(
    primary_predictions: Any,
    secondary_predictions: Any,
    *,
    torch: Any,
    n_bands: int | None,
    band_start_one_based: int = 2,
    epsilon: float = 1.0e-8,
) -> Any:
    """Return a bands-aware disagreement loss between two frozen surrogates."""

    primary = _reshape_band_tensor(primary_predictions, torch=torch, n_bands=n_bands)
    secondary = _reshape_band_tensor(secondary_predictions, torch=torch, n_bands=n_bands)
    start = max(int(band_start_one_based) - 1, 0)
    if int(primary.shape[2]) > start:
        primary = primary[:, :, start:]
        secondary = secondary[:, :, start:]
    denom = torch.clamp(torch.abs(secondary.detach()), min=float(epsilon))
    return (torch.abs(primary - secondary) / denom).mean()


def soft_gap_descriptor(
    values: Any,
    *,
    torch: Any,
    n_bands: int | None,
    band_start_one_based: int = 2,
    temperature: float = 0.05,
    epsilon: float = 1.0e-8,
) -> Any:
    """Return a differentiable adjacent-band complete-gap descriptor."""

    bands = _reshape_band_tensor(values, torch=torch, n_bands=n_bands)
    start = max(int(band_start_one_based) - 1, 0)
    selected = bands[:, :, start:]
    if int(selected.shape[2]) < 2:
        return bands.new_zeros((int(bands.shape[0]), 0))
    temp = max(float(temperature), float(epsilon))
    lower = selected[:, :, :-1]
    upper = selected[:, :, 1:]
    soft_upper_min = -temp * torch.logsumexp(-upper / temp, dim=1)
    soft_lower_max = temp * torch.logsumexp(lower / temp, dim=1)
    midgap = torch.clamp((torch.abs(soft_upper_min) + torch.abs(soft_lower_max)) * 0.5, min=float(epsilon))
    return (soft_upper_min - soft_lower_max) / midgap


def gap_descriptor_loss(
    predictions: Any,
    target: Any,
    *,
    torch: Any,
    n_bands: int | None,
    band_start_one_based: int = 2,
    temperature: float = 0.05,
) -> Any:
    """Match soft complete-gap descriptors instead of only pointwise bands."""

    pred_descriptor = soft_gap_descriptor(
        predictions,
        torch=torch,
        n_bands=n_bands,
        band_start_one_based=band_start_one_based,
        temperature=temperature,
    )
    raw_target = target.reshape(int(target.shape[0]), -1) if int(target.ndim) > 1 else target.reshape(1, -1)
    if int(raw_target.shape[0]) != int(predictions.shape[0]):
        raw_target = raw_target.reshape(1, -1).expand(int(predictions.shape[0]), -1)
    target_descriptor = soft_gap_descriptor(
        raw_target,
        torch=torch,
        n_bands=n_bands,
        band_start_one_based=band_start_one_based,
        temperature=temperature,
    )
    if int(pred_descriptor.numel()) == 0:
        return predictions.new_tensor(0.0)
    return torch.nn.functional.mse_loss(pred_descriptor, target_descriptor)


def candidate_diversity_loss(
    candidates: Any,
    *,
    torch: Any,
    target_diversity: float = 0.035,
    group_size: int | None = None,
) -> tuple[Any, Any]:
    """Penalize candidate groups whose mean pairwise distance collapses."""

    flat = candidates.reshape(int(candidates.shape[0]), -1)
    k = int(group_size or 0)
    if k > 1 and int(flat.shape[0]) >= k and int(flat.shape[0]) % k == 0:
        grouped = flat.reshape(int(flat.shape[0]) // k, k, int(flat.shape[1]))
        distance = torch.abs(grouped[:, :, None, :] - grouped[:, None, :, :]).mean(dim=-1)
        mask = ~torch.eye(k, dtype=torch.bool, device=flat.device).reshape(1, k, k)
        observed = distance.masked_select(mask).mean()
    elif int(flat.shape[0]) > 1:
        distance = torch.abs(flat[:, None, :] - flat[None, :, :]).mean(dim=-1)
        mask = ~torch.eye(int(flat.shape[0]), dtype=torch.bool, device=flat.device)
        observed = distance.masked_select(mask).mean()
    else:
        observed = flat.new_tensor(0.0)
    loss = torch.relu(flat.new_tensor(float(target_diversity)) - observed)
    return loss, observed


def nearest_neighbor_collision_loss(
    candidates: Any,
    *,
    torch: Any,
    target_nn_hamming: float,
    group_size: int | None = None,
    threshold: float = 0.5,
    temperature: float = 0.06,
) -> tuple[Any, Any, Any]:
    """Penalize near-duplicate candidates by nearest-neighbor Hamming distance.

    Returns (loss, nearest_neighbor_mean, nearest_neighbor_min).
    """

    flat = candidates.reshape(int(candidates.shape[0]), -1)
    temp = max(float(temperature), 1.0e-6)
    # Soft binary proxy keeps gradients while reflecting threshold-structure collisions.
    soft_binary = torch.sigmoid((flat - float(threshold)) / temp)
    n = int(soft_binary.shape[0])
    if n < 2:
        return (
            flat.new_tensor(0.0),
            flat.new_tensor(0.0),
            flat.new_tensor(0.0),
        )

    k = int(group_size or 0)
    if k > 1 and n >= k and n % k == 0:
        grouped = soft_binary.reshape(n // k, k, int(soft_binary.shape[1]))
        groups = grouped
    else:
        groups = soft_binary.reshape(1, n, int(soft_binary.shape[1]))

    nearest_values: list[Any] = []
    for g in groups:
        m = int(g.shape[0])
        if m < 2:
            continue
        dist = torch.abs(g[:, None, :] - g[None, :, :]).mean(dim=2)
        eye = torch.eye(m, device=dist.device, dtype=torch.bool)
        if m > 1:
            dist = dist.masked_fill(eye, 1.0)
        nearest = dist.min(dim=1).values
        nearest_values.append(nearest)

    if not nearest_values:
        return (
            flat.new_tensor(0.0),
            flat.new_tensor(0.0),
            flat.new_tensor(0.0),
        )

    nn_all = torch.cat(nearest_values, dim=0)
    nearest_mean = nn_all.mean()
    nearest_min = nn_all.min()
    loss = torch.relu(flat.new_tensor(float(target_nn_hamming)) - nearest_mean)
    return loss, nearest_mean, nearest_min


def _per_sample_band_relative_error(
    predictions: Any,
    target: Any,
    *,
    torch: Any,
    n_bands: int | None,
    band_start_one_based: int = 2,
    epsilon: float = 1.0e-8,
) -> Any:
    pred = predictions.reshape(int(predictions.shape[0]), -1)
    raw_target = target.reshape(int(target.shape[0]), -1) if int(target.ndim) > 1 else target.reshape(1, -1)
    if int(raw_target.shape[0]) == int(pred.shape[0]):
        tgt = raw_target
    else:
        tgt = raw_target.reshape(1, -1).expand_as(pred)
    nb = int(n_bands or 0)
    start = max(int(band_start_one_based) - 1, 0)
    if nb > start and int(pred.shape[1]) % nb == 0:
        pred = pred.reshape(int(pred.shape[0]), -1, nb)[:, :, start:]
        tgt = tgt.reshape(int(tgt.shape[0]), -1, nb)[:, :, start:]
    rel = torch.abs(pred - tgt) / torch.clamp(torch.abs(tgt), min=float(epsilon))
    return rel.reshape(int(rel.shape[0]), -1).mean(dim=1)


def quality_weighted_collision_loss(
    candidates: Any,
    *,
    predictions: Any,
    target: Any,
    torch: Any,
    n_bands: int | None,
    target_nn_hamming: float,
    group_size: int | None = None,
    threshold: float = 0.5,
    soft_binary_temperature: float = 0.08,
    quality_temperature: float = 0.03,
    quality_detach: bool = True,
    band_start_one_based: int = 2,
) -> tuple[Any, Any, Any, Any, Any]:
    """Penalize high-quality candidates whose nearest neighbor is too close."""

    flat = candidates.reshape(int(candidates.shape[0]), -1)
    n = int(flat.shape[0])
    if n < 2:
        zero = flat.new_tensor(0.0)
        return zero, zero, zero, zero, zero

    temp = max(float(soft_binary_temperature), 1.0e-6)
    soft_binary = torch.sigmoid((flat - float(threshold)) / temp)
    quality_errors = _per_sample_band_relative_error(
        predictions,
        target,
        torch=torch,
        n_bands=n_bands,
        band_start_one_based=band_start_one_based,
    )
    quality_temp = max(float(quality_temperature), 1.0e-6)
    quality_weights = torch.exp(-quality_errors / quality_temp)
    quality_weights = quality_weights / torch.clamp(quality_weights.mean(), min=1.0e-8)
    if bool(quality_detach):
        quality_weights = quality_weights.detach()

    k = int(group_size or 0)
    if k > 1 and n >= k and n % k == 0:
        candidate_groups = soft_binary.reshape(n // k, k, int(soft_binary.shape[1]))
        weight_groups = quality_weights.reshape(n // k, k)
    else:
        candidate_groups = soft_binary.reshape(1, n, int(soft_binary.shape[1]))
        weight_groups = quality_weights.reshape(1, n)

    nearest_values: list[Any] = []
    weighted_penalties: list[Any] = []
    margin = flat.new_tensor(float(target_nn_hamming))
    for group, weights in zip(candidate_groups, weight_groups):
        m = int(group.shape[0])
        if m < 2:
            continue
        dist = torch.abs(group[:, None, :] - group[None, :, :]).mean(dim=2)
        dist = dist.masked_fill(torch.eye(m, device=dist.device, dtype=torch.bool), 1.0)
        nearest = dist.min(dim=1).values
        nearest_values.append(nearest)
        weighted_penalties.append(torch.relu(margin - nearest) * weights)

    if not nearest_values:
        zero = flat.new_tensor(0.0)
        return zero, zero, zero, zero, zero

    nn_all = torch.cat(nearest_values, dim=0)
    penalties = torch.cat(weighted_penalties, dim=0)
    loss = penalties.mean()
    nearest_mean = nn_all.mean()
    nearest_min = nn_all.min()
    nearest_p10 = torch.quantile(nn_all.detach(), 0.10)
    return loss, nearest_mean, nearest_min, nearest_p10, quality_weights.mean()


def binary_candidate_diversity_loss(
    candidates: Any,
    *,
    torch: Any,
    target_diversity: float = 0.06,
    temperature: float = 0.08,
    threshold: float = 0.5,
    group_size: int | None = None,
) -> tuple[Any, Any]:
    """Penalize threshold-collapsed candidates using a differentiable binary proxy."""

    temp = max(float(temperature), 1.0e-6)
    soft_binary = torch.sigmoid((candidates - float(threshold)) / temp)
    return candidate_diversity_loss(
        soft_binary,
        torch=torch,
        target_diversity=float(target_diversity),
        group_size=group_size,
    )


def latent_response_loss(
    candidates: Any,
    *,
    torch: Any,
    target_std: float = 0.035,
    group_size: int | None = None,
) -> tuple[Any, Any]:
    """Penalize generator outputs that ignore latent noise within target groups."""

    flat = candidates.reshape(int(candidates.shape[0]), -1)
    k = int(group_size or 0)
    if k > 1 and int(flat.shape[0]) >= k and int(flat.shape[0]) % k == 0:
        grouped = flat.reshape(int(flat.shape[0]) // k, k, int(flat.shape[1]))
        observed = grouped.std(dim=1, unbiased=False).mean()
    elif int(flat.shape[0]) > 1:
        observed = flat.std(dim=0, unbiased=False).mean()
    else:
        observed = flat.new_tensor(0.0)
    loss = torch.relu(flat.new_tensor(float(target_std)) - observed)
    _ = torch
    return loss, observed


def candidate_regularization_loss(
    candidates: Any,
    *,
    torch: Any,
    target_fill_fraction: float | None = None,
    binarization_weight: float = 0.0,
    total_variation_weight: float = 0.0,
) -> Any:
    """Return soft fabrication regularizers for generated candidates."""

    loss = candidates.new_tensor(0.0)
    if target_fill_fraction is not None:
        fill = candidates.reshape(int(candidates.shape[0]), -1).mean(dim=1)
        loss = loss + (fill - float(target_fill_fraction)).pow(2).mean()
    if float(binarization_weight):
        soft_binary = candidates * (1.0 - candidates)
        loss = loss + float(binarization_weight) * soft_binary.mean()
    if float(total_variation_weight) and candidates.ndim >= 3:
        tv = candidates.new_tensor(0.0)
        tv = tv + torch.abs(candidates[..., 1:, :] - candidates[..., :-1, :]).mean()
        tv = tv + torch.abs(candidates[..., :, 1:] - candidates[..., :, :-1]).mean()
        loss = loss + float(total_variation_weight) * tv
    return loss


def objective_loss_components(
    *,
    predictions: Any,
    secondary_predictions: Any | None = None,
    secondary_primary_predictions: Any | None = None,
    target: Any,
    candidates: Any,
    reconstruction: Any | None,
    reference: Any | None,
    torch: Any,
    n_bands: int | None,
    objective: Mapping[str, Any],
) -> tuple[Any, dict[str, float]]:
    """Combine surrogate objective, optional anchor loss, and regularizers."""

    band_start = int(objective.get("band_start_one_based", objective.get("band_start", 2)))
    surrogate_weight = _as_float(objective.get("surrogate_weight"), 1.0)
    reconstruction_weight = _as_float(objective.get("reconstruction_weight"), 0.0)
    fill_weight = _as_float(objective.get("fill_fraction_weight"), 0.0)
    cross_weight = _as_float(objective.get("cross_surrogate_weight"), 0.0)
    gap_weight = _as_float(objective.get("gap_weight"), 0.0)
    diversity_weight = _as_float(objective.get("diversity_weight"), 0.0)
    binary_diversity_weight = _as_float(objective.get("binary_diversity_weight"), 0.0)
    latent_response_weight = _as_float(objective.get("latent_response_weight"), 0.0)
    collision_weight = _as_float(objective.get("collision_weight", objective.get("collision_penalty_weight")), 0.0)
    group_size = int(objective.get("candidate_group_size", objective.get("n_objective_candidates", 0)) or 0)

    surrogate_loss = band_relative_error_loss(
        predictions,
        target,
        torch=torch,
        n_bands=n_bands,
        band_start_one_based=band_start,
    )
    total = surrogate_weight * surrogate_loss

    cross_loss = candidates.new_tensor(0.0)
    if secondary_predictions is not None and cross_weight:
        cross_loss = cross_surrogate_consistency_loss(
            predictions
            if secondary_primary_predictions is None
            else secondary_primary_predictions,
            secondary_predictions,
            torch=torch,
            n_bands=n_bands,
            band_start_one_based=band_start,
        )
        total = total + cross_weight * cross_loss

    gap_loss = candidates.new_tensor(0.0)
    if gap_weight:
        gap_loss = gap_descriptor_loss(
            predictions,
            target,
            torch=torch,
            n_bands=n_bands,
            band_start_one_based=band_start,
            temperature=_as_float(objective.get("gap_temperature"), 0.05),
        )
        total = total + gap_weight * gap_loss

    diversity_loss = candidates.new_tensor(0.0)
    diversity_value = candidates.new_tensor(0.0)
    if diversity_weight:
        diversity_loss, diversity_value = candidate_diversity_loss(
            candidates,
            torch=torch,
            target_diversity=_as_float(objective.get("target_diversity_hamming"), 0.035),
            group_size=group_size,
        )
        total = total + diversity_weight * diversity_loss

    binary_diversity_loss = candidates.new_tensor(0.0)
    binary_diversity_value = candidates.new_tensor(0.0)
    if binary_diversity_weight:
        binary_diversity_loss, binary_diversity_value = binary_candidate_diversity_loss(
            candidates,
            torch=torch,
            target_diversity=_as_float(objective.get("target_binary_diversity_hamming"), 0.06),
            temperature=_as_float(objective.get("soft_binary_temperature"), 0.08),
            threshold=_as_float(objective.get("soft_binary_threshold"), 0.5),
            group_size=group_size,
        )
        total = total + binary_diversity_weight * binary_diversity_loss

    response_loss = candidates.new_tensor(0.0)
    response_value = candidates.new_tensor(0.0)
    if latent_response_weight:
        response_loss, response_value = latent_response_loss(
            candidates,
            torch=torch,
            target_std=_as_float(objective.get("target_latent_response_std"), 0.035),
            group_size=group_size,
        )
        total = total + latent_response_weight * response_loss

    collision_loss = candidates.new_tensor(0.0)
    collision_nearest = candidates.new_tensor(0.0)
    collision_min = candidates.new_tensor(0.0)
    collision_p10 = candidates.new_tensor(0.0)
    collision_quality_mean = candidates.new_tensor(0.0)
    if collision_weight:
        (
            collision_loss,
            collision_nearest,
            collision_min,
            collision_p10,
            collision_quality_mean,
        ) = quality_weighted_collision_loss(
            candidates,
            predictions=predictions,
            target=target,
            torch=torch,
            target_nn_hamming=_as_float(
                objective.get("target_nearest_neighbor_hamming", objective.get("target_collision_nn_hamming")),
                1.0 / float(max(int(candidates.shape[1]), 1)),
            ),
            n_bands=n_bands,
            band_start_one_based=band_start,
            group_size=group_size,
            threshold=_as_float(objective.get("collision_threshold"), 0.5),
            soft_binary_temperature=_as_float(
                objective.get("collision_soft_binary_temperature", objective.get("collision_temperature")),
                0.08,
            ),
            quality_temperature=_as_float(objective.get("collision_quality_temperature"), 0.03),
            quality_detach=bool(objective.get("collision_quality_detach", True)),
        )
        total = total + collision_weight * collision_loss

    reconstruction_loss = candidates.new_tensor(0.0)
    if reconstruction is not None and reference is not None and reconstruction_weight:
        reconstruction_loss = torch.nn.functional.mse_loss(reconstruction, reference)
        total = total + reconstruction_weight * reconstruction_loss

    target_fill = objective.get("target_fill_fraction")
    if target_fill is None and reference is not None and fill_weight:
        target_fill = float(reference.reshape(int(reference.shape[0]), -1).mean().detach().cpu().item())
    regularization_loss = candidate_regularization_loss(
        candidates,
        torch=torch,
        target_fill_fraction=float(target_fill) if target_fill is not None and fill_weight else None,
        binarization_weight=_as_float(objective.get("binarization_weight"), 0.0),
        total_variation_weight=_as_float(objective.get("total_variation_weight"), 0.0),
    )
    if fill_weight:
        total = total + fill_weight * regularization_loss
    elif regularization_loss.requires_grad and float(regularization_loss.detach().cpu().item()):
        total = total + regularization_loss

    metrics = {
        "surrogate_objective_loss": float(surrogate_loss.detach().cpu().item()),
        "cross_surrogate_disagreement_bands_2_10": float(cross_loss.detach().cpu().item()),
        "gap_descriptor_error": float(gap_loss.detach().cpu().item()),
        "diversity_loss": float(diversity_loss.detach().cpu().item()),
        "diversity_hamming": float(diversity_value.detach().cpu().item()),
        "binary_diversity_loss": float(binary_diversity_loss.detach().cpu().item()),
        "binary_diversity_hamming": float(binary_diversity_value.detach().cpu().item()),
        "latent_response_loss": float(response_loss.detach().cpu().item()),
        "latent_response_std": float(response_value.detach().cpu().item()),
        "collision_penalty": float(collision_loss.detach().cpu().item()),
        "quality_weighted_collision_loss": float(collision_loss.detach().cpu().item()),
        "nearest_neighbor_hamming_mean": float(collision_nearest.detach().cpu().item()),
        "nearest_neighbor_hamming_min": float(collision_min.detach().cpu().item()),
        "nearest_neighbor_hamming_p10": float(collision_p10.detach().cpu().item()),
        "collision_quality_weight_mean": float(collision_quality_mean.detach().cpu().item()),
        "reconstruction_loss": float(reconstruction_loss.detach().cpu().item()),
        "regularization_loss": float(regularization_loss.detach().cpu().item()),
        "objective_total_loss": float(total.detach().cpu().item()),
    }
    return total, metrics


__all__ = [
    "band_relative_error_loss",
    "binary_candidate_diversity_loss",
    "candidate_regularization_loss",
    "candidate_diversity_loss",
    "nearest_neighbor_collision_loss",
    "quality_weighted_collision_loss",
    "cross_surrogate_consistency_loss",
    "gap_descriptor_loss",
    "objective_loss_components",
    "latent_response_loss",
    "soft_gap_descriptor",
]
