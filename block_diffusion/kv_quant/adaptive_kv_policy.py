"""Attention-mass adaptive KV bit assignment (K4/K2, V4/V2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class BlockKAssignmentStats:
    block_idx: int
    num_cached: int
    frac_k4: float
    frac_k2: float
    mass_k4: float
    mass_k2: float
    mass_k2_band: float
    effective_k_bitwidth: float
    k4_mass_threshold: float
    k2_mass_upper: float

    def to_dict(self) -> dict:
        return {
            "block_idx": self.block_idx,
            "num_cached": self.num_cached,
            "frac_k4": self.frac_k4,
            "frac_k2": self.frac_k2,
            "mass_k4": self.mass_k4,
            "mass_k2": self.mass_k2,
            "mass_k2_band": self.mass_k2_band,
            "effective_k_bitwidth": self.effective_k_bitwidth,
            "k4_mass_threshold": self.k4_mass_threshold,
            "k2_mass_upper": self.k2_mass_upper,
        }


@dataclass
class LayerVAssignmentStats:
    layer: int
    block_idx: int
    num_cached: int
    frac_v4: float
    frac_v2: float
    mass_v4: float
    effective_v_bitwidth: float
    high_precision_mass: float

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "block_idx": self.block_idx,
            "num_cached": self.num_cached,
            "frac_v4": self.frac_v4,
            "frac_v2": self.frac_v2,
            "mass_v4": self.mass_v4,
            "effective_v_bitwidth": self.effective_v_bitwidth,
            "high_precision_mass": self.high_precision_mass,
        }


def aggregate_importance_across_layers(per_layer: dict[int, list[float]]) -> list[float]:
    """Sum cached-token importance across layers (single ranking per block)."""
    if not per_layer:
        return []
    n = max(len(v) for v in per_layer.values())
    agg = [0.0] * n
    for imp in per_layer.values():
        for i, v in enumerate(imp):
            agg[i] += float(v)
    return agg


def assign_k_bits_by_cumulative_mass(
    importance: Sequence[float],
    *,
    k4_mass: float = 0.75,
    k2_mass_upper: float = 0.95,
    k4_bits: int = 4,
    k2_bits: int = 2,
) -> tuple[list[int], float, float]:
    """Assign K4/K2 bits using cumulative normalized attention mass.

    Top mass until ``k4_mass`` → K4; remaining tokens through ``k2_mass_upper`` and
    beyond stay K2 (low tier). ``mass_k2_band`` is attention mass in (k4_mass, k2_mass_upper].
    """
    n = len(importance)
    if n == 0:
        return [], 0.0, 0.0

    total = float(sum(importance))
    if total <= 0:
        bits = [k4_bits] * n
        return bits, 1.0, 0.0

    bits = [k2_bits] * n
    order = sorted(range(n), key=lambda i: importance[i], reverse=True)
    cum = 0.0
    mass_k4 = 0.0
    mass_k2_band = 0.0
    for idx in order:
        prev_cum = cum
        token_mass = float(importance[idx]) / total
        cum += token_mass
        if prev_cum < k4_mass:
            bits[idx] = k4_bits
            mass_k4 += token_mass
        elif prev_cum < k2_mass_upper:
            mass_k2_band += token_mass

    return bits, mass_k4, mass_k2_band


def summarize_k_assignment(
    k_bits: list[int],
    importance: list[float],
    *,
    block_idx: int,
    k4_mass: float,
    k2_mass_upper: float,
    mass_k4: float,
    mass_k2_band: float,
) -> BlockKAssignmentStats:
    n = len(k_bits)
    if n == 0:
        return BlockKAssignmentStats(
            block_idx=block_idx,
            num_cached=0,
            frac_k4=0.0,
            frac_k2=0.0,
            mass_k4=0.0,
            mass_k2=0.0,
            mass_k2_band=0.0,
            effective_k_bitwidth=0.0,
            k4_mass_threshold=k4_mass,
            k2_mass_upper=k2_mass_upper,
        )
    n_k4 = sum(1 for b in k_bits if b == 4)
    total = float(sum(importance)) if importance else 0.0
    mass_k2 = (
        sum(float(importance[i]) for i, b in enumerate(k_bits) if b == 2) / total
        if total > 0
        else 0.0
    )
    return BlockKAssignmentStats(
        block_idx=block_idx,
        num_cached=n,
        frac_k4=n_k4 / n,
        frac_k2=1.0 - n_k4 / n,
        mass_k4=mass_k4,
        mass_k2=mass_k2,
        mass_k2_band=mass_k2_band,
        effective_k_bitwidth=sum(k_bits) / n,
        k4_mass_threshold=k4_mass,
        k2_mass_upper=k2_mass_upper,
    )


def assign_v_bits_by_cumulative_mass(
    importance: Sequence[float],
    *,
    high_precision_mass: float = 0.75,
    v4_bits: int = 4,
    v2_bits: int = 2,
) -> tuple[list[int], float]:
    """Assign V4/V2 bits using cumulative normalized attention mass."""
    n = len(importance)
    if n == 0:
        return [], 0.0

    total = float(sum(importance))
    if total <= 0:
        bits = [v4_bits] * n
        return bits, 1.0 if n else 0.0

    bits = [v2_bits] * n
    order = sorted(range(n), key=lambda i: importance[i], reverse=True)
    cum = 0.0
    mass_v4 = 0.0
    for idx in order:
        prev_cum = cum
        token_mass = float(importance[idx]) / total
        cum += token_mass
        if prev_cum < high_precision_mass:
            bits[idx] = v4_bits
            mass_v4 += token_mass

    return bits, mass_v4


def summarize_v_assignments(
    assignments_per_layer: dict[int, list[int]],
    importance_per_layer: dict[int, list[float]],
    *,
    block_idx: int,
    high_precision_mass: float,
) -> list[LayerVAssignmentStats]:
    rows: list[LayerVAssignmentStats] = []
    for layer in sorted(assignments_per_layer):
        bits = assignments_per_layer[layer]
        imp = importance_per_layer.get(layer, [])
        n = len(bits)
        if n == 0:
            continue
        n_v4 = sum(1 for b in bits if b == 4)
        total = float(sum(imp)) if imp else 0.0
        mass_v4 = (
            sum(float(imp[i]) for i, b in enumerate(bits) if b == 4) / total
            if total > 0
            else 0.0
        )
        eff = sum(bits) / n
        rows.append(
            LayerVAssignmentStats(
                layer=layer,
                block_idx=block_idx,
                num_cached=n,
                frac_v4=n_v4 / n,
                frac_v2=1.0 - n_v4 / n,
                mass_v4=mass_v4,
                effective_v_bitwidth=eff,
                high_precision_mass=high_precision_mass,
            )
        )
    return rows
