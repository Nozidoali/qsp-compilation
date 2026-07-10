from __future__ import annotations
import math
from typing import Dict, List, Set, Tuple
import numpy as np
from state import EPSILON, State

def apply_controlled_x(index_array: np.ndarray, amplitude_array: np.ndarray, control: int, control_value: bool, target: int) -> Tuple[np.ndarray, np.ndarray]:
    control_bit = (index_array >> control & 1).astype(bool)
    matched = control_bit == bool(control_value)
    new_index = np.where(matched, index_array ^ np.int64(1) << target, index_array)
    return (new_index, amplitude_array)

def apply_multi_controlled_ry(index_array: np.ndarray, amplitude_array: np.ndarray, controls: List[int], phases: List[bool], angle: float, target: int) -> Tuple[np.ndarray, np.ndarray]:
    cos_half = math.cos(angle / 2.0)
    sin_half = math.sin(angle / 2.0)
    target_mask = np.int64(1) << target
    matched = np.ones(len(index_array), dtype=bool)
    for control, phase in zip(controls, phases):
        control_bit = (index_array >> control & 1).astype(bool)
        matched &= control_bit == bool(phase)
    unmatched_index = index_array[~matched]
    unmatched_amplitude = amplitude_array[~matched]
    matched_index = index_array[matched]
    matched_amplitude = amplitude_array[matched]
    if len(matched_index) == 0:
        keep = np.abs(unmatched_amplitude) > EPSILON
        return (unmatched_index[keep], unmatched_amplitude[keep])
    matched_base = matched_index & ~target_mask
    target_bit_set = (matched_index >> target & 1).astype(bool)
    unique_bases = np.unique(matched_base)
    positions = np.searchsorted(unique_bases, matched_base)
    amplitude_zero = np.zeros(len(unique_bases))
    amplitude_one = np.zeros(len(unique_bases))
    amplitude_zero[positions[~target_bit_set]] = matched_amplitude[~target_bit_set]
    amplitude_one[positions[target_bit_set]] = matched_amplitude[target_bit_set]
    new_amplitude_zero = cos_half * amplitude_zero - sin_half * amplitude_one
    new_amplitude_one = sin_half * amplitude_zero + cos_half * amplitude_one
    keep_zero = np.abs(new_amplitude_zero) > EPSILON
    keep_one = np.abs(new_amplitude_one) > EPSILON
    keep_unmatched = np.abs(unmatched_amplitude) > EPSILON
    result_index = np.concatenate([unmatched_index[keep_unmatched], unique_bases[keep_zero], (unique_bases | target_mask)[keep_one]])
    result_amplitude = np.concatenate([unmatched_amplitude[keep_unmatched], new_amplitude_zero[keep_zero], new_amplitude_one[keep_one]])
    return (result_index, result_amplitude)

def split_on_most_unbalanced_qubit(num_qubits: int, indices: np.ndarray, fixed_values: Dict[int, bool]) -> Tuple[int, np.ndarray]:
    num_indices = len(indices)
    best_imbalance = -1
    best_qubit = 0
    best_value = False
    best_mask = None
    for qubit in range(num_qubits):
        if qubit in fixed_values:
            continue
        bit_is_set = (indices >> qubit & 1).astype(bool)
        num_set = int(bit_is_set.sum())
        imbalance = abs(num_indices - 2 * num_set)
        if imbalance == num_indices:
            continue
        if imbalance > best_imbalance:
            best_imbalance = imbalance
            best_qubit = qubit
            best_value = num_indices > 2 * num_set
            best_mask = bit_is_set
        if best_imbalance == num_indices - 1:
            break
    if best_value:
        remaining_indices = indices[best_mask]
    else:
        remaining_indices = indices[~best_mask]
    fixed_values[best_qubit] = best_value
    return (best_qubit, remaining_indices)

def select_informative_qubit(state: State, num_qubits: int, support_qubits: List[int]) -> int:
    best_qubit = 0
    min_imbalance = float('inf')
    num_amplitudes = len(state)
    for qubit in support_qubits:
        num_zero = sum((1 for index in state if not index >> qubit & 1))
        imbalance = abs(num_amplitudes - 2 * num_zero)
        if imbalance < min_imbalance:
            min_imbalance = imbalance
            best_qubit = qubit
    return best_qubit

def optimize_rotation_angles(angles: List[float], control_indices: List[int]) -> Tuple[List[float], List[int]]:
    num_controls = len(control_indices)
    min_separation = 1e-06
    redundant_controls: Set[int] = set()
    for control_position in range(num_controls):
        is_redundant = True
        for angle_index in range(len(angles)):
            flipped_index = angle_index ^ 1 << control_position
            if abs(angles[flipped_index] - angles[angle_index]) > min_separation:
                is_redundant = False
                break
        if is_redundant:
            redundant_controls.add(control_position)
    if not redundant_controls:
        return (angles, control_indices)
    kept_controls: List[int] = []
    kept_positions: List[int] = []
    for control_position in range(num_controls):
        if control_position not in redundant_controls:
            kept_controls.append(control_indices[control_position])
            kept_positions.append(control_position)
    reduced_angles: List[float] = []
    for reduced_index in range(1 << len(kept_controls)):
        original_index = 0
        for i, original_position in enumerate(kept_positions):
            original_index |= (reduced_index >> i & 1) << original_position
        reduced_angles.append(angles[original_index])
    return (reduced_angles, kept_controls)
