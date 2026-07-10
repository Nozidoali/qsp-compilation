from __future__ import annotations
import math
import warnings
from typing import Dict, List, Tuple
from qiskit.circuit import QuantumCircuit

def and_compute(qc: QuantumCircuit, c0: int, c1: int, anc: int) -> None:
    qc.h(anc)
    qc.cx(c0, anc)
    qc.tdg(anc)
    qc.cx(c1, anc)
    qc.t(anc)
    qc.cx(c0, anc)
    qc.tdg(anc)
    qc.cx(c1, anc)
    qc.t(anc)
    qc.h(anc)

def and_uncompute(qc: QuantumCircuit, c0: int, c1: int, anc: int) -> None:
    qc.h(anc)
    qc.tdg(anc)
    qc.cx(c1, anc)
    qc.t(anc)
    qc.cx(c0, anc)
    qc.tdg(anc)
    qc.cx(c1, anc)
    qc.t(anc)
    qc.cx(c0, anc)
    qc.h(anc)

def mcry_gidney(qc: QuantumCircuit, theta: float, ctrls: List[int], target: int, ancillas: List[int]) -> None:
    k = len(ctrls)
    if k == 0:
        qc.ry(theta, target)
        return
    if k == 1:
        qc.cry(theta, ctrls[0], target)
        return
    if len(ancillas) < k - 1:
        raise ValueError(f'mcry_gidney: need {k - 1} ancillas for {k} controls, got {len(ancillas)}')
    and_compute(qc, ctrls[0], ctrls[1], ancillas[0])
    for i in range(2, k):
        and_compute(qc, ancillas[i - 2], ctrls[i], ancillas[i - 1])
    qc.cry(theta, ancillas[k - 2], target)
    for i in range(k - 1, 1, -1):
        and_uncompute(qc, ancillas[i - 2], ctrls[i], ancillas[i - 1])
    and_uncompute(qc, ctrls[0], ctrls[1], ancillas[0])

def add_comparator_ct(qc: QuantumCircuit, a_qubits: List[int], b_qubits: List[int], flag_qubit: int, ancillae: List[int]) -> int:
    bits = len(a_qubits)
    t_count = 0
    for i in range(bits):
        qc.x(a_qubits[i])
    for i in range(bits):
        qc.cx(flag_qubit, b_qubits[i])
        qc.cx(flag_qubit, a_qubits[i])
        and_compute(qc, a_qubits[i], b_qubits[i], ancillae[i])
        qc.cx(ancillae[i], flag_qubit)
        and_uncompute(qc, a_qubits[i], b_qubits[i], ancillae[i])
        t_count += 4
    return t_count

def add_controlled_swap_ct(qc: QuantumCircuit, ctrl: int, reg_a: List[int], reg_b: List[int], ancillae: List[int]) -> Tuple[int, int]:
    t_count = 0
    cx_count = 0
    for i in range(len(reg_a)):
        qc.cx(reg_a[i], reg_b[i])
        and_compute(qc, ctrl, reg_b[i], ancillae[i])
        qc.cx(ancillae[i], reg_a[i])
        and_uncompute(qc, ctrl, reg_b[i], ancillae[i])
        qc.cx(reg_a[i], reg_b[i])
        t_count += 4
        cx_count += 2
    return (t_count, cx_count)

def add_comparator(qc: QuantumCircuit, a_qubits: list[int], b_qubits: list[int], flag_qubit: int) -> int:
    bits = len(a_qubits)
    assert len(a_qubits) == len(b_qubits)
    ccx_count = 0
    for i in range(bits):
        qc.x(a_qubits[i])
    for i in range(bits):
        qc.cx(flag_qubit, b_qubits[i])
        qc.cx(flag_qubit, a_qubits[i])
        qc.ccx(a_qubits[i], b_qubits[i], flag_qubit)
        ccx_count += 1
    return ccx_count

def add_controlled_swap(qc: QuantumCircuit, ctrl: int, reg_a: list[int], reg_b: list[int]) -> tuple[int, int]:
    assert len(reg_a) == len(reg_b)
    ccx_count = 0
    cx_count = 0
    for i in range(len(reg_a)):
        qc.cx(reg_a[i], reg_b[i])
        qc.ccx(ctrl, reg_b[i], reg_a[i])
        qc.cx(reg_a[i], reg_b[i])
        ccx_count += 1
        cx_count += 2
    return (ccx_count, cx_count)
_gridsynth_cache: Dict[Tuple[int, int], str] = {}

def _emit_clifford_rz(qc: QuantumCircuit, theta: float, qubit: int) -> bool:
    t = theta % (2 * math.pi)
    k = round(t / (math.pi / 4))
    if abs(t - k * math.pi / 4) > 1e-09:
        return False
    k = int(k) % 8
    if k == 0:
        pass
    elif k == 1:
        qc.t(qubit)
    elif k == 2:
        qc.s(qubit)
    elif k == 3:
        qc.s(qubit)
        qc.t(qubit)
    elif k == 4:
        qc.z(qubit)
    elif k == 5:
        qc.z(qubit)
        qc.t(qubit)
    elif k == 6:
        qc.sdg(qubit)
    elif k == 7:
        qc.tdg(qubit)
    return True

def _gridsynth_key(theta: float, precision_bits: int) -> Tuple[int, int]:
    epsilon = 2.0 ** (-precision_bits)
    return (round(theta / (epsilon / 10.0)), precision_bits)

def _gridsynth_decompose(theta: float, precision_bits: int) -> str:
    key = _gridsynth_key(theta, precision_bits)
    if key not in _gridsynth_cache:
        epsilon = 2.0 ** (-precision_bits)
        warnings.filterwarnings('ignore')
        from pygridsynth import gridsynth_gates
        _gridsynth_cache[key] = gridsynth_gates(theta, epsilon)
    return _gridsynth_cache[key]

def _pygridsynth_call(args):
    theta, epsilon = args
    import warnings as _w
    _w.filterwarnings('ignore')
    from pygridsynth import gridsynth_gates
    return gridsynth_gates(theta, epsilon)

def prewarm_gridsynth_cache(thetas, precision_bits: int, max_workers: int | None=None) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    epsilon = 2.0 ** (-precision_bits)
    missing: Dict[Tuple[int, int], float] = {}
    for theta in thetas:
        key = _gridsynth_key(theta, precision_bits)
        if key in _gridsynth_cache or key in missing:
            continue
        missing[key] = theta
    if not missing:
        return
    items = list(missing.items())
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_pygridsynth_call, (theta, epsilon)): key for key, theta in items}
        for fut in as_completed(futs):
            _gridsynth_cache[futs[fut]] = fut.result()

def _emit_gridsynth_gates(qc: QuantumCircuit, gate_str: str, qubit: int) -> None:
    for ch in gate_str:
        if ch == 'H':
            qc.h(qubit)
        elif ch == 'T':
            qc.t(qubit)
        elif ch == 'S':
            qc.s(qubit)
        elif ch == 'X':
            qc.x(qubit)

def _flatten(qc: QuantumCircuit, max_rounds: int=10) -> QuantumCircuit:
    basic = {'ry', 'cx', 'x', 'h', 's', 't', 'sdg', 'tdg', 'z', 'id', 'ccx', 'rz', 'sx', 'sxdg', 'cry', 'measure', 'barrier'}
    flat = qc
    for _ in range(max_rounds):
        needs_more = False
        for instr in flat.data:
            if instr.operation.name not in basic:
                needs_more = True
                break
        if not needs_more:
            break
        flat = flat.decompose()
    return flat

def _collect_gridsynth_angles(flat: QuantumCircuit):
    for instr in flat.data:
        name = instr.operation.name
        if name == 'ry':
            theta = float(instr.operation.params[0])
            if abs(theta) < 1e-12:
                continue
            if abs(abs(theta) - math.pi) < 1e-12:
                continue
            yield theta
        elif name == 'cry':
            theta = float(instr.operation.params[0])
            if abs(theta) < 1e-12:
                continue
            half = theta / 2.0
            yield half
            yield (-half)
        elif name == 'rz':
            theta = float(instr.operation.params[0])
            if abs(theta) < 1e-12:
                continue
            t = theta % (2 * math.pi)
            k = round(t / (math.pi / 4))
            if abs(t - k * math.pi / 4) <= 1e-09:
                continue
            yield theta
        elif name == 'u':
            theta = float(instr.operation.params[0])
            phi = float(instr.operation.params[1])
            lam = float(instr.operation.params[2])
            for a in (lam, theta, phi):
                if abs(a) < 1e-12:
                    continue
                if a is theta and abs(abs(a) - math.pi) < 1e-12:
                    continue
                t = a % (2 * math.pi)
                k = round(t / (math.pi / 4))
                if abs(t - k * math.pi / 4) <= 1e-09:
                    continue
                yield a

def decompose_to_clifford_t(qc: QuantumCircuit, precision_bits: int, parallel: bool=True, max_workers: int | None=None) -> QuantumCircuit:
    flat = _flatten(qc)
    if parallel:
        prewarm_gridsynth_cache(_collect_gridsynth_angles(flat), precision_bits, max_workers=max_workers)
    out = QuantumCircuit(flat.num_qubits)
    for instr in flat.data:
        name = instr.operation.name
        qubits = [flat.find_bit(q).index for q in instr.qubits]
        if name == 'ry':
            theta = float(instr.operation.params[0])
            if abs(theta) < 1e-12:
                continue
            if abs(abs(theta) - math.pi) < 1e-12:
                out.x(qubits[0])
                continue
            q = qubits[0]
            out.sdg(q)
            out.h(q)
            gate_str = _gridsynth_decompose(theta, precision_bits)
            _emit_gridsynth_gates(out, gate_str, q)
            out.h(q)
            out.s(q)
        elif name == 'cry':
            theta = float(instr.operation.params[0])
            if abs(theta) < 1e-12:
                continue
            ctrl, tgt = qubits
            half = theta / 2.0
            out.sdg(tgt)
            out.h(tgt)
            gate_str = _gridsynth_decompose(half, precision_bits)
            _emit_gridsynth_gates(out, gate_str, tgt)
            out.h(tgt)
            out.s(tgt)
            out.cx(ctrl, tgt)
            out.sdg(tgt)
            out.h(tgt)
            gate_str = _gridsynth_decompose(-half, precision_bits)
            _emit_gridsynth_gates(out, gate_str, tgt)
            out.h(tgt)
            out.s(tgt)
            out.cx(ctrl, tgt)
        elif name == 'rz':
            theta = float(instr.operation.params[0])
            if abs(theta) < 1e-12:
                continue
            q = qubits[0]
            if not _emit_clifford_rz(out, theta, q):
                gate_str = _gridsynth_decompose(theta, precision_bits)
                _emit_gridsynth_gates(out, gate_str, q)
        elif name == 'u':
            theta = float(instr.operation.params[0])
            phi = float(instr.operation.params[1])
            lam = float(instr.operation.params[2])
            q = qubits[0]
            if abs(lam) > 1e-12:
                if not _emit_clifford_rz(out, lam, q):
                    gate_str = _gridsynth_decompose(lam, precision_bits)
                    _emit_gridsynth_gates(out, gate_str, q)
            if abs(theta) < 1e-12:
                pass
            elif abs(abs(theta) - math.pi) < 1e-12:
                out.x(q)
            else:
                out.sdg(q)
                out.h(q)
                gate_str = _gridsynth_decompose(theta, precision_bits)
                _emit_gridsynth_gates(out, gate_str, q)
                out.h(q)
                out.s(q)
            if abs(phi) > 1e-12:
                if not _emit_clifford_rz(out, phi, q):
                    gate_str = _gridsynth_decompose(phi, precision_bits)
                    _emit_gridsynth_gates(out, gate_str, q)
        else:
            out.append(instr.operation, qubits)
    return out
