"""Tests for alias sampling state preparation."""
import pytest

from qsp.state import random_state, w_state
from qsp.metrics import (
    qrom_alias_analytical_fidelity,
    sparse_qrom_alias_analytical_fidelity,
)
from qsp.qrom.alias_sampling import (
    AliasTable,
    build_alias_table,
    build_sparse_alias_table,
    estimate_alias_sampling_t_count,
    estimate_sparse_alias_sampling_t_count,
    synthesize_alias_sampling,
    synthesize_sparse_alias_sampling,
)


class TestAliasTable:
    def test_uniform_distribution(self):
        state = {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5}
        table = build_alias_table(state, 2, 8)
        assert table.n_bins == 4
        assert len(table.keep_values) == 4
        assert len(table.alias_indices) == 4

    def test_w_state_table(self):
        state, n = w_state(3)
        table = build_alias_table(state, n, 8)
        assert table.n_bins == 8
        assert table.precision_bits == 8


class TestAliasSampling:
    def test_synthesize_basic(self):
        state, n = w_state(3)
        res = synthesize_alias_sampling(state, n, precision_bits=4, use_select_swap=False)
        assert res.t_count > 0
        assert res.qubit_count > n

    def test_synthesize_selectswap(self):
        state, n = w_state(3)
        res = synthesize_alias_sampling(state, n, precision_bits=4, use_select_swap=True)
        assert res.t_count > 0
        assert res.qubit_count > n

    def test_estimate_basic(self):
        t = estimate_alias_sampling_t_count(3, 4, use_select_swap=False)
        assert t > 0

    def test_estimate_selectswap(self):
        t = estimate_alias_sampling_t_count(3, 4, use_select_swap=True)
        assert t > 0

    def test_selectswap_saves_t(self):
        t_basic = estimate_alias_sampling_t_count(4, 8, use_select_swap=False)
        t_swap = estimate_alias_sampling_t_count(4, 8, use_select_swap=True)
        assert t_swap <= t_basic


class TestSparseAliasSampling:
    def test_table_compresses_to_pow2(self):
        state, n = random_state(8, 6, seed=1)
        table, index_map, m = build_sparse_alias_table(state, 6)
        # 6 nonzero entries -> 2^3 = 8 bins, not 2^8.
        assert m == 3
        assert table.n_bins == 8
        assert len(index_map) == 8
        # Real indices for the 6 occupied slots are the state's support.
        occupied = set(index_map[:6])
        assert occupied == set(state.keys())

    def test_estimate_matches_synthesis(self):
        state, n = random_state(8, 6, seed=2)
        res = synthesize_sparse_alias_sampling(state, n, precision_bits=6)
        est = estimate_sparse_alias_sampling_t_count(6, n, 6)
        assert res.t_count == est

    def test_sparse_cheaper_than_dense(self):
        # For a sparse state the compressed scheme must not cost more T.
        n, card, b = 12, 8, 6
        dense = estimate_alias_sampling_t_count(n, b)
        sparse = estimate_sparse_alias_sampling_t_count(card, n, b)
        assert sparse < dense

    def test_fidelity_matches_dense(self):
        state, n = random_state(8, 6, seed=3)
        dense_fid = qrom_alias_analytical_fidelity(state, n, 6)
        sparse_fid = sparse_qrom_alias_analytical_fidelity(state, n, 6)
        # Same quantization, just relabeled bins -> fidelities agree closely.
        assert abs(dense_fid - sparse_fid) < 1e-2
        assert sparse_fid > 0.99
