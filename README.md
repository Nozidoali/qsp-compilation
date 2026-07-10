# State Preparation Compilation

Compile a target quantum state (JSON) into an OpenQASM circuit via rotation-based (sparse/dense) or QROM-based (alias sampling) synthesis.

```bash
pip install -r requirements.txt
python main.py benchmarks/basic/w_n3.json -m selectswap -b 8 -o w_n3.qasm
# -m: sparse | dense | qrom | selectswap | sparse_alias   -b: precision bits   --clifford-t: emit Clifford+T
```

## Citation

```bibtex
@article{liu2026logical,
  title   = {Logical Resource Estimation for Quantum State Preparation with Compilation},
  author  = {Liu, Diyi and Wang, Hanyu and Zhu, Shuchen and Cong, Jason and de Jong, Wibe A. and Fang, Di and Huang, Zhen and Iancu, Costin and Yang, Chao},
  journal = {arXiv preprint arXiv:2605.18877},
  year    = {2026},
}
```
