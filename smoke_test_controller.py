from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import casadi as ca
except Exception as e:
    raise SystemExit(f"casadi import failed: {e}")

P = Path(__file__).resolve().parent / "controllers_N2_for_poliduckies.casadi"
if not P.exists():
    raise SystemExit(f"missing controller file: {P}")

M = ca.Function.load(str(P))
info = {
    "path": str(P),
    "name": M.name(),
    "n_in": M.n_in(),
    "in": [{"name": M.name_in(i), "size": list(M.size_in(i))} for i in range(M.n_in())],
    "n_out": M.n_out(),
    "out": [{"name": M.name_out(i), "size": list(M.size_out(i))} for i in range(M.n_out())],
}
print(json.dumps(info, indent=2))

# best-effort dummy forward
args = []
for i in range(M.n_in()):
    r, c = M.size_in(i)
    args.append(ca.DM.zeros(r, c))

try:
    y = M(*args)
    arr = np.array(y[0] if isinstance(y, (list, tuple)) else y).reshape(-1)
    print("dummy_call_ok", arr[:8].tolist())
except Exception as e:
    print("dummy_call_failed", str(e))
