"""Basic compatibility checks for duckfollow stack."""

from __future__ import annotations

import importlib
import platform
import sys

MODULES = [
    "numpy",
    "matplotlib",
    "casadi",
    "gym_duckietown",
]


def main() -> int:
    print(f"python={sys.version.split()[0]} ({platform.platform()})")
    ok = True
    for name in MODULES:
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", "unknown")
            print(f"ok {name} {ver}")
        except Exception as exc:
            ok = False
            print(f"fail {name}: {exc}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
