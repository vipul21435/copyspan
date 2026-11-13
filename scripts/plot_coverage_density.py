"""Plot coverage against density per record for several `copyspan score` JSON lines files.

uv run --with matplotlib python scripts/plot_coverage_density.py OUT.png NAME=FILE [NAME=FILE ...]
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    out, specs = sys.argv[1], sys.argv[2:]
    fig, axes = plt.subplots(1, len(specs), figsize=(3.4 * len(specs), 3.2), sharey=True)
    for ax, spec in zip(axes, specs, strict=True):
        name, path = spec.split("=", 1)
        cov, den = [], []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r["coverage"] is not None:
                    cov.append(r["coverage"])
                    den.append(max(r["density"], 0.05))
        ax.scatter(cov, den, s=3, alpha=0.25, color="#33658a", linewidths=0)
        ax.set_yscale("log")
        ax.set_xlim(-0.02, 1.02)
        ax.set_title(f"{name} (n = {len(cov):,})", fontsize=10)
        ax.set_xlabel("coverage")
        ax.grid(alpha=0.3, linewidth=0.5)
    axes[0].set_ylabel("density (log scale)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)


if __name__ == "__main__":
    main()
