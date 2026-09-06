"""Render a reproducible Phase 2 top-50 compute-utilization chart.

The source values were read from the live AIcrowd leaderboard's raw table
field on 2026-09-07.  `utilization` is C_m / B_m (before the 10% score floor),
not the final score multiplier.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROWS = [
    (1, "Puffi", 0.1478860181),
    (1, "suliman_tadros", 0.1504215170),
    (3, "J2W", 0.1664958591),
    (4, "marius_binner", 0.1871960857),
    (5, "oqaris", 0.1904424338),
    (6, "hyojun_kwon", 0.1461023935),
    (7, "mliston", 0.2220768611),
    (8, "Activation-Regions", 0.1920069677),
    (8, "fklassen", 0.2177564501),
    (10, "504aldo", 0.2686854930),
    (11, "lode_dockx", 0.1274423363),
    (12, "a_s6", 0.5300548246),
    (13, "Camaro", 0.3052550260),
    (14, "huang_chung_yi", 0.3350317025),
    (15, "neilarmstrong4", 0.2791193345),
    (16, "jamespayor", 0.4803597811),
    (17, "jlacombe", 0.3917028761),
    (18, "noa_nabeshima", 0.2804660557),
    (19, "MeatProxy", 0.4071974586),
    (19, "thegamers", 0.4674595253),
    (21, "kaileh57", 0.3595606668),
    (22, "NeuralForge", 0.3434226807),
    (23, "AndreasHad04", 0.1227303893),
    (24, "pranav_devarinti", 0.3406795707),
    (25, "bx0", 0.6473438135),
    (26, "utkarsh_agrawal", 0.2986102460),
    (27, "SaltyTaro", 0.4609660908),
    (28, "rphong", 0.4431453402),
    (29, "bgrubbs1984", 0.4346003637),
    (30, "drmohammad_banisalman", 0.4847239202),
    (31, "shiv_m", 0.3742410953),
    (32, "ArtemVoronov", 0.7866278468),
    (33, "bitter-lesson-pilled", 0.6975888036),
    (34, "ai_innovation", 0.3524159453),
    (35, "DLSolutions", 0.6761925374),
    (36, "zndrr", 0.7432003243),
    (37, "pricop_tudor", 0.7283541682),
    (38, "cwc", 0.6846580826),
    (39, "thibault_gounant", 0.8180513424),
    (40, "subarno_sadat_barno", 0.7868265013),
    (41, "ely2sh", 0.9108643243),
    (42, "neuron", 0.8291217709),
    (43, "konstantin_baltsat", 0.9836724943),
    (44, "trim_qewas", 0.1374427278),
    (45, "al_jannico", 0.6424551844),
    (46, "takuya_hatanaka", 0.3751455926),
    (47, "anay_garodia", 0.9867140119),
    (48, "sophie549", 0.9729436077),
    (49, "latticework_labs", 0.0868236586),
    (50, "LuDoe", 0.0920966022),
]

OUTDIR = Path(__file__).parent


def main() -> None:
    csv_path = OUTDIR / "phase2_compute_utilization_top50.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["leaderboard_rank", "participant", "compute_utilization_fraction", "compute_utilization_percent"])
        writer.writerows((rank, name, fraction, fraction * 100) for rank, name, fraction in ROWS)

    ranks = [rank for rank, _, _ in ROWS]
    names = [name for _, name, _ in ROWS]
    utilization = [fraction for _, _, fraction in ROWS]
    mine = names.index("subarno_sadat_barno")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(15, 7.5), constrained_layout=True)
    colors = ["#377eb8"] * len(ROWS)
    colors[mine] = "#e41a1c"
    ax.bar(ranks, utilization, width=0.72, color=colors, edgecolor="white", linewidth=0.35, zorder=2)
    ax.scatter([ranks[mine]], [utilization[mine]], s=88, color="#e41a1c", edgecolor="black", linewidth=0.9, zorder=4)

    median = sorted(utilization)[len(utilization) // 2 - 1 : len(utilization) // 2 + 1]
    median_value = sum(median) / 2
    ax.axhline(0.10, color="#6b7280", linestyle="--", linewidth=1.2, zorder=1, label="10% scoring floor")
    ax.axhline(median_value, color="#7c3aed", linestyle=":", linewidth=1.5, zorder=1, label=f"Top-50 median: {median_value:.1%}")
    ax.annotate(
        "You — rank 40\n78.68% of budget",
        xy=(ranks[mine], utilization[mine]),
        xytext=(31, 0.93),
        arrowprops={"arrowstyle": "->", "color": "#e41a1c", "lw": 1.5},
        color="#b91c1c",
        fontweight="bold",
        ha="left",
        va="center",
        bbox={"boxstyle": "round,pad=0.35", "fc": "#fff1f2", "ec": "#e41a1c", "alpha": 0.96},
    )

    ax.set_title("ARC White-Box Estimation Challenge 2026 — Phase 2\nCompute utilization of the live top 50", fontweight="bold", pad=14)
    ax.set_xlabel("Leaderboard rank (ties share a rank)")
    ax.set_ylabel("Compute utilization, $C_m/B_m$")
    ax.set_xlim(0, 51)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(range(1, 51, 2))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", color="#d1d5db", linewidth=0.7, alpha=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(OUTDIR / "phase2_compute_utilization_top50.png", dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
