#!/usr/bin/env python3
"""
plot_results.py - turn results/benchmarks.csv into graphs and a summary table.

    S(p) = T(1) / T(p)      T(1) = sequential program time
    E(p) = S(p) / p

Each configuration is run REPEAT times; we use the median to reduce noise.

Outputs (results/figures/):
    time.png, speedup.png, efficiency.png      one line per dataset (alltoall merge)
    breakdown_<dataset>.png                     stacked phase times per p
    merge_compare.png                           gather vs alltoall
and results/summary_table.csv
"""
import csv
import os
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = os.environ.get("CSV", "results/benchmarks.csv")
FIG = "results/figures"
PHASES = ["read", "compute", "comm", "merge", "output"]
COLORS = {"read": "#4C72B0", "compute": "#55A868", "comm": "#C44E52", "merge": "#8172B2", "output": "#CCB974"}


def load():
    runs = defaultdict(list)  # (dataset, mode, procs) -> list of row dicts
    sizes = {}
    with open(CSV) as f:
        for r in csv.DictReader(f):
            key = (r["dataset"], r["mode"], int(r["procs"]))
            runs[key].append({k: float(r[k]) for k in PHASES + ["total"]})
            sizes[r["dataset"]] = int(r["bytes"])
    med = {}
    for key, rows in runs.items():
        med[key] = {k: statistics.median(x[k] for x in rows) for k in PHASES + ["total"]}
        med[key]["n"] = len(rows)
    return med, sizes


def parallel_mode(mode):
    # p=1 MPI runs are labelled "mpi"; they belong to every parallel series
    return mode in ("alltoall", "gather", "mpi")


def series(med, dataset, mode):
    pts = {}
    for (d, m, p), v in med.items():
        if d == dataset and (m == mode or (m == "mpi" and p == 1)):
            pts[p] = v
    return dict(sorted(pts.items()))


def fmt_size(b):
    return f"{b / 2**30:.1f} GB" if b >= 2**30 else f"{b / 2**20:.0f} MB"


def main():
    os.makedirs(FIG, exist_ok=True)
    med, sizes = load()
    datasets = sorted({d for (d, m, p) in med if m != "sequential"}, key=lambda d: sizes[d])
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

    table = []
    fig_t, ax_t = plt.subplots(figsize=(6, 4))
    fig_s, ax_s = plt.subplots(figsize=(6, 4))
    fig_e, ax_e = plt.subplots(figsize=(6, 4))
    max_p = 1
    for d in datasets:
        seq = med.get((d, "sequential", 1))
        pts = series(med, d, "alltoall")
        if not seq or not pts:
            continue
        t1 = seq["total"]
        ps = list(pts)
        max_p = max(max_p, max(ps))
        times = [pts[p]["total"] for p in ps]
        sp = [t1 / t for t in times]
        ef = [s / p for s, p in zip(sp, ps)]
        lab = f"{d} ({fmt_size(sizes[d])})"
        ax_t.plot(ps, times, "o-", label=lab)
        ax_s.plot(ps, sp, "o-", label=lab)
        ax_e.plot(ps, ef, "o-", label=lab)
        table.append([d, sizes[d], "sequential", 1, round(t1, 3), 1.0, 1.0])
        for p, t, s, e in zip(ps, times, sp, ef):
            table.append([d, sizes[d], "alltoall" if p > 1 else "mpi", p, round(t, 3), round(s, 3), round(e, 3)])

    ideal = [p for p in (1, 2, 4, 8, 16, 32, 64) if p <= max_p]
    ax_s.plot(ideal, ideal, "k--", alpha=0.5, label="ideal (linear)")
    for ax, title, ylab in ((ax_t, "Execution time vs processes", "time (s)"),
                            (ax_s, "Speedup S(p) = T(1) / T(p)", "speedup"),
                            (ax_e, "Efficiency E(p) = S(p) / p", "efficiency")):
        ax.set_xscale("log", base=2)
        ax.set_xticks(ideal)
        ax.set_xticklabels([str(p) for p in ideal])
        ax.set_xlabel("number of MPI processes (p)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    ax_e.set_ylim(0, 1.1)
    for fig, name in ((fig_t, "time"), (fig_s, "speedup"), (fig_e, "efficiency")):
        fig.tight_layout()
        fig.savefig(f"{FIG}/{name}.png")
        plt.close(fig)

    # phase breakdown per dataset
    for d in datasets:
        pts = series(med, d, "alltoall")
        seq = med.get((d, "sequential", 1))
        if not pts:
            continue
        labels, rows = [], []
        if seq:
            labels.append("seq")
            rows.append(seq)
        for p, v in pts.items():
            labels.append(f"p={p}")
            rows.append(v)
        fig, ax = plt.subplots(figsize=(6.5, 4))
        bottom = [0.0] * len(rows)
        for ph in PHASES:
            vals = [r[ph] for r in rows]
            ax.bar(labels, vals, bottom=bottom, label=ph, color=COLORS[ph])
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_ylabel("time (s), slowest rank per phase")
        ax.set_title(f"Time per phase - {d}")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{FIG}/breakdown_{d}.png")
        plt.close(fig)

    # gather vs alltoall
    cmp_sets = [d for d in datasets if series(med, d, "gather") and len(series(med, d, "gather")) > 1]
    if cmp_sets:
        d = cmp_sets[0]
        g, a = series(med, d, "gather"), series(med, d, "alltoall")
        ps = [p for p in g if p in a and p > 1]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
        axes[0].plot(ps, [g[p]["total"] for p in ps], "o-", label="gather (MPI_Gatherv to rank 0)")
        axes[0].plot(ps, [a[p]["total"] for p in ps], "s-", label="alltoall (MPI_Alltoallv)")
        axes[0].set_title(f"Total time - {d}")
        axes[0].set_ylabel("time (s)")
        x = range(len(ps))
        w = 0.38
        axes[1].bar([i - w / 2 for i in x], [g[p]["comm"] + g[p]["merge"] for p in ps], w, label="gather")
        axes[1].bar([i + w / 2 for i in x], [a[p]["comm"] + a[p]["merge"] for p in ps], w, label="alltoall")
        axes[1].set_xticks(list(x))
        axes[1].set_xticklabels([str(p) for p in ps])
        axes[1].set_title("Communication + merge time")
        axes[1].set_ylabel("time (s)")
        axes[0].set_xscale("log", base=2)
        axes[0].set_xticks(ps)
        axes[0].set_xticklabels([str(p) for p in ps])
        for ax in axes:
            ax.set_xlabel("processes (p)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{FIG}/merge_compare.png")
        plt.close(fig)
        for p in ps:
            table.append([d, sizes[d], "gather", p, round(g[p]["total"], 3),
                          round(med[(d, "sequential", 1)]["total"] / g[p]["total"], 3) if (d, "sequential", 1) in med else "",
                          ""])

    with open("results/summary_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "bytes", "mode", "procs", "median_time_s", "speedup", "efficiency"])
        w.writerows(table)
    print(f"figures in {FIG}/, table in results/summary_table.csv")
    for row in table:
        print("  ", row)


if __name__ == "__main__":
    main()
