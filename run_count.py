import argparse
import csv
from pathlib import Path

import numpy as np

from Constants import CELL_DIM
from Forces import initialize_chain_numba


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run repeated random chain initializations and measure the "
            "distribution of atoms per spatial cell."
        )
    )
    parser.add_argument(
        "--n-atoms",
        type=int,
        required=True,
        help="Number of atoms in each initialized chain.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of randomized initializations to sample.",
    )
    parser.add_argument(
        "--r0",
        type=float,
        default=1.0,
        help="Bond length used during chain growth.",
    )
    parser.add_argument(
        "--box-size",
        type=float,
        default=None,
        help="Simulation box size. Defaults to n_atoms * r0 * 2.5.",
    )
    parser.add_argument(
        "--cell-dim",
        type=int,
        default=CELL_DIM,
        help="Number of cells per spatial dimension.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Seed used for the first trial; each later trial uses base_seed + trial.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float64",
        help="Position storage dtype used during initialization.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("count_experiments"),
        help="Directory where CSV summaries are written.",
    )
    parser.add_argument(
        "--save-raw-counts",
        action="store_true",
        help="Also save the per-trial cell-count matrix as a NumPy array.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save a bar chart of the aggregated occupancy distribution if matplotlib is installed.",
    )
    return parser.parse_args()


def label_atoms_cpu(positions: np.ndarray, box_size: float, cell_dim: int) -> np.ndarray:
    cell_size = box_size / cell_dim
    wrapped = np.mod(positions, box_size)
    cell_coords = np.floor(wrapped / cell_size).astype(np.int64) % cell_dim
    return (
        cell_coords[0]
        + cell_coords[1] * cell_dim
        + cell_coords[2] * cell_dim * cell_dim
    )


def count_atoms_per_cell(
    positions: np.ndarray,
    box_size: float,
    cell_dim: int,
) -> np.ndarray:
    labels = label_atoms_cpu(positions, box_size, cell_dim)
    n_total_cells = cell_dim ** 3
    return np.bincount(labels, minlength=n_total_cells)


def summarize_trial(counts: np.ndarray, n_atoms: int, trial: int, seed: int) -> dict:
    occupied = counts > 0
    occupied_counts = counts[occupied]
    occupied_cells = int(occupied.sum())
    total_cells = int(counts.size)

    return {
        "trial": trial,
        "seed": seed,
        "n_atoms": n_atoms,
        "total_cells": total_cells,
        "occupied_cells": occupied_cells,
        "empty_cells": total_cells - occupied_cells,
        "occupied_fraction": occupied_cells / total_cells,
        "mean_atoms_per_cell_all": float(counts.mean()),
        "mean_atoms_per_occupied_cell": (
            float(occupied_counts.mean()) if occupied_cells else 0.0
        ),
        "std_atoms_per_cell": float(counts.std()),
        "max_atoms_in_cell": int(counts.max()),
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_save_plot(output_path: Path, occupancy_hist: np.ndarray) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    x = np.arange(occupancy_hist.size)
    plt.figure(figsize=(10, 6))
    plt.bar(x, occupancy_hist, width=0.9, edgecolor="black")
    plt.xlabel("Atoms in cell")
    plt.ylabel("Number of cells across all trials")
    plt.title("Aggregated atoms-per-cell distribution")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return True


def main():
    args = parse_args()
    dtype = np.float32 if args.dtype == "float32" else np.float64
    box_size = args.box_size if args.box_size is not None else args.n_atoms * args.r0 * 2.5
    n_total_cells = args.cell_dim ** 3

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_counts = np.zeros((args.n_trials, n_total_cells), dtype=np.int32)
    trial_rows = []
    occupancy_hist = np.zeros(args.n_atoms + 1, dtype=np.int64)
    n_atoms = args.n_atoms

    for trial in range(args.n_trials):
        seed = args.base_seed + trial
        rng = np.random.default_rng(seed)
        positions = initialize_chain_numba(
            n_atoms,
            box_size,
            args.r0,
            rng,
            dtype=dtype,
        )
        counts = count_atoms_per_cell(positions, box_size, args.cell_dim)
        all_counts[trial] = counts
        trial_rows.append(summarize_trial(counts, args.n_atoms, trial, seed))

        trial_hist = np.bincount(counts, minlength=args.n_atoms + 1)
        occupancy_hist[: trial_hist.size] += trial_hist

    distribution_rows = []
    total_cell_samples = int(all_counts.size)
    for atoms_in_cell, num_cells in enumerate(occupancy_hist):
        distribution_rows.append(
            {
                "atoms_in_cell": atoms_in_cell,
                "num_cells": int(num_cells),
                "probability": float(num_cells / total_cell_samples),
            }
        )

    write_csv(
        args.output_dir / "trial_stats.csv",
        [
            "trial",
            "seed",
            "n_atoms",
            "total_cells",
            "occupied_cells",
            "empty_cells",
            "occupied_fraction",
            "mean_atoms_per_cell_all",
            "mean_atoms_per_occupied_cell",
            "std_atoms_per_cell",
            "max_atoms_in_cell",
        ],
        trial_rows,
    )
    write_csv(
        args.output_dir / "occupancy_distribution.csv",
        ["atoms_in_cell", "num_cells", "probability"],
        distribution_rows,
    )

    if args.save_raw_counts:
        np.save(args.output_dir / "cell_counts.npy", all_counts)

    plot_saved = False
    if args.plot:
        plot_saved = maybe_save_plot(
            args.output_dir / "occupancy_distribution.png",
            occupancy_hist,
        )

    occupied_cells = [row["occupied_cells"] for row in trial_rows]
    occupied_means = [row["mean_atoms_per_occupied_cell"] for row in trial_rows]
    max_counts = [row["max_atoms_in_cell"] for row in trial_rows]

    print("Experiment complete.")
    print(f"Trials: {args.n_trials}")
    print(f"Atoms per trial: {args.n_atoms}")
    print(f"Box size: {box_size}")
    print(f"Cell grid: {args.cell_dim} x {args.cell_dim} x {args.cell_dim} ({n_total_cells} cells)")
    print(f"Mean atoms per cell over all cells: {args.n_atoms / n_total_cells:.6f}")
    print(
        "Occupied cells per trial: "
        f"mean={np.mean(occupied_cells):.2f}, min={np.min(occupied_cells)}, max={np.max(occupied_cells)}"
    )
    print(
        "Mean atoms per occupied cell: "
        f"mean={np.mean(occupied_means):.4f}, min={np.min(occupied_means):.4f}, max={np.max(occupied_means):.4f}"
    )
    print(
        "Maximum occupancy in any cell per trial: "
        f"mean={np.mean(max_counts):.2f}, min={np.min(max_counts)}, max={np.max(max_counts)}"
    )
    print(f"Saved trial stats to {args.output_dir / 'trial_stats.csv'}")
    print(f"Saved occupancy distribution to {args.output_dir / 'occupancy_distribution.csv'}")
    if args.save_raw_counts:
        print(f"Saved raw counts to {args.output_dir / 'cell_counts.npy'}")
    if args.plot:
        if plot_saved:
            print(f"Saved plot to {args.output_dir / 'occupancy_distribution.png'}")
        else:
            print("Plot requested, but matplotlib is not installed in the active environment.")


if __name__ == "__main__":
    main()
