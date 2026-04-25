import argparse
import csv
from pathlib import Path

import numpy as np

from Constants import CELL_SIZE
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
        "--cell-size",
        type=float,
        default=CELL_SIZE,
        help="The dimension of the cell.",
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
    unique_labels, counts = np.unique(labels, return_counts=True)
    return unique_labels, counts

def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Saved CSV summary to {path}")


def save_plot(output_path: Path, counts: np.ndarray) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(counts, bins=range(1, int(counts.max()) + 2), edgecolor='black', alpha=0.7)
    plt.xlabel('Number of atoms per cell')
    plt.ylabel('Frequency (number of cells)')
    plt.title('Distribution of Atoms in Cells')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved plot to {output_path}")
    return True


def main():
    args = parse_args()
    dtype = np.float32 if args.dtype == "float32" else np.float64
    box_size = args.box_size if args.box_size is not None else args.n_atoms * args.r0 * 2.5

    args.output_dir.mkdir(parents=True, exist_ok=True)

    n_atoms = args.n_atoms
    cell_size = args.cell_size
    if cell_size <= 0:
        raise ValueError("cell_size must be positive.")
    cell_dim = int(box_size / cell_size)
    if cell_dim <= 0:
        raise ValueError("cell_size is too large for the selected box_size.")
    n_total_cells = cell_dim ** 3

    seed = args.base_seed
    rng = np.random.default_rng(seed)
    positions = initialize_chain_numba(
        n_atoms,
        box_size,
        args.r0,
        rng,
        dtype=dtype,
    )
    unique_labels, counts = count_atoms_per_cell(positions, box_size, cell_dim)
    
    write_csv(
        args.output_dir / "occupancy_distribution.csv",
        fieldnames=["cell_label", "atoms_in_cell"],
        rows=[{"cell_label": int(label), "atoms_in_cell": int(count)} for label, count in zip(unique_labels, counts)],
    )

    save_plot(
        args.output_dir / "occupancy_distribution.png",
        counts=counts,
    )

    print("Experiment complete.")
    print("-----------------------------")
    print(f"Atoms per trial: {args.n_atoms}")
    print(f"Box size: {box_size}")
    print(f"Cell grid: {cell_dim} x {cell_dim} x {cell_dim} ({n_total_cells} cells)")
    print(f"Mean atoms per cell over all cells: {args.n_atoms / n_total_cells:.6f}")
    print(f"Non-empty cell rate is: {counts.size / n_total_cells:.6f}")

if __name__ == "__main__":
    main()
