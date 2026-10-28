import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.data.generator import (
    compute_travel_time_matrix,
    generate_synthetic_dataset,
    load_or_create_graph,
)


def main() -> None:
    G = load_or_create_graph()
    carers, visits = generate_synthetic_dataset(G)
    matrix = compute_travel_time_matrix(G, carers, visits)

    non_zero = matrix[matrix > 0]
    print("\n--- Step 2 Diagnostics ---")
    print(f"Total Carers:            {len(carers)}")
    print(f"Total Visits:            {len(visits)}")
    print(f"Matrix Shape:            {matrix.shape}")
    print(f"Mean Travel Time (mins): {non_zero.mean():.2f}")
    print(f"Min / Max Travel (mins): {non_zero.min():.2f} / {non_zero.max():.2f}")
    print("--------------------------")


if __name__ == "__main__":
    main()