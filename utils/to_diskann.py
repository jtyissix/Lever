"""Convert a FAISS index into a DiskANN memory index."""

from __future__ import annotations

import argparse
from pathlib import Path

import faiss
import numpy as np
import diskannpy


def convert_to_diskann(input_index: Path, output_dir: Path, complexity: int, graph_degree: int) -> None:
    index = faiss.read_index(str(input_index))
    xb = np.array([index.reconstruct(i) for i in range(index.ntotal)], dtype='float32')
    output_dir.mkdir(parents=True, exist_ok=True)
    diskannpy.build_memory_index(
        data=xb,
        distance_metric='l2',
        index_directory=str(output_dir),
        complexity=complexity,
        graph_degree=graph_degree,
        num_threads=0,
        use_pq_build=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Convert a FAISS index to DiskANN')
    parser.add_argument('--input-index', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--complexity', type=int, default=128)
    parser.add_argument('--graph-degree', type=int, default=64)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    convert_to_diskann(args.input_index, args.output_dir, args.complexity, args.graph_degree)


if __name__ == '__main__':
    main()
