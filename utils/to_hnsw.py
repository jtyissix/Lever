"""Convert a FAISS index into an HNSW index."""

from __future__ import annotations

import argparse
from pathlib import Path

import faiss
import numpy as np


def convert_to_hnsw(input_index: Path, output_index: Path, dim: int, m: int, ef_search: int, ef_construction: int) -> None:
    index = faiss.read_index(str(input_index))
    xb = np.array([index.reconstruct(i) for i in range(index.ntotal)], dtype='float32')
    index_hnsw = faiss.IndexHNSWFlat(dim, m)
    index_hnsw.hnsw.efSearch = ef_search
    index_hnsw.hnsw.efConstruction = ef_construction
    index_hnsw.add(xb)
    output_index.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index_hnsw, str(output_index))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Convert a FAISS index to HNSW')
    parser.add_argument('--input-index', type=Path, required=True)
    parser.add_argument('--output-index', type=Path, required=True)
    parser.add_argument('--dim', type=int, default=768)
    parser.add_argument('--m', type=int, default=64)
    parser.add_argument('--ef-search', type=int, default=32)
    parser.add_argument('--ef-construction', type=int, default=32)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    convert_to_hnsw(args.input_index, args.output_index, args.dim, args.m, args.ef_search, args.ef_construction)


if __name__ == '__main__':
    main()
