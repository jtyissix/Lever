"""Build domain-specific HNSW indexes from a source FAISS index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np


def build_domain_indexes(input_index: Path, domain_map: Path, output_dir: Path, dim: int, m: int, ef_search: int, ef_construction: int) -> None:
    index = faiss.read_index(str(input_index))
    with domain_map.open('r', encoding='utf-8') as f:
        domain_id_map = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    for domain_name, ids in domain_id_map.items():
        xb = np.array([index.reconstruct(x) for x in ids], dtype='float32')
        index_hnsw = faiss.IndexHNSWFlat(dim, m)
        index_hnsw.hnsw.efSearch = ef_search
        index_hnsw.hnsw.efConstruction = ef_construction
        index_hnsw.add(xb)
        faiss.write_index(index_hnsw, str(output_dir / f'{domain_name}.faiss'))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Build domain-specific HNSW indexes')
    parser.add_argument('--input-index', type=Path, required=True)
    parser.add_argument('--domain-map', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dim', type=int, default=768)
    parser.add_argument('--m', type=int, default=64)
    parser.add_argument('--ef-search', type=int, default=32)
    parser.add_argument('--ef-construction', type=int, default=32)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_domain_indexes(args.input_index, args.domain_map, args.output_dir, args.dim, args.m, args.ef_search, args.ef_construction)


if __name__ == '__main__':
    main()
