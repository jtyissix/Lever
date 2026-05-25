"""Create an edge-side sub-index from a cloud index and a domain map."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from pathlib import Path

import faiss
import numpy as np


def build_edge_db(input_index: Path, domain_map: Path, output_index: Path, output_id_map: Path, output_domain_map: Path, total_size: int, dim: int = 768, m: int = 64, ef_search: int = 32, ef_construction: int = 32) -> None:
    with domain_map.open('r', encoding='utf-8') as f:
        domain_id_map = json.load(f)

    index = faiss.read_index(str(input_index))
    selection = []
    id_domain_array = []

    for domain_name, ids in domain_id_map.items():
        if domain_name == 'Others':
            continue
        num = math.ceil(total_size * len(ids) / sum(len(v) for v in domain_id_map.values()))
        selection.extend(random.sample(ids, min(num, len(ids))))
        id_domain_array.extend([domain_name] * min(num, len(ids)))

    remainder = total_size - len(selection)
    if remainder > 0 and 'Others' in domain_id_map:
        others = domain_id_map['Others']
        selection.extend(random.sample(others, min(remainder, len(others))))
        id_domain_array.extend(['Others'] * min(remainder, len(others)))

    xb = np.array([index.reconstruct(item) for item in selection], dtype='float32')
    index_hnsw = faiss.IndexHNSWFlat(dim, m)
    index_hnsw.hnsw.efSearch = ef_search
    index_hnsw.hnsw.efConstruction = ef_construction
    index_hnsw.add(xb)

    output_index.parent.mkdir(parents=True, exist_ok=True)
    output_id_map.parent.mkdir(parents=True, exist_ok=True)
    output_domain_map.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index_hnsw, str(output_index))
    with output_id_map.open('wb') as f:
        pickle.dump(selection, f)
    with output_domain_map.open('wb') as f:
        pickle.dump(id_domain_array, f)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Build an edge-side sub-index')
    parser.add_argument('--input-index', type=Path, required=True)
    parser.add_argument('--domain-map', type=Path, required=True)
    parser.add_argument('--output-index', type=Path, required=True)
    parser.add_argument('--output-id-map', type=Path, required=True)
    parser.add_argument('--output-domain-map', type=Path, required=True)
    parser.add_argument('--total-size', type=int, required=True)
    parser.add_argument('--dim', type=int, default=768)
    parser.add_argument('--m', type=int, default=64)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_edge_db(args.input_index, args.domain_map, args.output_index, args.output_id_map, args.output_domain_map, args.total_size, args.dim, args.m)


if __name__ == '__main__':
    main()
