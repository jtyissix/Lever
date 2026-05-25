"""Convert a FAISS index into an NSG index."""

from __future__ import annotations

import argparse
from pathlib import Path

import faiss
import numpy as np


def convert_to_nsg(input_index: Path, output_index: Path, dim: int, r: int, l: int, c: int, search_l: int) -> None:
    index = faiss.read_index(str(input_index))
    xb = np.array([index.reconstruct(i) for i in range(index.ntotal)], dtype='float32')
    index_nsg = faiss.IndexNSGFlat(dim, r)
    index_nsg.nsg.R = r
    index_nsg.nsg.L = l
    index_nsg.nsg.C = c
    index_nsg.nsg.search_L = search_l
    index_nsg.add(xb)
    output_index.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index_nsg, str(output_index))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Convert a FAISS index to NSG')
    parser.add_argument('--input-index', type=Path, required=True)
    parser.add_argument('--output-index', type=Path, required=True)
    parser.add_argument('--dim', type=int, default=768)
    parser.add_argument('--r', type=int, default=64)
    parser.add_argument('--l', type=int, default=32)
    parser.add_argument('--c', type=int, default=32)
    parser.add_argument('--search-l', type=int, default=32)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    convert_to_nsg(args.input_index, args.output_index, args.dim, args.r, args.l, args.c, args.search_l)


if __name__ == '__main__':
    main()
