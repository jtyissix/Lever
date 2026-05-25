"""Convert a FAISS index into an IVF index."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import faiss
import numpy as np


def convert_to_ivf(input_index: Path, output_index: Path, output_meta: Path, dim: int, nlist: int) -> None:
    index = faiss.read_index(str(input_index))
    xb = np.array([index.reconstruct(i) for i in range(index.ntotal)], dtype='float32')
    quantizer = faiss.IndexFlatL2(dim)
    index_ivf = faiss.IndexIVFFlat(quantizer, dim, nlist)
    index_ivf.train(xb)
    index_ivf.add(xb)
    output_index.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index_ivf, str(output_index))

    # Preserve the original docstore mapping if it exists alongside the source index.
    if input_index.with_suffix('.pkl').exists():
        with input_index.with_suffix('.pkl').open('rb') as f:
            payload = pickle.load(f)
        output_meta.parent.mkdir(parents=True, exist_ok=True)
        with output_meta.open('wb') as f:
            pickle.dump(payload, f)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Convert a FAISS index to IVF')
    parser.add_argument('--input-index', type=Path, required=True)
    parser.add_argument('--output-index', type=Path, required=True)
    parser.add_argument('--output-meta', type=Path, required=True)
    parser.add_argument('--dim', type=int, default=768)
    parser.add_argument('--nlist', type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    convert_to_ivf(args.input_index, args.output_index, args.output_meta, args.dim, args.nlist)


if __name__ == '__main__':
    main()
