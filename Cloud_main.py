"""Cloud-side entry script for Lever experiments.

This module exposes the cloud-side steps needed by the open-source flow:

1. Build `ask_normal_entry` from edge-side ask queries and entry metadata.
2. Consume update packets from the edge and call `storage.sampling`.
3. Run baseline / guided retrieval evaluation.
4. Generate semantic tags for the cloud index.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from DataStructure import update_packet
from Storage import Cloud_VectorStore


def _load_vectors(path: Path) -> np.ndarray:
    return np.load(path).astype("float32")


def _load_entry_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    entry_ids = np.asarray(data["id"], dtype="int32").reshape(-1, 1)
    entry_dist = np.asarray(data["dist"], dtype="float32").reshape(-1, 1)
    return entry_ids, entry_dist


def _dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def build_ask_normal_entry(args: argparse.Namespace) -> None:
    """Turn ask-stage edge entry data into cloud-side ask-normal pairs."""

    storage = Cloud_VectorStore(
        index_path=str(args.index_path),
        text_path=str(args.text_path),
        domain_map=str(args.domain_map) if args.domain_map else None,
    )

    vectors = _load_vectors(args.query_vectors)
    entry_ids, entry_dist = _load_entry_file(args.entry_file)

    if len(vectors) != len(entry_ids):
        raise ValueError("Query vectors and entry metadata must have the same length.")

    real_ids: list[int] = []
    batch_size = max(int(args.batch_size), 1)
    for start in range(0, len(vectors), batch_size):
        end = min(start + batch_size, len(vectors))
        batch = vectors[start:end]
        batch_entry_ids = entry_ids[start:end]
        batch_entry_dist = entry_dist[start:end]
        _, batch_real_ids = storage.search_by_edge_info(batch, (batch_entry_dist, batch_entry_ids), args.top_k)
        real_ids.extend(int(item[0]) for item in batch_real_ids)

    storage.get_edge_cloud_pair(entry_ids, real_ids, str(args.output_file))


def benchmark_retrieval(args: argparse.Namespace) -> None:
    """Compare baseline cloud retrieval with Lever-style guided retrieval."""

    storage = Cloud_VectorStore(
        index_path=str(args.index_path),
        text_path=str(args.text_path),
        domain_map=str(args.domain_map) if args.domain_map else None,
    )

    vectors = _load_vectors(args.query_vectors)
    entry_ids, entry_dist = _load_entry_file(args.entry_file)

    if len(vectors) != len(entry_ids):
        raise ValueError("Query vectors and entry-point metadata must have the same length.")

    result = {
        "baseline_time": [],
        "guided_time": [],
    }

    batch_size = max(int(args.batch_size), 1)
    for start in range(0, len(vectors), batch_size):
        end = min(start + batch_size, len(vectors))
        batch = vectors[start:end]
        batch_entry_ids = entry_ids[start:end]
        batch_entry_dist = entry_dist[start:end]

        t1 = time.perf_counter()
        storage.normal_search_for_comparison(batch, args.top_k)
        t2 = time.perf_counter()
        result["baseline_time"].append(t2 - t1)

        t1 = time.perf_counter()
        storage.search_by_edge_info(batch, (batch_entry_dist, batch_entry_ids), args.top_k)
        t2 = time.perf_counter()
        result["guided_time"].append(t2 - t1)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def run_update(args: argparse.Namespace) -> None:
    """Apply one or more device-index update packets and emit sampled artifacts."""

    storage = Cloud_VectorStore(
        index_path=str(args.index_path),
        text_path=str(args.text_path),
        domain_map=str(args.domain_map) if args.domain_map else None,
    )

    update_dir = Path(args.update_dir)
    packets = sorted(update_dir.glob(args.pattern))
    if not packets:
        raise FileNotFoundError(f"No update packets found in {update_dir}")

    for packet_path in packets:
        with packet_path.open("rb") as f:
            sample_array, no_update_array = pickle.load(f)
        packet = update_packet(None, sample_array, no_update_array)
        user_id = packet_path.stem.split("_")[-1]
        storage.sampling(
            packet,
            [args.one-hop-ratio, args.two-hop-ratio],
            user_id,
            output_dir=args.output_dir / f"user_{user_id}",
        )


def run_tag(args: argparse.Namespace) -> None:
    """Generate and cache semantic tags for the cloud index."""

    storage = Cloud_VectorStore(
        index_path=str(args.index_path),
        text_path=str(args.text_path),
        domain_map=str(args.domain_map) if args.domain_map else None,
    )
    storage.Tag(args.batch_size)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cloud-side entry points for Lever")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--index-path", type=Path, required=True)
    common.add_argument("--text-path", type=Path, required=True)
    common.add_argument("--domain-map", type=Path, default=None)

    ask = subparsers.add_parser("ask-normal-entry", parents=[common], help="Build ask_normal_entry pairs")
    ask.add_argument("--query-vectors", type=Path, required=True)
    ask.add_argument("--entry-file", type=Path, required=True)
    ask.add_argument("--output-file", type=Path, required=True)
    ask.add_argument("--top-k", type=int, default=1)
    ask.add_argument("--batch-size", type=int, default=32)
    ask.set_defaults(func=build_ask_normal_entry)

    bench = subparsers.add_parser("benchmark", parents=[common], help="Benchmark retrieval")
    bench.add_argument("--query-vectors", type=Path, required=True)
    bench.add_argument("--entry-file", type=Path, required=True)
    bench.add_argument("--output-file", type=Path, required=True)
    bench.add_argument("--top-k", type=int, default=3)
    bench.add_argument("--batch-size", type=int, default=32)
    bench.set_defaults(func=benchmark_retrieval)

    update = subparsers.add_parser("update", parents=[common], help="Run device-index update")
    update.add_argument("--update-dir", type=Path, required=True)
    update.add_argument("--pattern", default="user_*_update.pkl")
    update.add_argument("--one-hop-ratio", type=float, default=0.5)
    update.add_argument("--two-hop-ratio", type=float, default=0.5)
    update.add_argument("--output-dir", type=Path, required=True)
    update.set_defaults(func=run_update)

    tag = subparsers.add_parser("tag", parents=[common], help="Generate semantic tags")
    tag.add_argument("--batch-size", type=int, default=32)
    tag.set_defaults(func=run_tag)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()