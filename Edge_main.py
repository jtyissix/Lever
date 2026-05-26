"""Edge-side entry script for Lever experiments.

The open-source workflow is organized into three explicit edge-side phases:

1. Collect entry points for `user_*_ask.npy` or `user_*_test.npy`.
2. Reload the ask-stage statistics and call `sample_num_calc_2`.
3. Rebuild the edge index after the cloud emits updated vectors.
"""
#you have to change keywords manually here

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from DataStructure import update_packet
from Storage import Edge_VectorStore

keywords=['Biology', 'Business', 'Medicine', 'Health', 'Legal', 'Agriculture', 'History',
'Engineering', 'Science', 'Real Estate', 'Psychology', 'Environment', 'Sociology',
'Programming', 'Cooking', 'Healthcare', 'Art', 'Finance', 'Family', 'Insurance',
'Accounting', 'Philosophy', 'Computing', 'Literature', 'Politics', 'Law', 'Energy',
'General', 'Mathematics', 'Physics', 'Computer Science', 'Music', 'Fiction', 'Entertainment', 'Pharmaceutical', 'legal','Others']
def _load_vectors(path: Path) -> np.ndarray:
    return np.load(path).astype("float32")


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _load_keywords(path: Path | None, keywords: list[str] | None) -> list[str]:
    if keywords:
        return keywords
    if path is None:
        return []
    with path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    if isinstance(loaded, list):
        return [str(item) for item in loaded]
    if isinstance(loaded, dict) and "keywords" in loaded:
        return [str(item) for item in loaded["keywords"]]
    return [str(loaded)]


def collect_entry_points(args: argparse.Namespace) -> None:
    """Search the edge index and save entry-point distance/id pairs."""

    storage = Edge_VectorStore(
        str(args.index_path),
        str(args.map_path),
        str(args.tag_path) if args.tag_path else None,
    )
    vectors = _load_vectors(args.query_vectors)
    result = {"id": [], "dist": [], "search_time": []}

    for vec in vectors:
        q = np.asarray([vec], dtype="float32")
        dist, ids = storage.search_for_entry_point(q)
        result["id"].append(int(ids[0][0]))
        result["dist"].append(float(dist[0][0]))

    _save_json(args.output_file, result)
    if args.stats_output is not None:
        storage.export_entry_statistics(args.stats_output)


def build_sample_packet(args: argparse.Namespace) -> None:
    """Reload edge statistics and call sample_num_calc_2 for update planning."""

    storage = Edge_VectorStore(
        str(args.index_path),
        str(args.map_path),
        str(args.tag_path) if args.tag_path else None,
    )

    if args.stats_path is not None:
        storage.load_entry_statistics(args.stats_path)

    with args.pair_path.open("r", encoding="utf-8") as f:
        pair = json.load(f)
    id_domain = pickle.load(args.id_domain_map.open("rb"))
    #keywords = _load_keywords(args.keywords_path, args.keywords)
    domain1= {keyword:0 for keyword in keywords }
    for item in _load_vectors(args.query_vectors):
        _, ids = storage.search_for_entry_point(np.array([item]))
        #lst[ids[0][0]] += 1
        domain1[id_domain[storage.map.index(ids[0][0])]]+=1

    top = sorted(domain1.items(), key=lambda x: x[1], reverse=True)[:3]
    keep_keyword= [item[0] for item in top]
    sample_array, no_update_array = storage.sample_num_calc_2(
        args.drop_rate,
        args.locality_ratio,
        args.data_driven_ratio,
        pair,
        keep_keyword,
    )
    packet = update_packet(None, sample_array, no_update_array)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("wb") as f:
        pickle.dump((packet.sample_array, packet.no_update_array), f)


def rebuild_index(args: argparse.Namespace) -> None:
    """Rebuild the edge index from cloud-produced sampled vectors."""

    storage = Edge_VectorStore(
        str(args.index_path),
        str(args.map_path),
        str(args.tag_path) if args.tag_path else None,
    )
    sample_ids = np.load(args.sample_id_path).tolist()
    sample_vecs = np.load(args.sample_vec_path).astype("float32").tolist()
    storage.renew_by_sample(
        sample_ids,
        sample_vecs,
        str(args.output_index_path),
        str(args.output_map_path),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Edge-side entry points for Lever")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--index-path", type=Path, required=True)
    common.add_argument("--map-path", type=Path, required=True)
    common.add_argument("--tag-path", type=Path, default=None)

    entry = subparsers.add_parser("entry", parents=[common], help="Collect entry points")
    entry.add_argument("--query-vectors", type=Path, required=True)
    entry.add_argument("--output-file", type=Path, required=True)
    entry.add_argument("--stats-output", type=Path, default=None)
    entry.set_defaults(func=collect_entry_points)

    sample = subparsers.add_parser("sample-num-calc-2", parents=[common], help="Build a sampling packet with sample_num_calc_2")
    sample.add_argument("--stats-path", type=Path, default=None, help="Optional npy stats produced by the ask phase")
    sample.add_argument("--pair-path", type=Path, required=True, help="Cloud-produced ask_normal_entry pair json")
    sample.add_argument("--output-file", type=Path, required=True)
    sample.add_argument("--drop-rate", type=float, default=0.1)
    sample.add_argument("--locality-ratio", type=float, default=0.45)
    sample.add_argument("--data-driven-ratio", type=float, default=0.45)
    #sample.add_argument("--keywords", nargs="*", default=None, help="Keywords passed directly on the command line")
    #sample.add_argument("--keywords-path", type=Path, default=None, help="Optional json file containing keywords")
    sample.add_argument("--query-vectors", type=Path, required=True, help="Used for calculating top domains")
    sample.add_argument("--id-domain-map", type=Path, required=True, help="pkl mapping from entry ID to domain")
    sample.set_defaults(func=build_sample_packet)

    rebuild = subparsers.add_parser("rebuild", parents=[common], help="Rebuild the edge index from cloud-produced vectors")
    rebuild.add_argument("--sample-id-path", type=Path, required=True)
    rebuild.add_argument("--sample-vec-path", type=Path, required=True)
    rebuild.add_argument("--output-index-path", type=Path, required=True)
    rebuild.add_argument("--output-map-path", type=Path, required=True)
    rebuild.set_defaults(func=rebuild_index)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
