"""Build synthetic per-user train/test splits for MMLU-style query sets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np


def safe_parse(choice_text: str) -> list[str]:
    """Parse mixed single- and double-quoted strings safely."""
    result = []
    i = 0
    while i < len(choice_text):
        ch = choice_text[i]
        if ch in {"'", '"'}:
            quote = ch
            i += 1
            content = ""
            while i < len(choice_text):
                if choice_text[i] == "\\" and i + 1 < len(choice_text):
                    content += choice_text[i : i + 2]
                    i += 2
                elif choice_text[i] == quote:
                    if content.strip():
                        result.append(content)
                    i += 1
                    break
                else:
                    content += choice_text[i]
                    i += 1
        else:
            i += 1
    return result


def distribute(total: int, groups: int) -> list[int]:
    base = total // groups
    rem = total % groups
    out = [base] * groups
    for i in range(rem):
        out[i] += 1
    return out


def distribute_k_items(total: int, num_categories: int) -> list[int]:
    out = [0] * num_categories
    for _ in range(total):
        out[random.randint(0, num_categories - 1)] += 1
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build synthetic user splits for MMLU")
    parser.add_argument("--jsonl-root", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--user-question", type=int, default=100)
    parser.add_argument("--main-num", type=int, default=70)
    parser.add_argument("--distribution", type=int, nargs="+", required=True)
    parser.add_argument("--domains", type=str, nargs="+", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    domain_data: dict[str, dict[int, dict]] = {}
    domain_embeddings: dict[str, np.ndarray] = {}

    for domain in args.domains:
        domain_data[domain] = {}
        with (args.jsonl_root / f"{domain}.jsonl").open("r", encoding="utf-8") as f:
            idx = 1
            for line in f:
                obj = json.loads(line.strip())
                choices = safe_parse(obj["choices"]["A"])
                if len(choices) != 4:
                    continue
                question = (
                    f"Choose the correct answer: {obj['question']}\n\n"
                    f"A:{choices[0]}   B:{choices[1]}   C:{choices[2]}   D:{choices[3]}"
                )
                domain_data[domain][idx] = {"question": question, "answer": obj["answer"]}
                idx += 1
        domain_embeddings[domain] = np.load(args.embedding_root / f"{domain}.npy")

    user_id = 1
    for group_index, num_users in enumerate(args.distribution):
        num_tags = group_index + 1
        main_split = distribute(args.main_num, num_tags)
        for _ in range(num_users):
            ask = {"domain": [], "question": []}
            test = {"domain": [], "question": []}
            emb_ask = []
            emb_test = []

            sorted_domains = [k for k, _ in sorted(domain_data.items(), key=lambda x: len(x[1]), reverse=True)]
            selected = sorted_domains[:num_tags]
            ask["domain"].extend(selected)
            test["domain"].extend(selected)

            for d_idx, d_name in enumerate(selected):
                need = main_split[d_idx]
                keys = list(domain_data[d_name].keys())
                if len(keys) < 2 * need:
                    continue
                sample_keys = random.sample(keys, need)
                for k in sample_keys:
                    ask["question"].append(domain_data[d_name][k])
                    emb_ask.append(domain_embeddings[d_name][k - 1])
                    del domain_data[d_name][k]

                keys = list(domain_data[d_name].keys())
                sample_keys = random.sample(keys, need)
                for k in sample_keys:
                    test["question"].append(domain_data[d_name][k])
                    emb_test.append(domain_embeddings[d_name][k - 1])
                    del domain_data[d_name][k]

            random_split = distribute_k_items(args.user_question - args.main_num, len(args.domains))
            for d_idx, count in enumerate(random_split):
                if count == 0:
                    continue
                d_name = args.domains[d_idx]
                keys = list(domain_data[d_name].keys())
                if len(keys) < 2 * count:
                    continue
                sample_keys = random.sample(keys, count)
                for k in sample_keys:
                    ask["question"].append(domain_data[d_name][k])
                    emb_ask.append(domain_embeddings[d_name][k - 1])
                    del domain_data[d_name][k]

                keys = list(domain_data[d_name].keys())
                sample_keys = random.sample(keys, count)
                for k in sample_keys:
                    test["question"].append(domain_data[d_name][k])
                    emb_test.append(domain_embeddings[d_name][k - 1])
                    del domain_data[d_name][k]

            np.save(args.output_root / f"user_{user_id}_ask.npy", np.asarray(emb_ask, dtype="float32"))
            np.save(args.output_root / f"user_{user_id}_test.npy", np.asarray(emb_test, dtype="float32"))
            with (args.output_root / f"user_{user_id}_ask.json").open("w", encoding="utf-8") as f:
                json.dump(ask, f)
            with (args.output_root / f"user_{user_id}_test.json").open("w", encoding="utf-8") as f:
                json.dump(test, f)
            user_id += 1


if __name__ == "__main__":
    main()
