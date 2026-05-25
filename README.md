# [KDD 2026] Lever: Locality-Aware Device–Cloud Collaboration for Graph-Based RAG

Lever is a graph-based Retrieval-Augmented Generation (RAG) framework that uses a
compact device-side personalized index to guide cloud-side retrieval. The core idea
is to exploit user-level query locality so the cloud search starts from a better entry
point, reducing traversal cost while preserving retrieval quality. Extensive
experiments on multiple RAG benchmarks demonstrate that Lever
significantly reduces retrieval latency and improves throughput
while preserving retrieval quality, highlighting query locality as a
powerful and complementary lever for scalable RAG retrieval.

## Highlights

- Device–cloud collaborative graph retrieval.
- Personalized device-side index initialization and update.
- [News 05/16] Our paper has been accepted to KDD 2026 Research Track!

## Repository Structure

```text
.
├── Cloud_main.py          # Cloud-side experiment entry point
├── Edge_main.py           # Edge-side experiment entry point
├── Storage.py            # Cloud/edge vector store logic and update strategies
├── DataStructure.py      # Lightweight data containers used across modules
├── llms.py               # Embedding and LLM wrapper utilities
├── Prompts.py            # Prompt templates for semantic tagging
├── shuffle_dataset/      # Synthetic user workload generation scripts
├── utils/                # Index conversion and index-building helpers
└── Web.py                # Optional websocket demo; 
```

## Environment Setup

Create a Python environment and install the required packages for FAISS-based retrieval,
embedding models, and optional LLM-based tagging.

Example:

```bash
conda create -n lever python=3.10 -y
conda activate lever
pip install -U pip
pip install faiss-gpu==1.7.2 numpy scipy scikit-learn langchain-community transformers
# Tag generation and semantic labeling require vLLM.
pip install vllm
```

If you plan to run semantic tagging with a local LLM, you must install the runtime
required by your model backend. This repository uses `vllm` for tag generation.

## Configuration

The repository no longer depends on personal experiment paths. Configure the data,
model, and artifact locations with environment variables:

| Variable | Purpose | Default |
| --- | --- | --- |
| `RAG_DATA_ROOT` | Root directory for datasets and indexes | `./data` |
| `RAG_MODEL_ROOT` | Root directory for local model checkpoints | `./models` |
| `RAG_ARTIFACT_ROOT` | Root directory for generated artifacts | `./artifacts` |
| `RAG_TAG_MODEL_PATH` | Path to the LLM used for semantic tagging | `./models/qwen-8b` |
| `RAG_TAG_CACHE_PATH` | Cache file for generated tags | `./artifacts/tag.pkl` |
| `RAG_DOMAIN_KNN_DIR` | Directory for per-domain KNN indexes | `./artifacts/domain_knn` |
| `RAG_OUTPUT_ROOT` | Output directory for sampled indexes and update artifacts | `./artifacts` |

Example on Windows PowerShell:

```powershell
$env:RAG_DATA_ROOT = "D:\Lever\data"
$env:RAG_MODEL_ROOT = "D:\Lever\models"
$env:RAG_ARTIFACT_ROOT = "D:\Lever\artifacts"
```

Example on macOS/Linux:

```bash
export RAG_DATA_ROOT=/path/to/lever/data
export RAG_MODEL_ROOT=/path/to/lever/models
export RAG_ARTIFACT_ROOT=/path/to/lever/artifacts
```
## Datasets Preparation
- For MS MARCO dataset, you can refer to `https://github.com/ibm-aur-nlp/domain-specific-QA` for domain augmented MS MARCO dataset
- For Ultradomain dataset, refer to `https://huggingface.co/datasets/TommyChien/UltraDomain`
- For MMLU dataset, refer to `https://huggingface.co/datasets/cais/mmlu`
-For RobustQA, refer to `https://github.com/awslabs/robustqa-acl23`
## Reproduction Workflow

The recommended reproduction flow is shown below. Replace the paths with your own data locations.

1. **Prepare the corpus and embeddings**
   - Place the preprocessed corpus, embeddings, and FAISS indexes under `RAG_DATA_ROOT`. The codes are in `utils/to_hnsw.py`
   - Keep cloud-side full indexes and edge-side sub-index outputs in separate subdirectories.
   - Example:

```bash
mkdir -p "$RAG_DATA_ROOT" "$RAG_ARTIFACT_ROOT"
```

2. **Generate synthetic user workloads**
   - Use the scripts in `shuffle_dataset/` to split benchmark queries into per-user ask/test sets.
   - These scripts create the user-local sequences that simulate locality-aware query behavior.
   - The domains are the same as described in original datasets
   - The distribution can take reference to `reference_distribution.png`
   - Example:

```bash
python shuffle_dataset/robustqa.py \
  --jsonl-root "$RAG_DATA_ROOT/robustqa" \
  --embedding-root "$RAG_DATA_ROOT/robustqa_embeddings" \
  --output-root "$RAG_ARTIFACT_ROOT/robustqa/users" \
  --domains agriculture art biology biography cooking cs fiction fin health history legal literature mathematics mix music philosophy physics politics psychology technology \
  --user-question 100 \
  --main-num 70 \
  --distribution 5 80 35
```
3. **Generate tags with vLLM**
   - This step runs semantic tagging on the cloud index and saves the result to `RAG_TAG_CACHE_PATH`. You have to transform the long tail tags to 'Others'.
   - Example:

```bash
python Cloud_main.py tag \
  --index-path "$RAG_DATA_ROOT/index_hnsw.faiss" \
  --text-path "$RAG_DATA_ROOT/index.pkl" \
  --domain-map "$RAG_DATA_ROOT/domain_id_map.json" \
  --batch-size 32
```

4. **Build the edge-side sub-index**
   - Use `utils/create_edge_db.py` to sample a compact personalized edge index from the cloud index.
   - This step produces the edge FAISS index, the edge id map, and the edge domain map.
   - Example:

```bash
python utils/create_edge_db.py \
  --input-index "$RAG_DATA_ROOT/index_hnsw.faiss" \
  --domain-map "$RAG_DATA_ROOT/domain_id_map.json" \
  --output-index "$RAG_ARTIFACT_ROOT/edge/index_hnsw.faiss" \
  --output-id-map "$RAG_ARTIFACT_ROOT/edge/index.pkl" \
  --output-domain-map "$RAG_ARTIFACT_ROOT/edge/domain_id_map.pkl" \
  --total-size 100000
```

5. **Build domain-specific subgraphs**
   - Use `utils/domain_nsw.py` to build the domain subgraphs required by cloud-side sampling and guided updates.
   - The generated per-domain indexes are stored under `RAG_DOMAIN_KNN_DIR`.
   - Example:

```bash
python utils/domain_nsw.py \
  --input-index "$RAG_DATA_ROOT/index_hnsw.faiss" \
  --domain-map "$RAG_DATA_ROOT/domain_id_map.json" \
  --output-dir "$RAG_DOMAIN_KNN_DIR" \
  --dim 768 \
  --m 64 \
  --ef-search 32 \
  --ef-construction 32
```

6. **Run the first edge ask phase**
    - Use `user_*_ask.npy` to simulate the device-side ask stage and collect entry points.
    - The edge output records the ask-stage entry ids/distances and can also export the current ask statistics as `npy`.
    - Example:

```bash
python Edge_main.py entry \
  --index-path "$RAG_ARTIFACT_ROOT/edge/index_hnsw.faiss" \
  --map-path "$RAG_ARTIFACT_ROOT/edge/index.pkl" \
  --tag-path "$RAG_TAG_CACHE_PATH" \
  --query-vectors "$RAG_ARTIFACT_ROOT/robustqa/users/user_1_ask.npy" \
  --output-file "$RAG_ARTIFACT_ROOT/robustqa/ask_entry/user_1_ask_entry.json" \
  --stats-output "$RAG_ARTIFACT_ROOT/robustqa/ask_stats/user_1_ask_entry.npy"
```

7. **Build `ask_normal_entry` on the cloud and generate the update packet**
    - The cloud uses the ask-stage entry metadata to build `ask_normal_entry`.
    - Then the edge reloads the ask statistics and calls `sample_num_calc_2` to produce the update packet.
    - Example cloud command:

```bash
python Cloud_main.py ask-normal-entry \
  --index-path "$RAG_DATA_ROOT/index_hnsw.faiss" \
  --text-path "$RAG_DATA_ROOT/index.pkl" \
  --domain-map "$RAG_DATA_ROOT/domain_id_map.json" \
  --query-vectors "$RAG_ARTIFACT_ROOT/robustqa/users/user_1_ask.npy" \
  --entry-file "$RAG_ARTIFACT_ROOT/robustqa/ask_entry/user_1_ask_entry.json" \
  --output-file "$RAG_ARTIFACT_ROOT/robustqa/ask_normal_entry/user_1_ask_normal_edge_search.json"
```

```bash
python Edge_main.py sample-num-calc-2 \
  --index-path "$RAG_ARTIFACT_ROOT/edge/index_hnsw.faiss" \
  --map-path "$RAG_ARTIFACT_ROOT/edge/index.pkl" \
  --tag-path "$RAG_TAG_CACHE_PATH" \
  --stats-path "$RAG_ARTIFACT_ROOT/robustqa/ask_stats/user_1_ask_entry.npy" \
  --pair-path "$RAG_ARTIFACT_ROOT/robustqa/ask_normal_entry/user_1_ask_normal_edge_search.json" \
  --output-file "$RAG_ARTIFACT_ROOT/robustqa/update_packets/user_1_update.pkl" \
  --drop-rate 0.1 \
  --locality-ratio 0.45 \
  --data-driven-ratio 0.45 \
  --keywords-path "$RAG_ARTIFACT_ROOT/robustqa/keywords/user_1.json"
```

8. **Send update info to the cloud and rebuild the edge index**
    - The cloud receives the update packet and calls `storage.sampling`.
    - It then emits the updated `pkl` and `npy` artifacts for the edge side to rebuild from.
    - Example cloud command:

```bash
python Cloud_main.py update \
  --index-path "$RAG_DATA_ROOT/index_hnsw.faiss" \
  --text-path "$RAG_DATA_ROOT/index.pkl" \
  --domain-map "$RAG_DATA_ROOT/domain_id_map.json" \
  --update-dir "$RAG_ARTIFACT_ROOT/robustqa/update_packets" \
  --pattern "user_*_update.pkl" \
  --locality-ratio 0.45 \
  --tag-ratio 0.45 \
  --output-dir "$RAG_ARTIFACT_ROOT/robustqa/updated_index"
```

```bash
python Edge_main.py rebuild \
  --index-path "$RAG_ARTIFACT_ROOT/edge/index_hnsw.faiss" \
  --map-path "$RAG_ARTIFACT_ROOT/edge/index.pkl" \
  --sample-id-path "$RAG_ARTIFACT_ROOT/robustqa/updated_index/user_1/user_1_id.npy" \
  --sample-vec-path "$RAG_ARTIFACT_ROOT/robustqa/updated_index/user_1/user_1_xb.npy" \
  --output-index-path "$RAG_ARTIFACT_ROOT/edge/index_hnsw.faiss" \
  --output-map-path "$RAG_ARTIFACT_ROOT/edge/index.pkl"
```

9. **Run the test ask phase and then evaluate latency / QPS / recall**
    - After rebuilding, use `user_*_test.npy` to collect the final entry information for the evaluation stage.
    - This step connects directly to the paper’s later evaluation workflow.
    - Example:

```bash
python Edge_main.py entry \
  --index-path "$RAG_ARTIFACT_ROOT/edge/index_hnsw.faiss" \
  --map-path "$RAG_ARTIFACT_ROOT/edge/index.pkl" \
  --tag-path "$RAG_TAG_CACHE_PATH" \
  --query-vectors "$RAG_ARTIFACT_ROOT/robustqa/users/user_1_test.npy" \
  --output-file "$RAG_ARTIFACT_ROOT/robustqa/test_entry/user_1_test_entry.json"
```

```bash
python Cloud_main.py benchmark \
  --index-path "$RAG_DATA_ROOT/index_hnsw.faiss" \
  --text-path "$RAG_DATA_ROOT/index.pkl" \
  --domain-map "$RAG_DATA_ROOT/domain_id_map.json" \
  --query-vectors "$RAG_ARTIFACT_ROOT/robustqa/users/user_1_test.npy" \
  --entry-file "$RAG_ARTIFACT_ROOT/robustqa/test_entry/user_1_test_entry.json" \
  --output-file "$RAG_ARTIFACT_ROOT/result/retrieval.json" \
  --top-k 3
```

## What Each File Does

### Core modules

- **`Storage.py`**: Implements the edge and cloud vector store classes, including search,
  entry-point selection, personalized sampling, and device index updates.
- **`DataStructure.py`**: Provides small container classes for queries and update packets.
- **`llms.py`**: Wraps embedding inference and local text-generation backends used for semantic tagging.
- **`Prompts.py`**: Stores the prompt template used for tag extraction.

### Entry scripts

- **`Cloud_main.py`**: Cloud-side experimental entry point for retrieval evaluation and device-index update experiments.
- **`Edge_main.py`**: Edge-side experimental entry point for generating personalized entry points and update artifacts.

### Data utilities

- **`shuffle_dataset/*.py`**: Build synthetic per-user query splits for different benchmarks.
- **`utils/*.py`**: Convert or construct vector indexes in different ANN formats.

### Optional 

- **`Web.py`**: A websocket demo for transmitting a query packet. It is not required for the main Lever reproduction flow. We are looking forward to engineering contribution from the community 

## Notes for Reproducibility

- Set the environment variables before running any script.
- Make sure the same embedding model is used for both corpus chunks and queries.
- Ensure the cloud index and the edge index use consistent vector dimensions.
- You should manually transform long tail tags to `'Others'`
- `RAG_TAG_CACHE_PATH` controls where the generated tag pickle is saved.

## Citation

If you use this repository in your research, please cite the paper associated with Lever.
