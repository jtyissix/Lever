"""Embedding and LLM wrappers used by Lever."""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer

try:
    from vllm import LLM, SamplingParams
except ImportError:  # pragma: no cover - optional dependency
    LLM = None
    SamplingParams = None

class Embeddings:
    def __init__(self,emb_model_path,model_kwargs,encode_kwargs=None):
        self.emb_model_path=emb_model_path
        self.model_kwargs=model_kwargs
        self.encode_kwargs=encode_kwargs
        if encode_kwargs!=None:
            self.model=HuggingFaceEmbeddings(model_name=emb_model_path,
                                       model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
        else:
            self.model = HuggingFaceEmbeddings(model_name=emb_model_path,
                                              model_kwargs=model_kwargs)

    def emb_query(self,query):
        return self.model.embed_query(query)


class summary_LLM:
    def __init__(self,model_path):
        if LLM is None or SamplingParams is None:
            raise ImportError(
                "vllm is required for summary_LLM. Install it or avoid calling semantic tagging."
            )
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        self.llm = LLM(model=model_path,
        gpu_memory_utilization=0.92,
        block_size=32,
        swap_space=16,
        tensor_parallel_size = 8
        )
        self.sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=32768)
    def generate(self,messages):
        """Generate responses for a batch of chat-style message lists."""
        text = [self.tokenizer.apply_chat_template(
            x,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        ) for x in messages]
        outputs = self.llm.generate(text, self.sampling_params)
        return [output.outputs[0].text for output in outputs ]

class LLM_wrap:
    def __init__(self):
        raise NotImplementedError(
            "LLM_wrap is not included in the open-source demo path. Use summary_LLM or ans_LLM instead."
        )
    async def generate(self,prompt,ids):
        results = self.llm.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=ids
        )
        # results is an iterator where text[0] is the newest text chunk.
        async for output in results:
            print(output.outputs[0].text)
class ans_LLM:
    def __init__(self,model_path):
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        self.llm = LLM(model=model_path,
        gpu_memory_utilization=0.92,
        block_size=32,
        swap_space=16,
        tensor_parallel_size = 8
        )
        self.sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=32768)
    def generate(self,messages):
        """Generate responses for a batch of chat-style message lists."""
        text = [self.tokenizer.apply_chat_template(
            x,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        ) for x in messages]
        outputs = self.llm.generate(text, self.sampling_params)
        return [output.outputs[0].text for output in outputs ]

if __name__=='__main__':
    raise SystemExit("This module provides utility wrappers only.")