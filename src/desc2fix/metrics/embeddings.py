#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional

from desc2fix.metrics.common import cosine_similarity, normalize_text
from desc2fix.utils.config import get_required_env


@lru_cache(maxsize=4)
def load_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def sbert_similarity(
    text_a: Any,
    text_b: Any,
    *,
    model_name: str = "sentence-transformers/paraphrase-mpnet-base-v2",
) -> float:
    a = normalize_text(text_a)
    b = normalize_text(text_b)

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    model = load_sentence_transformer(model_name)
    emb = model.encode([a, b], convert_to_numpy=True, normalize_embeddings=True)

    return float(cosine_similarity(emb[0].tolist(), emb[1].tolist()))


@lru_cache(maxsize=4)
def load_codebert_model(model_name: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    return tokenizer, model, torch


def mean_pooling(last_hidden_state, attention_mask, torch):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(1)
    counted = mask.sum(1).clamp(min=1e-9)
    return summed / counted


def codebert_embedding(
    text: Any,
    *,
    model_name: str = "microsoft/codebert-base",
    max_length: int = 512,
) -> List[float]:
    text = normalize_text(text)

    if not text:
        return []

    tokenizer, model, torch = load_codebert_model(model_name)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    pooled = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"], torch)
    vec = pooled[0].detach().cpu().numpy().tolist()

    return vec


def codebert_similarity(
    text_a: Any,
    text_b: Any,
    *,
    model_name: str = "microsoft/codebert-base",
) -> float:
    a = normalize_text(text_a)
    b = normalize_text(text_b)

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    emb_a = codebert_embedding(a, model_name=model_name)
    emb_b = codebert_embedding(b, model_name=model_name)

    return cosine_similarity(emb_a, emb_b)


def openai_embedding(
    text: Any,
    *,
    model_name: str = "text-embedding-3-small",
    api_key_env: str = "OPENAI_API_KEY",
) -> List[float]:
    from openai import OpenAI

    text = normalize_text(text)

    if not text:
        return []

    client = OpenAI(api_key=get_required_env(api_key_env))
    response = client.embeddings.create(
        model=model_name,
        input=text,
    )

    return response.data[0].embedding


def openai_embedding_similarity(
    text_a: Any,
    text_b: Any,
    *,
    model_name: str = "text-embedding-3-small",
    api_key_env: str = "OPENAI_API_KEY",
) -> float:
    a = normalize_text(text_a)
    b = normalize_text(text_b)

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    emb_a = openai_embedding(a, model_name=model_name, api_key_env=api_key_env)
    emb_b = openai_embedding(b, model_name=model_name, api_key_env=api_key_env)

    return cosine_similarity(emb_a, emb_b)


def embedding_metrics(
    text_a: Any,
    text_b: Any,
    *,
    include_sbert: bool = True,
    include_codebert: bool = True,
    include_openai: bool = False,
) -> Dict[str, float]:
    out: Dict[str, float] = {}

    if include_sbert:
        out["sbert"] = round(sbert_similarity(text_a, text_b), 6)

    if include_codebert:
        out["codebert"] = round(codebert_similarity(text_a, text_b), 6)

    if include_openai:
        out["openai_embedding"] = round(openai_embedding_similarity(text_a, text_b), 6)

    return out
