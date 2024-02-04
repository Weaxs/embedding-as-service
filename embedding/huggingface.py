import logging
import os
from typing import List

import torch.nn.functional as F
from fastapi import HTTPException
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

modelMap = {}
for m in os.getenv("EMBEDDING_MODELS", "").split(","):
    m = m.strip()
    if m is None or m == '':
        continue

    if m == "tao-8k":
        modelMap[m] = {
            'model': AutoModel.from_pretrained("tao-8k"),
            'tokenizer': AutoTokenizer.from_pretrained("tao-8k")
        }
    elif m == "gte-large-zh":
        modelMap[m] = {
            'model': AutoModel.from_pretrained("thenlper/gte-large-zh"),
            'tokenizer': AutoTokenizer.from_pretrained("thenlper/gte-large-zh")
        }
    elif m == "acge-large-zh":
        modelMap[m] = {
            'model': AutoModel.from_pretrained("acge-large-zh"),
            'tokenizer': AutoTokenizer.from_pretrained("acge-large-zh")
        }
    else:
        logger.error("Not Supported model: {}", m)


def huggingface_embedding(model_type: str, contents: List[str], auto_truncate: bool = True):
    if model_type not in modelMap:
        raise HTTPException(status_code=400, detail="Not Supported " + model_type)

    if auto_truncate:
        truncation = True
    else:
        truncation = "do_not_truncate"

    results = []
    model = modelMap[model_type]["model"]
    tokenizer = modelMap[model_type]["tokenizer"]
    for content in contents:
        batch_data = tokenizer([content],
                               max_length=tokenizer.model_max_length,
                               padding="longest",
                               truncation=truncation,
                               return_tensors='pt')
        outputs = model(**batch_data)
        vectors = outputs.last_hidden_state[:, 0]
        # normalize embeddings
        vectors = F.normalize(vectors, p=2, dim=1)
        results.append({"tokens": len(batch_data.encodings[0].ids), "vector": vectors[0].detach().numpy().tolist()})

    return results
