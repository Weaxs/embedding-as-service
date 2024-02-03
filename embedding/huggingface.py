from typing import List

import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def huggingface_embedding(model: str, contents: List[str], auto_truncate: bool = True):
    results = []

    if auto_truncate:
        truncation = True
    else:
        truncation = "do_not_truncate"

    for content in contents:
        if model == "tao-8k":
            results.append(embedding("tao-8k", content, truncation))
        elif model == "gte-large-zh":
            results.append(embedding("thenlper/gte-large-zh", content, truncation))
        elif model == "acge-large-zh":
            results.append(embedding("acge-large-zh", content, truncation))
        else:
            ValueError("Not Support " + model)

    return results


def embedding(model: str, sentence: str, truncation):
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModel.from_pretrained(model)

    batch_data = tokenizer([sentence],
                           max_length=tokenizer.model_max_length,
                           padding="longest",
                           truncation=truncation,
                           return_tensors='pt')

    outputs = model(**batch_data)
    vectors = outputs.last_hidden_state[:, 0]

    # normalize embeddings
    vectors = F.normalize(vectors, p=2, dim=1)

    return {"tokens": len(batch_data.encodings[0].ids), "vector": vectors[0]}

