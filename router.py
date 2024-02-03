from typing import List

from main import app
from embedding import ernie, glm, huggingface


class EmbeddingApiRequest:
    contents: List[str] = []
    content: str
    model: str
    auto_truncate: bool = True


@app.post("/api/ernie/embedding")
def ernie_embedding(req: EmbeddingApiRequest):
    ernie.ernie_embedding_v1(req.contents)


@app.post("/api/glm/embedding")
def glm_embedding(req: EmbeddingApiRequest):
    return glm.embedding(req.content)


@app.post("/api/huggingface/embedding")
def huggingface_embedding(req: EmbeddingApiRequest):
    huggingface.huggingface_embedding(req.model, req.contents, req.auto_truncate)