from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

from embedding import ernie_embedding_v1, gle_embedding_v2, huggingface_embedding

router = APIRouter()


class EmbeddingApiRequest(BaseModel):
    contents: List[str] = []
    content: str = ""
    model: str = ""
    auto_truncate: bool = True


@router.post("/api/ernie/embedding")
def ernie_api(req: EmbeddingApiRequest):
    return ernie_embedding_v1(req.contents)


@router.post("/api/glm/embedding")
def glm_api(req: EmbeddingApiRequest):
    return gle_embedding_v2(req.content)


@router.post("/api/huggingface/embedding")
def huggingface_api(req: EmbeddingApiRequest):
    return huggingface_embedding(req.model, req.contents, req.auto_truncate)