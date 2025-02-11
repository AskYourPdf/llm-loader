from typing import Optional, List

from pydantic import BaseModel


class Chunk(BaseModel):
    content: str
    theme: Optional[str] = None


class OCRResponse(BaseModel):
    chunks: List[Chunk]
