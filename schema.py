from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    content: str
    metadata: dict = Field(
        default_factory=lambda: {
            "source": "",
            "page_number": 0,
            "pillar": None,
            "best_practice_id": None,
            "source_url": None,
        }
    )
