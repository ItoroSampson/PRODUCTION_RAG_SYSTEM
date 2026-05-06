from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    content: str
    metadata: dict = Field(
        default_factory=lambda: {
            "source": "AWS_Well_Architected_Framework.pdf",
            "page_number": 0,
            "section": "",
        }
    )
