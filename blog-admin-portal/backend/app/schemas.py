from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Status = Literal["draft", "published"]


class PostBase(BaseModel):
    title: str = Field(default="", max_length=255)
    body: str = ""
    tags: str = Field(default="", max_length=500)
    status: Status = "draft"


class PostCreate(PostBase):
    pass


class PostUpdate(PostBase):
    pass


class PostOut(PostBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    updated_at: datetime


class SuggestRequest(BaseModel):
    body: str
