# models/filters.py
from pydantic import BaseModel
from typing import Optional

class ImageRequest(BaseModel):
    prompt: str
    style: Optional[str] = None
    color: Optional[str] = None
    theme: Optional[str] = None
    n: Optional[int] = 2
    size: Optional[str] = "1024x1024"
    background_removal: Optional[bool] = False
    upscale: Optional[bool] = False
    upscale_scale: Optional[int] = 2
