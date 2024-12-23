import openai
import requests
import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional

load_dotenv()

class OpenAIService:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found in environment variables.")
        openai.api_key = self.api_key

    def generate_images(self, prompt: str, n: int = 2, size: str = "1024x1024"):
        """
        Generate images using OpenAI API.
        """
        try:
            response = openai.images.generate(
                prompt=prompt,
                n=n,
                size=size
            )
            return response
        except Exception as e:
            raise Exception(f"OpenAI API Error: {e}")

    def construct_prompt(self, prompt: str, style: Optional[str] = None, color: Optional[str] = None, 
                         theme: Optional[str] = None, background_removal: bool = False) -> str:
        """
        Construct a detailed prompt for image generation with controlled background
        for easy background removal.
        """
        full_prompt = prompt

        if style:
            full_prompt += f", in {style} style"
        if color:
            full_prompt += f", with {color} tones"
        if theme:
            full_prompt += f", inspired by {theme} theme"
        
        # Include a single solid background for easy removal
        if background_removal:
            full_prompt += ", with a single solid green (hex code: #04F404) background"

        return full_prompt

    def download_image(self, url: str) -> bytes:
        """
        Download the image from a URL.
        """
        response = requests.get(url)
        if response.status_code == 200:
            return response.content
        else:
            raise Exception(f"Failed to download image from {url}")
