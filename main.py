from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from PIL import Image
import hashlib
import io
import os
import logging
import uuid

from services.openai_service import OpenAIService
from services.image_processing_service import remove_background
from models.filters import ImageRequest

# Initialize FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="processed_images"), name="static")
openai_service = OpenAIService()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


import io
from PIL import Image
import numpy as np

def remove_dynamic_green_background(image_binary: bytes, green_hue_range: tuple = (60, 120), saturation_threshold: int = 50, value_threshold: int = 50) -> bytes:
    """
    Remove a dynamically varying green background using HSV color filtering.
    Returns a binary image with transparency where green exists.

    :param image_binary: Binary data of the image.
    :param green_hue_range: Tuple specifying the range of green hues in degrees (0-360).
    :param saturation_threshold: Minimum saturation for a pixel to be considered green.
    :param value_threshold: Minimum value (brightness) for a pixel to be considered green.
    """
    try:
        # Open the image and convert it to RGBA
        with Image.open(io.BytesIO(image_binary)) as img:
            img = img.convert("RGBA")
            rgba_data = np.array(img)

            # Convert RGBA to HSV
            rgb_data = rgba_data[:, :, :3]  # Exclude alpha channel
            hsv_data = np.array([Image.fromarray(rgb_data).convert("HSV")])[0]

            # Split HSV channels
            hue, saturation, value = hsv_data[..., 0], hsv_data[..., 1], hsv_data[..., 2]

            # Identify green pixels based on HSV thresholds
            green_mask = (
                (hue >= green_hue_range[0])
                & (hue <= green_hue_range[1])
                & (saturation >= saturation_threshold)
                & (value >= value_threshold)
            )

            # Set green pixels to transparent
            rgba_data[green_mask, 3] = 0  # Set alpha channel to 0

            # Convert the modified RGBA data back to an image
            output_img = Image.fromarray(rgba_data, "RGBA")

            # Save the image back to bytes
            output = io.BytesIO()
            output_img.save(output, format="PNG")
            return output.getvalue()

    except Exception as e:
        raise Exception(f"Error removing dynamic green background: {e}")


# @app.post("/generate-images")
# async def generate_images(request: ImageRequest):
#     """
#     Generate images based on the provided prompt and return file links for processed images.
#     """
#     try:
#         logger.info("Received image generation request.")
        
#         # Construct the prompt
#         prompt = openai_service.construct_prompt(
#             prompt=request.prompt,
#             style=request.style,
#             color=request.color,
#             theme=request.theme,
#             background_removal=request.background_removal
#         )
#         logger.info(f"Constructed prompt: {prompt}")

#         # Generate images from OpenAI API
#         generated_images = openai_service.generate_images(
#             prompt=prompt,
#             n=request.n,
#             size=request.size
#         )

#         processed_image_links = []
#         for idx, img_data in enumerate(generated_images.data):
#             img_url = img_data.url
#             img_binary = openai_service.download_image(img_url)

#             # Apply optional processing
#             if request.background_removal:
#                 logger.info("Removing background from image.")
#                 img_binary = remove_background(img_binary)

#             # Save processed image to disk
#             filename = f"image_{idx + 1}.png"
#             file_path = os.path.join("processed_images", filename)
#             os.makedirs("processed_images", exist_ok=True)
#             with open(file_path, "wb") as f:
#                 f.write(img_binary)

#             # Generate link to the saved image
#             processed_image_links.append(f"/static/{filename}")

#         return {"prompt": prompt, "image_links": processed_image_links}

#     except Exception as e:
#         logger.error(f"Error in generating images: {e}")
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/generate-images")
# async def generate_images(request: ImageRequest):
#     """
#     Generate images based on the provided prompt and return file links for processed images.
#     """
#     try:
#         logger.info("Received image generation request.")

#         # Construct the prompt
#         prompt = openai_service.construct_prompt(
#             prompt=request.prompt,
#             style=request.style,
#             color=request.color,
#             theme=request.theme,
#             background_removal=request.background_removal
#         )

#         logger.info(f"Constructed prompt: {prompt}")

#         # Generate images from OpenAI API
#         generated_images = openai_service.generate_images(
#             prompt=prompt,
#             n=request.n,
#             size=request.size
#         )

#         # Ensure response has valid data
#         if not hasattr(generated_images, 'data') or not generated_images.data:
#             raise HTTPException(status_code=500, detail="No images returned by OpenAI API.")

#         processed_image_links = []

#         for idx, img_data in enumerate(generated_images.data):
#             # Extract the image URL
#             if not hasattr(img_data, "url"):
#                 logger.warning(f"No URL found for image index {idx}. Skipping.")
#                 continue

#             img_url = img_data.url  # Access the `url` attribute directly

#             # Download the image binary
#             img_binary = openai_service.download_image(img_url)

#             # Apply background removal if requested
#             if request.background_removal:
#                 logger.info(f"Removing green background from image {idx + 1}.")
#                 img_binary = remove_dynamic_green_background(img_binary)

#             # Save processed image
#             os.makedirs("processed_images", exist_ok=True)
#             filename = f"image_{idx + 1}.png"
#             file_path = os.path.join("processed_images", filename)
#             with open(file_path, "wb") as f:
#                 f.write(img_binary)

#             logger.info(f"Image saved: {file_path}")

#             # Add the file link to the response
#             processed_image_links.append(f"/static/{filename}")

#         # Check if any images were processed
#         if not processed_image_links:
#             raise HTTPException(status_code=500, detail="No images processed successfully.")

#         return {"prompt": prompt, "image_links": processed_image_links}

#     except Exception as e:
#         logger.error(f"Error in generating images: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-images")
async def generate_images(request: ImageRequest):
    """
    Generate images based on the provided prompt and return file links for processed images.
    """
    try:
        logger.info("Received image generation request.")

        # Construct the prompt
        prompt = openai_service.construct_prompt(
            prompt=request.prompt,
            style=request.style,
            color=request.color,
            theme=request.theme,
            background_removal=request.background_removal
        )

        logger.info(f"Constructed prompt: {prompt}")

        # Generate a hash of the prompt for unique identification
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]  # First 16 characters for brevity

        # Generate images from OpenAI API
        generated_images = openai_service.generate_images(
            prompt=prompt,
            n=request.n,
            size=request.size
        )

        # Ensure response has valid data
        if not hasattr(generated_images, 'data') or not generated_images.data:
            raise HTTPException(status_code=500, detail="No images returned by OpenAI API.")

        processed_image_links = []

        for idx, img_data in enumerate(generated_images.data):
            # Extract the image URL
            if not hasattr(img_data, "url"):
                logger.warning(f"No URL found for image index {idx}. Skipping.")
                continue

            img_url = img_data.url  # Access the `url` attribute directly

            # Download the image binary
            img_binary = openai_service.download_image(img_url)

            # Apply background removal if requested
            if request.background_removal:
                logger.info(f"Removing green background from image {idx + 1}.")
                img_binary = remove_dynamic_green_background(img_binary)

            # Generate a unique filename using UUID and prompt hash
            unique_id = uuid.uuid4().hex
            filename = f"image_{idx + 1}_{prompt_hash}_{unique_id}.png"
            file_path = os.path.join("processed_images", filename)
            os.makedirs("processed_images", exist_ok=True)

            # Save the image
            with open(file_path, "wb") as f:
                f.write(img_binary)

            logger.info(f"Image saved: {file_path}")

            # Add the file link to the response
            processed_image_links.append(f"/static/{filename}")

        # Check if any images were processed
        if not processed_image_links:
            raise HTTPException(status_code=500, detail="No images processed successfully.")

        return {"prompt": prompt, "image_links": processed_image_links}

    except Exception as e:
        logger.error(f"Error in generating images: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/filters")
async def get_filters():
    """
    Return metadata for available filters, themes, and customization options.
    """
    return {
        "styles": ["vintage", "modern", "cartoon", "sketch", "watercolor"],
        "colors": ["black and white", "sepia", "vibrant", "cool tones", "warm tones"],
        "themes": ["festive holidays", "futuristic cityscapes", "nature"],
        "use_cases": ["logos", "posters", "t-shirt graphics"]
    }
