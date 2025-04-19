# Automatic1111 Integration via Flux POC
> Verify integration capabilities between Python and Automatic1111 via Flux for image generation

## High-Level Objective
* Create a proof-of-concept for generating and upscaling images through Automatic1111's API using Flux, making it usable within a Python environment for the thumbnail generation system

## Mid-Level Objective
* Set up Automatic1111 with the necessary models
* Configure Flux for API access
* Create Python wrapper functions for text-to-image generation
* Create Python wrapper functions for image upscaling
* Test generation with different parameters and style preferences
* Document the complete workflow

## Implementation Notes
* Test with anime/manga and fantasy style prompts relevant to thumbnails
* Use appropriate resolution settings for YouTube thumbnails
* Track generation times and resource usage
* Document all API endpoints and parameters for future reference

## Context

### Beginning Context
* Empty project structure
* Installed Automatic1111 and Flux

### Ending Context
* `poc/automatic1111_flux/README.md` - Setup and usage documentation
* `poc/automatic1111_flux/flux_wrapper.py` - Python wrapper for Flux API
* `poc/automatic1111_flux/generate_test.py` - Test script for image generation
* `poc/automatic1111_flux/upscale_test.py` - Test script for image upscaling
* `poc/automatic1111_flux/examples/` - Directory with example outputs

## Low-Level Tasks

> Ordered from start to finish

1. Create a Python API wrapper for interacting with Automatic1111 via Flux:
```python
from typing import List, Optional, Dict, Any, Union
import requests
import base64
from PIL import Image
import io
import os
import time

class FluxWrapper:
    """
    A wrapper class for interacting with Automatic1111 via Flux API.
    Provides methods for text-to-image generation and image upscaling.
    """
    
    def __init__(self, api_url: str = "http://127.0.0.1:7860"):
        """
        Initialize the FluxWrapper with the API URL.
        
        Args:
            api_url: The URL where Automatic1111 API is running
        """
        self.api_url = api_url
        self.txt2img_endpoint = f"{api_url}/sdapi/v1/txt2img"
        self.upscale_endpoint = f"{api_url}/sdapi/v1/extra-single-image"
        
    def test_connection(self) -> bool:
        """
        Test if the connection to Automatic1111 API is working.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = requests.get(f"{self.api_url}/sdapi/v1/sd-models")
            return response.status_code == 200
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False
            
    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 640,
        height: int = 360,
        batch_size: int = 1,
        steps: int = 25,
        cfg_scale: float = 7.0,
        sampler: str = "DPM++ 2M Karras",
        seed: int = -1,
        style_preset: Optional[str] = None,
        lora_params: Optional[Dict[str, float]] = None
    ) -> List[Image.Image]:
        """
        Generate images from text prompt using Automatic1111 API.
        
        Args:
            prompt: The text prompt for image generation
            negative_prompt: Things to avoid in the image
            width: Width of the generated image
            height: Height of the generated image
            batch_size: Number of images to generate
            steps: Number of sampling steps
            cfg_scale: Guidance scale
            sampler: Sampling method to use
            seed: Random seed (-1 for random)
            style_preset: Optional style preset name
            lora_params: Dictionary of LoRA models and their weights
            
        Returns:
            List of PIL Image objects
        """
        # Build the request payload
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "batch_size": batch_size,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "sampler_name": sampler,
            "seed": seed,
        }
        
        # Add LoRA parameters if provided
        if lora_params:
            for lora_name, weight in lora_params.items():
                payload["prompt"] += f" <lora:{lora_name}:{weight}>"
                
        # Send request
        try:
            start_time = time.time()
            response = requests.post(self.txt2img_endpoint, json=payload)
            response.raise_for_status()
            generation_time = time.time() - start_time
            
            # Process response
            result = response.json()
            images = []
            
            for img_data in result["images"]:
                img_bytes = base64.b64decode(img_data.split(",", 1)[0])
                img = Image.open(io.BytesIO(img_bytes))
                images.append(img)
                
            print(f"Generated {len(images)} images in {generation_time:.2f} seconds")
            return images
            
        except Exception as e:
            print(f"Image generation failed: {e}")
            return []
            
    def upscale_image(
        self,
        image: Union[Image.Image, str],
        scale: int = 2,
        upscaler_1: str = "R-ESRGAN 4x+",
        upscaler_2: str = "None",
        upscaler_2_weight: float = 0.0
    ) -> Optional[Image.Image]:
        """
        Upscale an image using Automatic1111 API.
        
        Args:
            image: PIL Image object or path to image file
            scale: Scale factor for upscaling
            upscaler_1: Primary upscaler model
            upscaler_2: Secondary upscaler model (optional)
            upscaler_2_weight: Weight of secondary upscaler (0-1)
            
        Returns:
            Upscaled PIL Image or None if failed
        """
        # Convert image to base64 if it's a PIL Image
        if isinstance(image, str):
            if os.path.exists(image):
                with open(image, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode("utf-8")
            else:
                print(f"Image file not found: {image}")
                return None
        else:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            img_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
        # Build payload
        payload = {
            "image": f"data:image/png;base64,{img_data}",
            "resize_mode": 0,
            "upscaling_resize": scale,
            "upscaler_1": upscaler_1,
            "upscaler_2": upscaler_2,
            "upscaler_2_visibility": upscaler_2_weight
        }
        
        # Send request
        try:
            start_time = time.time()
            response = requests.post(self.upscale_endpoint, json=payload)
            response.raise_for_status()
            upscale_time = time.time() - start_time
            
            # Process response
            result = response.json()
            img_bytes = base64.b64decode(result["image"])
            upscaled_img = Image.open(io.BytesIO(img_bytes))
            
            print(f"Upscaled image in {upscale_time:.2f} seconds")
            return upscaled_img
            
        except Exception as e:
            print(f"Image upscaling failed: {e}")
            return None
            
    def save_images(self, images: List[Image.Image], output_dir: str, prefix: str = "gen"):
        """
        Save a list of PIL images to disk.
        
        Args:
            images: List of PIL Image objects
            output_dir: Directory to save images in
            prefix: Filename prefix
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for i, img in enumerate(images):
            timestamp = int(time.time())
            filename = f"{prefix}_{timestamp}_{i}.png"
            filepath = os.path.join(output_dir, filename)
            img.save(filepath)
            print(f"Saved image to {filepath}")
```

2. Create a test script for image generation with YouTube thumbnail settings:
```python
def test_thumbnail_generation(
    flux_wrapper: FluxWrapper,
    prompt: str,
    style: str = "anime",
    output_dir: str = "examples"
) -> None:
    """
    Test generating YouTube thumbnail images using different styles and settings.
    
    Args:
        flux_wrapper: Initialized FluxWrapper instance
        prompt: Base prompt for thumbnail generation
        style: Style preset to use (anime, fantasy, etc.)
        output_dir: Directory to save output images
    """
    # Define style presets
    style_presets = {
        "anime": {
            "prompt_prefix": "anime style, detailed, vibrant colors, dynamic composition",
            "negative": "low quality, blurry, pixelated, text, watermark, signature",
            "loras": {"animeScreencap_v14": 0.7}
        },
        "fantasy": {
            "prompt_prefix": "fantasy art style, detailed, magical, epic lighting, dramatic",
            "negative": "low quality, blurry, pixelated, text, watermark, signature",
            "loras": {}
        },
        "manga": {
            "prompt_prefix": "manga style, black and white, detailed linework, dynamic",
            "negative": "low quality, blurry, pixelated, watermark, signature",
            "loras": {"mangaCleanLineart_v1": 0.8}
        }
    }
    
    # Get the selected style preset
    preset = style_presets.get(style, style_presets["anime"])
    
    # Build the full prompt
    full_prompt = f"{preset['prompt_prefix']}, {prompt}, high contrast, YouTube thumbnail, 16:9 aspect ratio"
    
    # Generate images at half resolution (640x360)
    images = flux_wrapper.generate_image(
        prompt=full_prompt,
        negative_prompt=preset["negative"],
        width=640,
        height=360,
        batch_size=4,  # Generate 4 variations
        steps=30,
        cfg_scale=8.0,
        lora_params=preset["loras"]
    )
    
    # Save the generated images
    if images:
        flux_wrapper.save_images(images, output_dir, f"thumb_{style}")
        
        # Upscale the first image to full resolution (1280x720)
        upscaled = flux_wrapper.upscale_image(images[0], scale=2)
        if upscaled:
            upscaled_dir = os.path.join(output_dir, "upscaled")
            os.makedirs(upscaled_dir, exist_ok=True)
            timestamp = int(time.time())
            upscaled.save(os.path.join(upscaled_dir, f"thumb_{style}_{timestamp}_upscaled.png"))
```

3. Create a script to test different upscaling methods:
```python
def test_upscaling_methods(
    flux_wrapper: FluxWrapper,
    image_path: str,
    output_dir: str = "examples/upscaled"
) -> None:
    """
    Test different upscaling methods on a thumbnail image.
    
    Args:
        flux_wrapper: Initialized FluxWrapper instance
        image_path: Path to the image to upscale
        output_dir: Directory to save upscaled images
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # List of upscalers to test
    upscalers = [
        "R-ESRGAN 4x+",
        "ESRGAN_4x",
        "SwinIR_4x",
        "LDSR"
    ]
    
    # Load source image
    source_img = Image.open(image_path)
    timestamp = int(time.time())
    
    # Test each upscaler
    for upscaler in upscalers:
        try:
            print(f"Testing upscaler: {upscaler}")
            result = flux_wrapper.upscale_image(
                image=source_img,
                scale=2,
                upscaler_1=upscaler
            )
            
            if result:
                safe_name = upscaler.replace(" ", "_").replace("-", "_")
                result.save(os.path.join(output_dir, f"upscale_{safe_name}_{timestamp}.png"))
        except Exception as e:
            print(f"Failed testing {upscaler}: {e}")
```

4. Create a comprehensive test script that demonstrates the complete workflow:
```python
def run_complete_workflow() -> None:
    """
    Run a complete workflow test for thumbnail generation:
    1. Generate multiple thumbnail concepts
    2. Upscale the best one
    3. Generate comparison with different styles
    """
    # Initialize the wrapper
    flux = FluxWrapper()
    
    # Test connection
    if not flux.test_connection():
        print("Could not connect to Automatic1111 API. Please make sure it's running.")
        return
        
    # Create output directories
    os.makedirs("examples", exist_ok=True)
    os.makedirs("examples/upscaled", exist_ok=True)
    
    # Test prompt for a YouTube thumbnail
    test_prompt = "Software developer crafting custom tools, programming code visible on multiple screens, productivity visualization"
    
    # Generate thumbnails in different styles
    print("Generating anime style thumbnails...")
    test_thumbnail_generation(flux, test_prompt, "anime")
    
    print("Generating fantasy style thumbnails...")
    test_thumbnail_generation(flux, test_prompt, "fantasy")
    
    print("Generating manga style thumbnails...")
    test_thumbnail_generation(flux, test_prompt, "manga")
    
    # Find a generated image to test upscaling
    example_images = [f for f in os.listdir("examples") if f.endswith(".png")]
    if example_images:
        test_image = os.path.join("examples", example_images[0])
        print(f"Testing upscaling methods on {test_image}")
        test_upscaling_methods(flux, test_image)
    
    print("Workflow test completed!")
```

5. Create a README with setup instructions and usage examples:
```markdown
# Automatic1111 Integration via Flux

This proof-of-concept demonstrates integration between Python and Automatic1111's Stable Diffusion web UI via the Flux API for YouTube thumbnail generation.

## Setup Instructions

1. Install Automatic1111's Stable Diffusion Web UI:
   ```bash
   git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
   cd stable-diffusion-webui
   ./webui.sh --api
   ```

2. Download and install models:
   - Base model: `sdXL-1.0`
   - LoRAs: 
     - `animeScreencap_v14`
     - `mangaCleanLineart_v1`

3. Install Python dependencies:
   ```bash
   pip install requests pillow
   ```

## Usage

Basic usage of the FluxWrapper class:

```python
from flux_wrapper import FluxWrapper

# Initialize the wrapper
flux = FluxWrapper()

# Generate a thumbnail
images = flux.generate_image(
    prompt="Software developer crafting custom tools, programming code visible on screens",
    width=640,
    height=360,
    batch_size=4
)

# Save generated images
flux.save_images(images, "output", "thumbnail")

# Upscale the best image
if images:
    upscaled = flux.upscale_image(images[0], scale=2)
    if upscaled:
        upscaled.save("output/upscaled_thumbnail.png")
```

## Performance Notes

- Image generation at 640×360 takes approximately 1-2 minutes for a batch of 4 images
- Upscaling from 640×360 to 1280×720 takes approximately 15-30 seconds
- Resource usage:
  - GPU memory: ~16GB during generation
  - CPU usage: Minimal
  - Disk usage: ~20MB per generated image

## Style Guidelines

For best results with YouTube thumbnails:

1. **Anime Style**:
   - Use vibrant colors and clear character focus
   - Include dynamic poses and expressions
   - Add strong contrast between foreground and background

2. **Fantasy Style**:
   - Use dramatic lighting effects
   - Include epic scenery or backgrounds
   - Focus on creating a sense of story or adventure

3. **Manga Style**:
   - Use clear, bold linework
   - Create dynamic compositions with action lines
   - Keep details clean and focused

## Troubleshooting

Common issues and solutions:

1. **Connection errors**:
   - Ensure Automatic1111 is running with the `--api` flag
   - Check that the API URL is correct (default: http://127.0.0.1:7860)

2. **Model loading errors**:
   - Verify that all required models are downloaded and placed in the correct directories
   - Check Automatic1111 logs for specific errors

3. **Poor generation results**:
   - Refine prompts to be more specific
   - Adjust sampling steps (25-30 recommended)
   - Increase CFG scale for more prompt adherence (7-9 recommended)
```
