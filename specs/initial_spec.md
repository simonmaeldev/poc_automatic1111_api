# Automatic1111 Integration via Flux POC

> Verify integration capabilities between Python and Automatic1111 for image generation via Flux

## High-Level Objective

- Create a proof-of-concept for generating and upscaling images through Automatic1111's API using Flux, making it usable within a Python environment

## Mid-Level Objective

- Set up Automatic1111 with Flux for inference
- Create Python wrapper functions for text-to-image generation
- Create Python wrapper functions for image-to-image generation
- Create Python wrapper functions for image upscaling
- Test generation with different parameters
- Document the complete workflow

## Implementation Notes

- Test with anime/manga and fantasy style prompts relevant to thumbnails
- Default image size is 640x360
- Track generation times
- Document all API endpoints and parameters for future reference
- Use Pydantic for model handling. Use typing in the functions. Use `model_dump(warnings="error")` to convert a model to a dict.

## Context

### Beginning Context

- Empty project structure

### Ending Context

- `README.md` - Setup and usage documentation
- `install_script_wsl2_automatic1111.sh` - Local installation for wsl2 for automatic1111
- `install_script_ubuntu_automatic111.sh` - Local installation for ubuntu for automatic1111
- `automatic1111_python_wrapper.py` - Python wrapper for Automatic1111
- `tests_automatic1111.py` - Test script for image generation and upscaling
- `install_script_flux.py` - installation script for flux
- `flux_python_wrapper.py` - Python wrapper for flux
- `tests_flux.py` - Test script for image generation and upscaling

## Low-Level Tasks

> Ordered from start to finish

- installation (must test on pc)
- docker (must test on pc)

1.  CREATE `install_script_ubuntu.sh`
    CREATE an installation script for linux

    ```sh
    sudo apt install git software-properties-common -y
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt install python3.10-venv -y
    git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui && cd stable-diffusion-webui
    python3.10 -m venv venv
    ./webui.sh
    ```

2.  CREATE `install_script_wsl.sh`
    CREATE an installation script for wsl2

    ```sh
    # install conda (if not already done)
    wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
    chmod +x Anaconda3-2022.05-Linux-x86_64.sh
    ./Anaconda3-2022.05-Linux-x86_64.sh

    # Clone webui repo
    git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
    cd stable-diffusion-webui

    # Create and activate conda env
    conda env create -f environment-wsl2.yaml
    conda activate automatic
    # clone repositories for Stable Diffusion and (optionally) CodeFormer
    mkdir repositories
    git clone https://github.com/CompVis/stable-diffusion.git repositories/stable-diffusion-stability-ai
    git clone https://github.com/CompVis/taming-transformers.git repositories/taming-transformers
    git clone https://github.com/sczhou/CodeFormer.git repositories/CodeFormer
    git clone https://github.com/salesforce/BLIP.git repositories/BLIP

    # install requirements of Stable Diffusion
    pip install transformers==4.19.2 diffusers invisible-watermark --prefer-binary

    # install k-diffusion
    pip install git+https://github.com/crowsonkb/k-diffusion.git --prefer-binary

    # (optional) install GFPGAN (face restoration)
    pip install git+https://github.com/TencentARC/GFPGAN.git --prefer-binary

    # (optional) install requirements for CodeFormer (face restoration)
    pip install -r repositories/CodeFormer/requirements.txt --prefer-binary

    # install requirements of web ui
    pip install -r requirements.txt  --prefer-binary

    # update numpy to latest version
    pip install -U numpy  --prefer-binary

    # (outside of command line) put stable diffusion model into web ui directory
    # the command below must output something like: 1 File(s) 4,265,380,512 bytes
    dir model.ckpt

    The installation is finished, to start the web ui, run:

    python webui.py --api --nowebui
    ```

3.  CREATE `automatic1111_python_wrapper.py` and `tests.py`
    Here is an example script for how to use automatic1111. SPLIT the functionnality in 2, the utils in the wrapper, and the effective tests in the tests file

    ```python
    from datetime import datetime
    import urllib.request
    import base64
    import json
    import time
    import os

    webui_server_url = 'http://127.0.0.1:7860'

    out_dir = 'api_out'
    out_dir_t2i = os.path.join(out_dir, 'txt2img')
    out_dir_i2i = os.path.join(out_dir, 'img2img')
    os.makedirs(out_dir_t2i, exist_ok=True)
    os.makedirs(out_dir_i2i, exist_ok=True)


    def timestamp():
        return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")


    def encode_file_to_base64(path):
        with open(path, 'rb') as file:
            return base64.b64encode(file.read()).decode('utf-8')


    def decode_and_save_base64(base64_str, save_path):
        with open(save_path, "wb") as file:
            file.write(base64.b64decode(base64_str))


    def call_api(api_endpoint, **payload):
        data = json.dumps(payload).encode('utf-8')
        request = urllib.request.Request(
            f'{webui_server_url}/{api_endpoint}',
            headers={'Content-Type': 'application/json'},
            data=data,
        )
        response = urllib.request.urlopen(request)
        return json.loads(response.read().decode('utf-8'))


    def call_txt2img_api(**payload):
        response = call_api('sdapi/v1/txt2img', **payload)
        for index, image in enumerate(response.get('images')):
            save_path = os.path.join(out_dir_t2i, f'txt2img-{timestamp()}-{index}.png')
            decode_and_save_base64(image, save_path)


    def call_img2img_api(**payload):
        response = call_api('sdapi/v1/img2img', **payload)
        for index, image in enumerate(response.get('images')):
            save_path = os.path.join(out_dir_i2i, f'img2img-{timestamp()}-{index}.png')
            decode_and_save_base64(image, save_path)


    if __name__ == '__main__':
        payload = {
            "prompt": "masterpiece, (best quality:1.1), 1girl <lora:lora_model:1>",  # extra networks also in prompts
            "negative_prompt": "",
            "seed": 1,
            "steps": 20,
            "width": 512,
            "height": 512,
            "cfg_scale": 7,
            "sampler_name": "DPM++ 2M",
            "n_iter": 1,
            "batch_size": 1,

            # example args for x/y/z plot
            # "script_name": "x/y/z plot",
            # "script_args": [
            #     1,
            #     "10,20",
            #     [],
            #     0,
            #     "",
            #     [],
            #     0,
            #     "",
            #     [],
            #     True,
            #     True,
            #     False,
            #     False,
            #     0,
            #     False
            # ],

            # example args for Refiner and ControlNet
            # "alwayson_scripts": {
            #     "ControlNet": {
            #         "args": [
            #             {
            #                 "batch_images": "",
            #                 "control_mode": "Balanced",
            #                 "enabled": True,
            #                 "guidance_end": 1,
            #                 "guidance_start": 0,
            #                 "image": {
            #                     "image": encode_file_to_base64(r"B:\path\to\control\img.png"),
            #                     "mask": None  # base64, None when not need
            #                 },
            #                 "input_mode": "simple",
            #                 "is_ui": True,
            #                 "loopback": False,
            #                 "low_vram": False,
            #                 "model": "control_v11p_sd15_canny [d14c016b]",
            #                 "module": "canny",
            #                 "output_dir": "",
            #                 "pixel_perfect": False,
            #                 "processor_res": 512,
            #                 "resize_mode": "Crop and Resize",
            #                 "threshold_a": 100,
            #                 "threshold_b": 200,
            #                 "weight": 1
            #             }
            #         ]
            #     },
            #     "Refiner": {
            #         "args": [
            #             True,
            #             "sd_xl_refiner_1.0",
            #             0.5
            #         ]
            #     }
            # },
            # "enable_hr": True,
            # "hr_upscaler": "R-ESRGAN 4x+ Anime6B",
            # "hr_scale": 2,
            # "denoising_strength": 0.5,
            # "styles": ['style 1', 'style 2'],
            # "override_settings": {
            #     'sd_model_checkpoint': "sd_xl_base_1.0",  # this can use to switch sd model
            # },
        }
        call_txt2img_api(**payload)

        init_images = [
            encode_file_to_base64(r"B:\path\to\img_1.png"),
            # encode_file_to_base64(r"B:\path\to\img_2.png"),
            # "https://image.can/also/be/a/http/url.png",
        ]

        batch_size = 2
        payload = {
            "prompt": "1girl, blue hair",
            "seed": 1,
            "steps": 20,
            "width": 512,
            "height": 512,
            "denoising_strength": 0.5,
            "n_iter": 1,
            "init_images": init_images,
            "batch_size": batch_size if len(init_images) == 1 else len(init_images),
            # "mask": encode_file_to_base64(r"B:\path\to\mask.png")
        }
        # if len(init_images) > 1 then batch_size should be == len(init_images)
        # else if len(init_images) == 1 then batch_size can be any value int >= 1
        call_img2img_api(**payload)

        # there exist a useful extension that allows converting of webui calls to api payload
        # particularly useful when you wish setup arguments of extensions and scripts
        # https://github.com/huchenlei/sd-webui-api-payload-display
    ```

4.  MODIFY `automatic1111_python_wrapper.py`
    ADD Pydantic model for the Payload.

5.  CREATE the files for flux
    Here are basic instructions for that:
    Direct Flux Inference

    Install Flux:

    ```bash

    pip install flux-protocol
    ```

    Basic Inference Example:

    ```python

    import flux
    from PIL import Image

    def flux_txt2img(prompt, model_path="flux_model.bin"):
        config = flux.Config(
            model=model_path,
            dtype="float16",
            height=512,
            width=512,
            steps=20
        )

        pipeline = flux.TextToImagePipeline(config)
        result = pipeline.run(
            prompt=prompt,
            guidance_scale=7.5,
            seed=42
        )

        return Image.fromarray(result.image)

     # Usage

     image = flux_txt2img("A cyberpunk city at night")
     image.save("flux_output.png")
    ```

    Advanced Features Implementation:

         Image-to-Image:

    ```python

    def flux_img2img(prompt, init_image, strength=0.7):
        config = flux.Config(...)  # Same as before
        pipeline = flux.ImageToImagePipeline(config)
        result = pipeline.run(
            prompt=prompt,
            init_image=np.array(init_image),
            strength=strength
        )
        return Image.fromarray(result.image)
    ```

         LoRA Integration:

    ```python

    def apply_lora(model, lora_path, alpha=0.75):
        base_model = flux.load_model(model)
        lora_weights = flux.load_lora(lora_path)
        return flux.merge_lora(base_model, lora_weights, alpha)
    ```
