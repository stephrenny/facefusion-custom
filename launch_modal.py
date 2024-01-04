from modal import Image, Stub, Secret, Volume, web_endpoint, gpu
from facefusion.download import conditional_download
from fastapi import UploadFile, File, Form, HTTPException
from typing import Optional

# This file lives in the base repo because we need to access facefusion specific functionalities
model_urls = [
    "https://github.com/facefusion/facefusion-assets/releases/download/models/open_nsfw.onnx",
    "https://github.com/facefusion/facefusion-assets/releases/download/models/retinaface_10g.onnx",
    "https://github.com/facefusion/facefusion-assets/releases/download/models/yunet_2023mar.onnx",
    "https://github.com/facefusion/facefusion-assets/releases/download/models/arcface_w600k_r50.onnx",
    "https://github.com/facefusion/facefusion-assets/releases/download/models/arcface_simswap.onnx",
    "https://github.com/facefusion/facefusion-assets/releases/download/models/gender_age.onnx",
    "https://github.com/facefusion/facefusion-assets/releases/download/models/face_occluder.onnx",
    "https://github.com/facefusion/facefusion-assets/releases/download/models/face_parser.onnx",
    "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx",
    "https://github.com/facefusion/facefusion-assets/releases/download/models/gfpgan_1.4.onnx",
    "https://github.com/facefusion/facefusion-assets/releases/download/models/real_esrgan_x2plus.pth",
]

install_commands = [
    "python install.py --torch cuda --onnxruntime cuda --skip-venv",
]

def download_models():
    download_directory_path = ('/facefusion-custom/.assets/models') # remove hardcode
    conditional_download(download_directory_path, model_urls)

image = (
    Image.debian_slim()
    .apt_install("python3.10", "python-is-python3", "pip", "git", "curl", "ffmpeg")
    .workdir("/facefusion-custom")
    .run_commands("git clone https://github.com/stephrenny/facefusion-custom .")
    .run_commands(install_commands)
    .pip_install(["Pillow", "boto3"])
    .run_function(download_models)
    .run_commands("pwd", force_build=True) # Dev purposes
    .workdir('facefusion-custom')
    .run_commands("git pull")
)

vol = Volume.persisted("alias-sources")
stub = Stub("alias-faceswap-endpoint-prod-v1", image=image)

@stub.function(secret=Secret.from_name("aws-s3-secret"), gpu=gpu.T4(memory=16), volumes={"/face-sources": vol})
@web_endpoint(method="POST")
async def swap_face(user_id: Optional[str] = Form(None), 
              source_image_id: str = Form(...), 
              target_image: UploadFile = File(...)):
    import os
    import boto3
    from PIL import Image
    import io
    import uuid
    from pathlib import Path
    from facefusion import core
    import torch

    s3 = boto3.client("s3")
    bucket_name = 'faceswap-outputs'

    vol.reload()

    # Directory setup
    os.chdir("/facefusion-custom")
    targets_dir = Path("tmp/targets")
    outputs_dir = Path("tmp/outputs")
    
    os.makedirs("tmp", exist_ok=True)
    os.makedirs(targets_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    def save_as_jpeg(img, target_path):
        # If the image has an alpha channel, convert it to RGB
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            img = img.convert('RGB')

        img.save(target_path, 'JPEG')

    def get_saved_source(source_image_id: str, user_id: Optional[str]):
        sources_path = Path("/face-sources")
        user_path = sources_path / user_id if user_id else sources_path
        source_path = user_path / f"{source_image_id}.jpeg"
        return source_path
    
    # Get the source image (user's face)
    source_filename = get_saved_source(source_image_id, user_id)
    if not os.path.isfile(source_filename):
        raise HTTPException(status_code=404, detail=f"Base photo with id {source_image_id} for user {user_id} was not found.")
    
    # Write target image to disk
    target_image_contents = await target_image.read()
    image_stream = io.BytesIO(target_image_contents)
    target_image = Image.open(image_stream)
    target_filename = targets_dir / f"{uuid.uuid4()}.jpeg"
    save_as_jpeg(target_image, target_filename)

    # Reserve a filename for the output
    output_filename = outputs_dir / f"{uuid.uuid4()}.jpeg"

    print("Is cuda available", torch.cuda.is_available())
    
    # Run facefusion
    args = [
    '-s', str(source_filename),
    '-t', str(target_filename),
    '-o', str(output_filename),
    '--execution-providers', 'cuda',
    '--headless'
    ]
    core.cli(args)

    # Upload the file
    aws_filename = f"{uuid.uuid4()}.jpeg"

    s3.upload_file(output_filename, bucket_name, aws_filename)

    # Retrieve signed url
    signed_url = s3.generate_presigned_url('get_object', Params={'Bucket': bucket_name, 'Key': aws_filename}, ExpiresIn=604800)

    return {"image_url": signed_url, "s3_uri": f"s3://{bucket_name}/{aws_filename}"}