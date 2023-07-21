# from huggingface_hub import snapshot_download

# local_dir = "./dog"
# snapshot_download(
#     "diffusers/dog-example",
#     local_dir=local_dir, repo_type="dataset",
#     ignore_patterns=".gitattributes",
# )

"""
accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
  --instance_data_dir=dog \
  --output_dir=/home/ssd3/lz/dataset/trained_checkpoint/dreambooth/ \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400 \
  --push_to_hub
"""

"""
CUDA_VISIBLE_DEVICES=7 python train_dreambooth.py   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4    --instance_data_dir=dog   --output_dir=/home/ssd3/lz/dataset/trained_checkpoint/dreambooth/   --instance_prompt="a photo of sks dog"   --resolution=512   --train_batch_size=1   --gradient_accumulation_steps=1 --learning_rate=5e-6   --lr_scheduler="constant"   --lr_warmup_steps=0   --max_train_steps=400 
"""
from diffusers import StableDiffusionPipeline
import torch

model_id = "/home/paper/diffusers/examples/dreambooth/trained_ckp/starbucks"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompt = "A photo of starbucks mug on a pink table"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

# image.save("dog-bucket.png")
image.save("starbuck_on_table.png")