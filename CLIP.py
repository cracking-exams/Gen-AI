import torch
from torchvision import datasets, transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

# --- Load Stable Diffusion ---
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# --- Generate 2 airplane images ---
prompts = [
    "an aeroplane flying in the sky",
    "an aeroplane on the runway"
]
gen_images = [pipe(p, num_inference_steps=25).images[0] for p in prompts]

# --- Load 3–4 airplane images from CIFAR-10 ---
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor()
])
cifar10 = datasets.CIFAR10(root=".", train=True, download=True, transform=transform)
real_airplanes = [img for img, label in cifar10 if label == 0][:4]  # 0 = airplane class

# --- Prepare tensors for FID ---
to_tensor = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor()
])
fake_tensors = [to_tensor(img) * 255 for img in gen_images]
real_tensors = [img * 255 for img in real_airplanes]
# --- Optional: show images ---
fig, axs = plt.subplots(2, 4, figsize=(12,6))
for i, img in enumerate(gen_images):
    axs[0,i].imshow(img)
    axs[0,i].set_title("Generated")
    axs[0,i].axis("off")

for i, img in enumerate(real_airplanes):
    axs[1,i].imshow(img.permute(1,2,0))
    axs[1,i].set_title("CIFAR-10 Real")
    axs[1,i].axis("off")
plt.tight_layout()
plt.show()

# --- Compute FID ---
fid = FrechetInceptionDistance(feature=64)
fid.update(torch.stack(real_tensors).to(torch.uint8), real=True)
fid.update(torch.stack(fake_tensors).to(torch.uint8), real=False)

score = fid.compute()
print(f"FID Score (Generated Airplanes vs CIFAR-10 Airplanes): {score.item():.2f}")


