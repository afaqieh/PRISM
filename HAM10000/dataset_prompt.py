import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from diffusers import AutoencoderKL, DDPMScheduler
from PromptBuilder import create_prompt

class HAM10000Dataset(Dataset):

    def __init__(self, csv_path, img_dir, mode, vae: AutoencoderKL, device: str = "cuda"):
        super().__init__()

        df = pd.read_csv(csv_path)
        
        df = df.dropna(subset=["dx", "localization", "sex", "age"]).reset_index(drop=True)
        df["filename"] = df["image_id"].astype(str) + ".jpg"
        available = set(os.listdir(img_dir))
        df = df[df["filename"].isin(available)].reset_index(drop=True)

        self.df = df
        self.img_dir = img_dir
        self.vae = vae
        self.device = device
        self.mode = mode
        
        self.transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        
        self.dx2idx = {k: i for i, k in enumerate(sorted(self.df["dx"].unique()))}
        self.site2idx = {k: i for i, k in enumerate(sorted(self.df["localization"].unique()))}
        self.sex2idx = {k: i for i, k in enumerate(sorted(self.df["sex"].unique()))}
        
        self.scheduler = DDPMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        path = os.path.join(self.img_dir, row["filename"])
        img = Image.open(path).convert("RGB")
        img = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            latents = self.vae.encode(img).latent_dist.sample()
            latents = 0.18215 * latents
            latents = latents.detach()

        assert img.shape[-2:] == (512, 512)
        assert latents.shape[-2:] == (64, 64)

        latents = latents.squeeze(0)

        t = torch.randint(
            0,
            self.scheduler.num_train_timesteps,
            (1,),
            device=self.device,
            dtype=torch.long,
        )
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, t)

        if self.mode == "metadata":
            age_val = float(row["age"])
            if pd.isna(age_val):
                age_val = 0.0
            age_val = age_val / 100

            meta = {
                "dx": torch.tensor(self.dx2idx[row["dx"]], dtype=torch.long),
                "site": torch.tensor(self.site2idx[row["localization"]], dtype=torch.long),
                "sex": torch.tensor(self.sex2idx[row["sex"]], dtype=torch.long),
                "age": torch.tensor(age_val, dtype=torch.float32),
            }

            return latents, noise, noisy_latents, t, meta

        elif self.mode == "prompt":
            prompt = create_prompt('ham', row)
            return latents, noise, noisy_latents, t, prompt

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

