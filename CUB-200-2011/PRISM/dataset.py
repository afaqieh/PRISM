import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from diffusers import AutoencoderKL, DDPMScheduler


NUM_THROAT_COLORS   = 15
NUM_FOREHEAD_COLORS = 15
NUM_BELLY_COLORS    = 15
NUM_NAPE_COLORS     = 15


class CUBDataset(Dataset):
    def __init__(self, csv_path: str, vae: AutoencoderKL, device: str = "cuda"):
        super().__init__()

        df = pd.read_csv(csv_path)
        df = df[df["full_path"].apply(os.path.exists)].reset_index(drop=True)

        self.df      = df
        self.vae     = vae
        self.device  = device

        self.transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        species_list      = sorted(self.df["species_name"].unique())
        self.species2idx  = {s: i for i, s in enumerate(species_list)}

        self.num_throat_colors   = NUM_THROAT_COLORS
        self.num_forehead_colors = NUM_FOREHEAD_COLORS
        self.num_belly_colors    = NUM_BELLY_COLORS
        self.num_nape_colors     = NUM_NAPE_COLORS

        self.scheduler = DDPMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
        )

        print(f"CUBDataset loaded: {len(self.df)} images, "
              f"{len(self.species2idx)} species")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(row["full_path"]).convert("RGB")
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
        noise         = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, t)

        meta = {
            "species":        torch.tensor(
                                  self.species2idx[row["species_name"]],
                                  dtype=torch.long),
            "throat_color":   torch.tensor(int(row["throat_color"]),
                                           dtype=torch.long),
            "forehead_color": torch.tensor(int(row["forehead_color"]),
                                           dtype=torch.long),
            "belly_color":    torch.tensor(int(row["belly_color"]),
                                           dtype=torch.long),
            "nape_color":     torch.tensor(int(row["nape_color"]),
                                           dtype=torch.long),
        }

        return latents, noise, noisy_latents, t, meta
