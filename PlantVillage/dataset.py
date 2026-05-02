import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from diffusers import AutoencoderKL, DDPMScheduler

NUM_PLANTS     = 3
NUM_CONDITIONS = 10


class PlantVillageDataset(Dataset):
    def __init__(self, csv_path: str, vae: AutoencoderKL, device: str = "cuda"):
        super().__init__()

        df = pd.read_csv(csv_path)
        df = df[df["full_path"].apply(os.path.exists)].reset_index(drop=True)

        self.df     = df
        self.vae    = vae
        self.device = device

        self.transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        class_list        = sorted(self.df["class_name"].unique())
        self.class2idx    = {c: i for i, c in enumerate(class_list)}

        plant_list        = sorted(self.df["plant"].unique())
        self.plant2idx    = {p: i for i, p in enumerate(plant_list)}

        condition_list        = sorted(self.df["condition"].unique())
        self.condition2idx    = {c: i for i, c in enumerate(condition_list)}

        self.num_plants     = len(self.plant2idx)
        self.num_conditions = len(self.condition2idx)

        self.scheduler = DDPMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
        )

        print(f"PlantVillageDataset loaded: {len(self.df)} images, "
              f"{len(self.class2idx)} classes, "
              f"{self.num_plants} plants, {self.num_conditions} conditions")

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
            "plant":     torch.tensor(
                             self.plant2idx[row["plant"]],
                             dtype=torch.long),
            "condition": torch.tensor(
                             self.condition2idx[row["condition"]],
                             dtype=torch.long),
        }

        return latents, noise, noisy_latents, t, meta
