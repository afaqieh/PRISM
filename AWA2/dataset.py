import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from diffusers import AutoencoderKL, DDPMScheduler

SELECTED_ATTRS = [
    "black", "white", "blue", "brown", "gray", "orange", "red", "yellow",
    "patches", "spots", "stripes", "furry", "hairless", "toughskin",
    "big", "small", "bulbous", "lean",
    "flippers", "hands", "hooves", "pads", "paws",
    "longleg", "longneck", "tail", "horns", "claws", "tusks",
    "bipedal", "quadrapedal",
]


class AWA2Dataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        vae: AutoencoderKL,
        device: str = "cuda",
        base_seed: int = 0,
    ):
        super().__init__()

        df = pd.read_csv(csv_path)
        df = df[df["full_path"].apply(os.path.exists)].reset_index(drop=True)

        self.df        = df
        self.vae       = vae
        self.device    = device
        self.base_seed = int(base_seed)
        self.attr_names = SELECTED_ATTRS

        self.transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        class_list = sorted(self.df["class_name"].unique())
        self.class2idx = {c: i for i, c in enumerate(class_list)}

        self.scheduler = DDPMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="scheduler"
        )

        print(
            f"AWA2Dataset loaded: {len(self.df)} images, "
            f"{len(self.class2idx)} classes, "
            f"{len(self.attr_names)} attributes, "
            f"base_seed={self.base_seed}"
        )

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
            "class": torch.tensor(self.class2idx[row["class_name"]], dtype=torch.long),
        }
        for attr in self.attr_names:
            meta[attr] = torch.tensor(int(row[attr]), dtype=torch.long)

        return latents, noise, noisy_latents, t, meta
