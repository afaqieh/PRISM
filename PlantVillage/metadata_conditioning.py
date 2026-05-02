import torch
import torch.nn as nn


class MetadataConditionEncoder(nn.Module):
    def __init__(
        self,
        field_configs: list,
        hidden_dim:    int = 128,
        final_dim:     int = 768,
        seq_len:       int = 77,
    ):
        super().__init__()

        self.field_configs = field_configs
        self.seq_len       = seq_len
        self.final_dim     = final_dim

        self.embeddings  = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        total_emb_dim    = 0

        for cfg in field_configs:
            name    = cfg["name"]
            emb_dim = cfg["emb_dim"]
            total_emb_dim += emb_dim

            if cfg["type"] == "categorical":
                self.embeddings[name] = nn.Embedding(cfg["vocab_size"], emb_dim)
            elif cfg["type"] == "continuous":
                self.embeddings[name] = nn.Sequential(
                    nn.Linear(1, emb_dim),
                    nn.SiLU(),
                    nn.Linear(emb_dim, emb_dim),
                )
            else:
                raise ValueError(
                    f"Unknown field type '{cfg['type']}' for field '{name}'. "
                    f"Must be 'categorical' or 'continuous'."
                )

            self.projections[name] = nn.Sequential(
                nn.Linear(emb_dim, final_dim),
                nn.LayerNorm(final_dim),
            )

        self.summary_proj = nn.Sequential(
            nn.Linear(total_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, final_dim),
            nn.LayerNorm(final_dim),
        )

        n_content_tokens = len(field_configs) + 1
        n_padding        = seq_len - n_content_tokens
        if n_padding < 0:
            raise ValueError(
                f"Too many fields: {len(field_configs)} fields + 1 summary token "
                f"= {n_content_tokens} > seq_len={seq_len}. "
                f"Reduce number of fields or increase seq_len."
            )
        self.n_padding      = n_padding
        self.padding_tokens = nn.Parameter(torch.zeros(1, n_padding, final_dim))
        nn.init.normal_(self.padding_tokens, std=0.02)

        field_summary = ", ".join(
            f"{c['name']}({'cat' if c['type']=='categorical' else 'cont'}"
            f",dim={c['emb_dim']})"
            for c in field_configs
        )
        print(
            f"MetadataConditionEncoder: {len(field_configs)} fields "
            f"[{field_summary}] | "
            f"summary_dim={total_emb_dim} | "
            f"tokens={n_content_tokens} content + {n_padding} padding = {seq_len}"
        )

    def forward(self, meta_batch: dict) -> torch.Tensor:
        B      = next(iter(meta_batch.values())).shape[0]
        embs   = {}
        tokens = []

        for cfg in self.field_configs:
            name = cfg["name"]

            if cfg["type"] == "categorical":
                emb = self.embeddings[name](meta_batch[name])
            else:
                emb = self.embeddings[name](meta_batch[name].float().unsqueeze(-1))

            embs[name] = emb
            tok = self.projections[name](emb).unsqueeze(1)
            tokens.append(tok)

        concat      = torch.cat(list(embs.values()), dim=-1)
        tok_summary = self.summary_proj(concat).unsqueeze(1)
        tokens.append(tok_summary)

        if self.n_padding > 0:
            padding = self.padding_tokens.expand(B, -1, -1)
            tokens.append(padding)

        cond = torch.cat(tokens, dim=1)
        assert cond.shape == (B, self.seq_len, self.final_dim), \
            f"Output shape mismatch: {cond.shape} != ({B}, {self.seq_len}, {self.final_dim})"
        return cond


def plantvillage_field_configs(dataset) -> list:
    return [
        {
            "name":       "plant",
            "type":       "categorical",
            "vocab_size": dataset.num_plants,
            "emb_dim":    32,
        },
        {
            "name":       "condition",
            "type":       "categorical",
            "vocab_size": dataset.num_conditions,
            "emb_dim":    32,
        },
    ]
