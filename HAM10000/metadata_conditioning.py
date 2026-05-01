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

    @classmethod
    def for_ham10000(cls, num_dx: int, num_sites: int, num_sex: int, **kwargs):

        field_configs = [
            {"name": "dx",   "type": "categorical", "vocab_size": num_dx,    "emb_dim": 32},
            {"name": "site", "type": "categorical", "vocab_size": num_sites, "emb_dim": 32},
            {"name": "sex",  "type": "categorical", "vocab_size": num_sex,   "emb_dim": 16},
            {"name": "age",  "type": "continuous",                            "emb_dim": 16},
        ]
        return cls(field_configs, **kwargs)



def ham10000_field_configs(dataset) -> list:

    return [
        {
            "name":       "dx",
            "type":       "categorical",
            "vocab_size": len(dataset.dx2idx),
            "emb_dim":    32,
        },
        {
            "name":       "site",
            "type":       "categorical",
            "vocab_size": len(dataset.site2idx),
            "emb_dim":    32,
        },
        {
            "name":       "sex",
            "type":       "categorical",
            "vocab_size": len(dataset.sex2idx),
            "emb_dim":    16,
        },
        {
            "name":  "age",
            "type":  "continuous",
            "emb_dim": 16,
        },
    ]


def remap_ham10000_checkpoint(state_dict: dict) -> dict:

    mapping = {
        
        "dx_emb.weight":      "embeddings.dx.weight",
        "site_emb.weight":    "embeddings.site.weight",
        "sex_emb.weight":     "embeddings.sex.weight",
        
        "age_mlp.0.weight":   "embeddings.age.0.weight",
        "age_mlp.0.bias":     "embeddings.age.0.bias",
        "age_mlp.2.weight":   "embeddings.age.2.weight",
        "age_mlp.2.bias":     "embeddings.age.2.bias",
       
        "dx_proj.0.weight":   "projections.dx.0.weight",
        "dx_proj.0.bias":     "projections.dx.0.bias",
        "dx_proj.1.weight":   "projections.dx.1.weight",
        "dx_proj.1.bias":     "projections.dx.1.bias",
       
        "site_proj.0.weight": "projections.site.0.weight",
        "site_proj.0.bias":   "projections.site.0.bias",
        "site_proj.1.weight": "projections.site.1.weight",
        "site_proj.1.bias":   "projections.site.1.bias",
       
        "sex_proj.0.weight":  "projections.sex.0.weight",
        "sex_proj.0.bias":    "projections.sex.0.bias",
        "sex_proj.1.weight":  "projections.sex.1.weight",
        "sex_proj.1.bias":    "projections.sex.1.bias",
       
        "age_proj.0.weight":  "projections.age.0.weight",
        "age_proj.0.bias":    "projections.age.0.bias",
        "age_proj.1.weight":  "projections.age.1.weight",
        "age_proj.1.bias":    "projections.age.1.bias",
       
    }

    new_state = {}
    for k, v in state_dict.items():
        new_state[mapping.get(k, k)] = v 

    return new_state

