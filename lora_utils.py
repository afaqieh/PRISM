import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, original, r=4, alpha=1.0):
        super().__init__()
        self.original = original
        self.r = r
        self.alpha = alpha

        self.down = nn.Linear(original.in_features, r, bias=False)
        self.up = nn.Linear(r, original.out_features, bias=False)

        self.scaling = alpha / r

        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.original(x) + self.up(self.down(x)) * self.scaling
    
def apply_lora_to_unet(unet, r=4):
    lora_layers = []
    for name, module in unet.named_modules():
        if isinstance(module, nn.Linear):
            if any(keyword in name for keyword in ["to_q", "to_k", "to_v", "to_out"]):
                parent = unet
                parts = name.split(".")
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                last = parts[-1]

                orig_layer = getattr(parent, last)
                lora_layer = LoRALinear(orig_layer, r=r)

                setattr(parent, last, lora_layer)
                lora_layers.append(lora_layer)
    return lora_layers

def inject_metadata_into_attention(unet, device):

    def make_replacement(attn2):
        original_call = attn2.processor.__call__

        def new_call(attn, hidden_states, encoder_hidden_states=None, **kwargs):
            encoder_hidden_states = attn2.metadata_context
            return original_call(attn, hidden_states, encoder_hidden_states, **kwargs)

        return new_call

    for _, module in unet.named_modules():
        if hasattr(module, "attn2"):
            module.attn2.metadata_context = torch.zeros(1, 77, 768, device=device)
            module.attn2.processor.__call__ = make_replacement(module.attn2)


def apply_lora_to_clip(text_encoder, r=16):
    lora_layers = []
    for name, module in text_encoder.named_modules():
        if isinstance(module, nn.Linear):
            if any(k in name for k in ["q_proj", "k_proj", "v_proj", "out_proj"]):
                parent = text_encoder
                parts  = name.split(".")
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                orig = getattr(parent, parts[-1])
                lora = LoRALinear(orig, r=r)
                setattr(parent, parts[-1], lora)
                lora_layers.append(lora)
    return lora_layers