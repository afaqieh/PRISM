
import numpy as np

SPECIES_SCIENTIFIC = {
    "American Pipit":          "Anthus rubescens",
    "Black and white Warbler": "Mniotilta varia",
    "Blue winged Warbler":     "Vermivora cyanoptera",
    "Canada Warbler":          "Cardellina canadensis",
    "Cape May Warbler":        "Setophaga tigrina",
    "Golden winged Warbler":   "Vermivora chrysoptera",
    "Kentucky Warbler":        "Geothlypis formosa",
    "Louisiana Waterthrush":   "Parkesia motacilla",
    "Magnolia Warbler":        "Setophaga magnolia",
    "Mourning Warbler":        "Geothlypis philadelphia",
    "Northern Waterthrush":    "Parkesia noveboracensis",
    "Pine Warbler":            "Setophaga pinus",
    "Prairie Warbler":         "Setophaga discolor",
    "Tennessee Warbler":       "Leiothlypis peregrina",
    "Yellow Warbler":          "Setophaga petechia",
}

SPECIES_DISPLAY = {
    "American Pipit":          "American Pipit",
    "Black and white Warbler": "Black-and-white Warbler",
    "Blue winged Warbler":     "Blue-winged Warbler",
    "Canada Warbler":          "Canada Warbler",
    "Cape May Warbler":        "Cape May Warbler",
    "Golden winged Warbler":   "Golden-winged Warbler",
    "Kentucky Warbler":        "Kentucky Warbler",
    "Louisiana Waterthrush":   "Louisiana Waterthrush",
    "Magnolia Warbler":        "Magnolia Warbler",
    "Mourning Warbler":        "Mourning Warbler",
    "Northern Waterthrush":    "Northern Waterthrush",
    "Pine Warbler":            "Pine Warbler",
    "Prairie Warbler":         "Prairie Warbler",
    "Tennessee Warbler":       "Tennessee Warbler",
    "Yellow Warbler":          "Yellow Warbler",
}

BIOCLIP_PROMPTS = {
    "full": [
        "{common} {scientific}",
        "{common}, {scientific}",
        "{scientific} ({common})",
        "a photo of {common} {scientific}",
        "a wildlife photograph of {common} {scientific}",
        "iNaturalist observation: {common} {scientific}",
        "{common} - {scientific}",
    ],
    "common_only": [
        "{common}",
        "a photo of a {common}",
        "a wildlife photograph of a {common}",
        "a bird photograph of a {common}",
        "{common}, a species of warbler",
    ],
    "scientific_only": [
        "{scientific}",
        "a photo of {scientific}",
        "a wildlife photograph of {scientific}",
        "iNaturalist: {scientific}",
    ],
}


def create_prompt_bioclip(row, inference=False):

    species_name = row["species_name"] if hasattr(row, "__getitem__") \
                   else getattr(row, "species_name")

    common     = SPECIES_DISPLAY.get(species_name, species_name)
    scientific = SPECIES_SCIENTIFIC.get(species_name, species_name)

    if inference:
        category = "full"
    else:
        category = np.random.choice(list(BIOCLIP_PROMPTS.keys()))

    template = np.random.choice(BIOCLIP_PROMPTS[category])
    return template.format(common=common, scientific=scientific)
