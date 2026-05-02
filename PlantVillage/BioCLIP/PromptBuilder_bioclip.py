import numpy as np

PLANT_SCIENTIFIC = {
    "Pepper": "Capsicum annuum",
    "Potato": "Solanum tuberosum",
    "Tomato": "Solanum lycopersicum",
}

PATHOGEN_SCIENTIFIC = {
    "Bacterial_spot":         "Xanthomonas vesicatoria",
    "Early_blight":           "Alternaria solani",
    "Late_blight":            "Phytophthora infestans",
    "Leaf_Mold":              "Passalora fulva",
    "Septoria_leaf_spot":     "Septoria lycopersici",
    "Spider_mites":           "Tetranychus urticae",
    "Target_Spot":            "Corynespora cassiicola",
    "Yellow_Leaf_Curl_Virus": "Tomato yellow leaf curl virus",
    "mosaic_virus":           "Tomato mosaic virus",
}

BIOCLIP_PROMPTS_PLANT = {
    "full": [
        "{common} {plant_sci} {pathogen}",
        "{common} ({plant_sci}) {pathogen}",
        "{plant_sci} {common} {pathogen}",
        "a photo of {common} {plant_sci} {pathogen}",
        "a plant photograph of {common} {plant_sci} {pathogen}",
        "{common} {plant_sci} — {pathogen}",
        "{plant_sci} — {pathogen}",
    ],
    "common": [
        "{common} {pathogen}",
        "a photo of {common} {pathogen}",
        "a plant photograph of {common} {pathogen}",
        "a leaf photograph of {common} {pathogen}",
        "{common} plant {pathogen}",
    ],
    "scientific": [
        "{plant_sci} {pathogen}",
        "a photo of {plant_sci} {pathogen}",
        "a plant photograph of {plant_sci} {pathogen}",
        "{plant_sci}, {pathogen}",
    ],
}

BIOCLIP_PROMPTS_PLANT_HEALTHY = {
    "full": [
        "{common} {plant_sci}",
        "{common} ({plant_sci}) healthy",
        "{plant_sci} {common}",
        "a photo of {common} {plant_sci}",
        "a plant photograph of {common} {plant_sci}",
        "{common} {plant_sci} — healthy",
        "{plant_sci} — healthy",
    ],
    "common": [
        "{common}",
        "a photo of {common}",
        "a plant photograph of {common}",
        "a healthy {common} plant",
        "{common} plant healthy",
    ],
    "scientific": [
        "{plant_sci}",
        "a photo of {plant_sci}",
        "a plant photograph of {plant_sci}",
        "{plant_sci}, healthy",
    ],
}


def create_prompt_bioclip_plantvillage(row, inference=False):
    plant     = row["plant"]
    condition = row["condition"]

    common    = plant
    plant_sci = PLANT_SCIENTIFIC.get(plant, plant)
    is_healthy = (condition == "healthy")

    templates = BIOCLIP_PROMPTS_PLANT_HEALTHY if is_healthy else BIOCLIP_PROMPTS_PLANT

    if inference:
        category = "full"
    else:
        category = np.random.choice(list(templates.keys()))

    template = np.random.choice(templates[category])

    if is_healthy:
        return template.format(common=common, plant_sci=plant_sci)
    else:
        pathogen = PATHOGEN_SCIENTIFIC.get(condition, condition.replace("_", " "))
        return template.format(common=common, plant_sci=plant_sci, pathogen=pathogen)
