import numpy as np

CONDITION_DISPLAY = {
    "Bacterial_spot":       "bacterial spot",
    "Early_blight":         "early blight",
    "Late_blight":          "late blight",
    "Leaf_Mold":            "leaf mold",
    "Septoria_leaf_spot":   "septoria leaf spot",
    "Spider_mites":         "spider mite infestation",
    "Target_Spot":          "target spot",
    "Yellow_Leaf_Curl_Virus": "yellow leaf curl virus",
    "mosaic_virus":         "mosaic virus",
    "healthy":              "healthy",
}


PLANTVILLAGE_PROMPTS = {
    "full": [
        "A close-up photograph of a {plant} leaf showing {condition}.",
        "A high-quality plant disease photograph of a {plant} leaf with {condition}.",
        "A detailed agricultural photograph of a {plant} plant leaf affected by {condition}.",
        "A field photograph of a {plant} leaf exhibiting {condition} symptoms.",
        "A macro photograph of a {plant} leaf with {condition} disease signs.",
        "An agricultural science photograph of a {plant} leaf: {condition}.",
        "A plant pathology photograph of a {plant} leaf showing {condition} infection.",
    ],
    "descriptive": [
        "A {plant} leaf with {condition}. Plant disease photograph.",
        "Plant photo: {plant} leaf affected by {condition}.",
        "Agricultural image: {plant} plant showing {condition} on its leaf.",
        "A field guide photograph of a {plant} leaf with {condition}.",
        "A {plant} photographed in natural setting — {condition} on leaf.",
    ],
    "compact": [
        "{plant} leaf, {condition}.",
        "{plant} plant: {condition} leaf.",
        "Photo of a {plant} leaf with {condition}.",
        "{plant} with {condition} on leaf.",
        "A {plant} leaf: {condition}.",
    ],
}

PLANTVILLAGE_HEALTHY_PROMPTS = {
    "full": [
        "A close-up photograph of a healthy {plant} leaf.",
        "A high-quality photograph of a {plant} plant with a healthy green leaf.",
        "A detailed agricultural photograph of a healthy {plant} leaf.",
        "A field photograph of a {plant} leaf with no disease signs.",
        "A macro photograph of a healthy {plant} leaf.",
        "An agricultural science photograph of a healthy {plant} leaf.",
        "A plant photograph of a {plant} leaf: healthy, no disease.",
    ],
    "descriptive": [
        "A healthy {plant} leaf. Plant photograph.",
        "Plant photo: {plant} leaf, healthy and green.",
        "Agricultural image: {plant} plant with healthy leaf.",
        "A field guide photograph of a healthy {plant} leaf.",
        "A {plant} photographed in natural setting — healthy leaf.",
    ],
    "compact": [
        "{plant} leaf, healthy.",
        "{plant} plant: healthy leaf.",
        "Photo of a healthy {plant} leaf.",
        "{plant} with healthy leaf.",
        "A healthy {plant} leaf.",
    ],
}


def create_prompt_plantvillage(row, inference=False):
    plant     = row["plant"]
    condition = row["condition"]

    is_healthy = (condition == "healthy")
    templates  = PLANTVILLAGE_HEALTHY_PROMPTS if is_healthy else PLANTVILLAGE_PROMPTS

    if inference:
        category = "full"
    else:
        category = np.random.choice(list(templates.keys()))

    template = np.random.choice(templates[category])

    if is_healthy:
        return template.format(plant=plant)
    else:
        condition_display = CONDITION_DISPLAY.get(condition, condition.replace("_", " "))
        return template.format(plant=plant, condition=condition_display)
