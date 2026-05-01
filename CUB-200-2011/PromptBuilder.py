import numpy as np

SPECIES_MAP = {
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

CUB_COLORS = [
    "blue", "brown", "iridescent", "purple", "rufous", "grey",
    "yellow", "olive", "green", "pink", "orange", "black", "white", "red", "buff",
]

CUB_PROMPTS = {
    "full": [
        "A wildlife photograph of a {species} with a {throat} throat, {forehead} forehead, {belly} belly, and {nape} nape.",
        "A high-quality bird photograph of a {species}: {throat} throat, {forehead} forehead, {belly} belly, {nape} nape.",
        "A close-up nature photo of a {species} showing {throat} throat, {forehead} forehead, {belly} belly, and {nape} nape coloring.",
        "A detailed wildlife photo of a {species} with {throat} throat plumage, {forehead} forehead, {belly} belly, and {nape} nape.",
        "An ornithological photograph of a {species} — throat: {throat}, forehead: {forehead}, belly: {belly}, nape: {nape}.",
        "A sharp field photograph of a {species} featuring {throat} throat, {forehead} forehead, {belly} underparts, and {nape} nape.",
        "A professional bird photography image of a {species} with {throat} throat, {forehead} crown, {belly} belly, and {nape} nape.",
    ],
    "descriptive": [
        "A {species} with {throat} throat and {forehead} forehead, {belly} belly, {nape} nape. Wildlife photograph.",
        "Bird photo: {species} with {throat} throat coloring, {forehead} forehead, {belly} belly, and {nape} nape.",
        "Wildlife image: {species} showing {throat} throat, {forehead} head, {belly} underparts, {nape} nape.",
        "A field guide photograph of a {species}: {throat} throat, {forehead} forehead, {belly} belly, {nape} nape.",
        "A {species} photographed in natural habitat — {throat} throat, {forehead} forehead, {belly} belly, {nape} nape.",
    ],
    "compact": [
        "{species}, {throat} throat, {forehead} forehead, {belly} belly, {nape} nape.",
        "{species} bird: {throat} throat, {forehead} forehead, {belly} belly, {nape} nape.",
        "Photo of a {species} with {throat} throat, {forehead} forehead, {belly} belly, {nape} nape.",
        "{species} with {throat} throat and {forehead} forehead.",
        "A {species}: {throat} throat, {forehead} forehead, {belly} belly.",
    ],
}


def idx_to_color(idx):
    idx = int(idx)
    if 0 <= idx < len(CUB_COLORS):
        return CUB_COLORS[idx]
    return "unknown"


def create_prompt_cub(row, inference=False):
    species_name = row["species_name"]
    display_name = SPECIES_MAP.get(species_name, species_name)

    throat   = idx_to_color(row["throat_color"])
    forehead = idx_to_color(row["forehead_color"])
    belly    = idx_to_color(row["belly_color"])
    nape     = idx_to_color(row["nape_color"])

    if inference:
        category = "full"
    else:
        category = np.random.choice(list(CUB_PROMPTS.keys()))

    template = np.random.choice(CUB_PROMPTS[category])
    return template.format(
        species=display_name,
        throat=throat,
        forehead=forehead,
        belly=belly,
        nape=nape,
    )
