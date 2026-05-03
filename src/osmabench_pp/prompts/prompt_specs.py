from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptSpec:
    """Configuration that defines one prompt-generation regime."""

    scene_type: str
    default_raw_target: int
    default_final_target: int
    default_batch_size: int
    output_dir_name: str

    room_hints: list[str]
    scenario_hints: list[str]
    spatial_cues: list[str]
    forbidden_vague: list[str]
    object_hints: list[str]
    forbidden_object_patterns: list[str]
    risky_patterns: list[str]
    strong_patterns: list[str]

    min_words: int
    max_words: int
    min_explicit_counts: int
    min_spatial_cues: int
    min_total_objects: int
    max_total_objects: int | None
    min_sentences: int
    max_sentences: int

    system_prompt: str
    examples: str


COMMON_ROOM_HINTS = [
    "kitchen",
    "bedroom",
    "living room",
    "office",
    "dining room",
    "storage room",
    "bathroom",
    "hallway",
    "laundry room",
    "pantry",
    "home office",
    "workshop",
    "garage",
    "utility room",
    "entryway",
    "kids room",
    "guest room",
]


FURNITURE_SPEC = PromptSpec(
    scene_type="furniture",
    default_raw_target=300,
    default_final_target=20,
    default_batch_size=12,
    output_dir_name="generated_furniture_prompts",

    room_hints=[
        *COMMON_ROOM_HINTS,
        "studio apartment",
        "reception area",
        "meeting room",
        "waiting area",
    ],
    scenario_hints=[
        "central furniture cluster with surrounding furniture",
        "target furniture positioned between larger furniture items",
        "dense furniture arrangement with a clear central zone",
        "constrained movement space created by furniture placement",
        "furniture arranged along walls with a central passage",
        "front-to-back furniture ordering with stable room layout",
        "countable multi-furniture arrangement with exact quantities",
        "furniture layout with a dominant anchor object and supporting furniture",
        "room-scale furniture arrangement with narrow passageways",
        "symmetrical or near-symmetrical furniture layout",
    ],
    spatial_cues=[
        "under",
        "between",
        "behind",
        "next to",
        "near",
        "around",
        "left of",
        "right of",
        "in front of",
        "against the wall",
        "along the wall",
        "facing",
        "surrounding",
        "center of the room",
        "parallel to",
        "opposite",
    ],
    forbidden_vague=[
        "several",
        "some",
        "many",
        "few",
        "multiple",
        "various",
        "a couple of",
        "a few",
        "lots of",
        "a set of",
        "a group of",
    ],
    object_hints=[
        "table",
        "dining table",
        "desk",
        "chair",
        "dining chair",
        "office chair",
        "armchair",
        "sofa",
        "bench",
        "cabinet",
        "bookshelf",
        "shelf",
        "storage cabinet",
        "filing cabinet",
        "wardrobe",
        "dresser",
        "nightstand",
        "bed",
        "bunk bed",
        "coffee table",
        "console table",
        "sideboard",
        "counter",
        "island",
        "workbench",
        "storage rack",
        "tv stand",
        "credenza",
    ],
    forbidden_object_patterns=[
        "book", "books",
        "cup", "cups",
        "mug", "mugs",
        "bottle", "bottles",
        "phone", "phones",
        "tool", "tools",
        "toy", "toys",
        "plate", "plates",
        "lamp", "lamps",
        "container", "containers",
        "box", "boxes",
        "sign", "signs",
        "remote",
        "monitor", "monitors",
        "keyboard", "keyboards",
        "laptop", "laptops",
        "apple", "apples",
        "bag", "bags",
        "basket", "baskets",
    ],
    risky_patterns=[
        "umbrella stand",
        "ironing board",
        "mirror",
        "pegboard",
        "drying rack",
        "luggage rack",
        "crib",
        "changing table",
        "shoe rack",
        "coat rack",
        "foldable rack",
        "wall shelf",
        "wall shelves",
        "from the entrance",
        "from the doorway",
        "viewed from",
        "visible from",
        "partially occluded",
        "partially blocked",
        "occluded",
        "blocked from view",
        "partially visible",
        "layered depth",
        "foreground",
        "background",
    ],
    strong_patterns=[
        "between",
        "behind",
        "around",
        "facing",
        "surrounding",
        "parallel to",
        "opposite",
        "along the wall",
        "against the wall",
        "center of the room",
        "narrow passage",
        "constrained space",
        "corridor-like space",
        "u shape",
        "u-shaped",
    ],
    min_words=28,
    max_words=170,
    min_explicit_counts=4,
    min_spatial_cues=2,
    min_total_objects=5,
    max_total_objects=None,
    min_sentences=3,
    max_sentences=7,
    system_prompt="""
Generate compact, simulation-friendly indoor scene prompts for SceneSmith.

Rules:
- Output JSON only.
- Each prompt must be a natural-language scene description.
- Furniture-stage only: use furniture-scale objects and room-scale layout.
- Do not use small manipulable objects such as books, cups, bottles, boxes, phones, tools, toys, plates, lamps, containers, decorations, signs, or electronics.
- Prefer stable floor-supported furniture.
- Avoid narrow fragile objects and wall-mounted accessories.
- Use exact object counts whenever countable objects are mentioned.
- Avoid vague quantifiers.
- Avoid colors, materials, textures, style adjectives, and decorative details.
- Avoid narrative or cinematic wording.
- Avoid viewpoint-dependent wording.
- Avoid visual-effect wording such as occlusion, partial visibility, foreground, background, or layered depth.
- Use simple explicit geometric spatial relations.
- Prefer room-level geometric phrasing such as "in the center of the room", "around the table", "behind the sofa", "between two chairs", "next to the cabinet", "against the wall", "along the wall", "parallel to", "opposite".
- Prompts should describe a stable room layout useful for later diagnostic testing.
- Do not mention people, actions, or interaction verbs.
- Do not use IDs like chair#1.
""".strip(),
    examples="""
Examples of good prompts:

1. A dining room with 1 dining table in the center of the room, 6 dining chairs around it, 1 sideboard against the back wall, and 1 bench near the left wall. Two chairs on one side are placed closer together between the table and the sideboard. The bench is to the left of the sideboard. The sideboard is behind the table from the front half of the room.

2. A living room with 2 sofas facing each other, 1 coffee table between them, 2 armchairs near the outer sides of the sofas, and 1 console table against the far wall. The coffee table is centered between the sofas. The armchairs create a narrow passage between the sofa group and the far wall. The console table is behind the second sofa and parallel to it.
""".strip(),
)


MANIPULAND_SPEC = PromptSpec(
    scene_type="manipuland",
    default_raw_target=100,
    default_final_target=15,
    default_batch_size=12,
    output_dir_name="generated_manipuland_prompts",

    room_hints=COMMON_ROOM_HINTS,
    scenario_hints=[
        "one manipulable target associated with a stable furniture support",
        "one manipulable object under or inside a furniture structure",
        "one manipulable object in front of or behind a larger blocker",
        "countable furniture layout with one main target object",
        "simple support relation with one secondary small-object group",
        "one target object with one blocker and one support relation",
        "small object arrangement anchored by one large furniture item",
        "one target object near a furniture edge or support surface",
    ],
    spatial_cues=[
        "on",
        "under",
        "inside",
        "between",
        "behind",
        "next to",
        "near",
        "in front of",
        "against the wall",
        "along the wall",
    ],
    forbidden_vague=[
        "several",
        "some",
        "many",
        "few",
        "multiple",
        "various",
        "a couple of",
        "a few",
        "lots of",
    ],
    object_hints=[
        "box",
        "plastic container",
        "backpack",
        "suitcase",
        "helmet",
        "clock",
        "pillow",
        "blanket",
        "bucket",
        "oil can",
        "car jack",
        "drill",
        "tape measure",
        "wrench",
        "plate",
        "book",
        "books",
        "laptop",
        "microwave",
        "towel",
        "towels",
        "stuffed rabbit",
        "stuffed rabbits",
    ],
    forbidden_object_patterns=[],
    risky_patterns=[
        "a pile of",
        "scattered",
        "messy",
        "cluttered with many",
        "randomly placed",
        "loosely placed",
        "stacked irregularly",
        "partially visible from",
        "from the doorway",
        "from the entrance",
        "from the side",
        "from the front",
        "direct access",
        "occluded",
        "partially occluded",
        "layered depth",
        "foreground",
        "background",
    ],
    strong_patterns=[
        "under",
        "inside",
        "behind",
        "in front of",
        "between",
    ],
    min_words=26,
    max_words=170,
    min_explicit_counts=3,
    min_spatial_cues=2,
    min_total_objects=4,
    max_total_objects=18,
    min_sentences=3,
    max_sentences=7,
    system_prompt="""
Generate compact, simulation-friendly indoor scene prompts for SceneSmith.

Rules:
- Output JSON only.
- Each prompt must be a natural-language scene description.
- Each scene must contain furniture and may contain a small number of manipulable objects.
- Furniture must define the main room layout.
- Include exactly one main manipulable target object.
- Allow at most one small secondary object group in addition to the main target.
- Use exact object counts whenever countable objects are mentioned.
- Avoid vague quantifiers.
- Avoid colors, materials, textures, style adjectives, and decorative details.
- Avoid narrative or cinematic wording.
- Avoid viewpoint-dependent wording.
- Avoid occlusion wording.
- Do not describe visual effects; describe only geometry and placement.
- Use simple explicit spatial relations such as "on the desk", "under the table", "inside the locker", "in front of the cabinet", "behind the chair", "next to the shelf".
- Prefer stable floor-supported furniture as anchors.
- Keep prompts suitable for simulation and later question generation.
- Do not use IDs like chair#1.
""".strip(),
    examples="""
Examples of good prompts:

1. An office with 1 desk and 2 chairs placed on opposite sides. On the desk, there is 1 laptop next to 3 books stacked near one side. Under the desk, there is 1 box placed near the left side. One chair stands in front of the desk and the other stands behind it. The box is under the desk between the desk legs and behind the front chair.

2. Inside a garage, there is 1 metal shelf with 3 buckets on the bottom level. Under the shelf, 1 car jack is positioned near the left support post. A stack of 2 tires stands in front of the car jack. On the middle shelf, 1 oil can is placed. The shelf is against the wall and the tires are between the car jack and the open floor area.
""".strip(),
)


SPECS = {
    "furniture": FURNITURE_SPEC,
    "manipuland": MANIPULAND_SPEC,
}


def get_prompt_spec(scene_type: str) -> PromptSpec:
    """Return prompt-generation specification by scene type."""

    try:
        return SPECS[scene_type]
    except KeyError as exc:
        valid = ", ".join(sorted(SPECS))
        raise ValueError(f"Unknown scene type: {scene_type}. Valid values: {valid}") from exc


def build_generation_prompt(spec: PromptSpec, batch_size: int, room_hint: str, scenario_hint: str) -> str:
    """Build the user prompt passed to the LLM."""

    return f"""
Generate {batch_size} different natural-language indoor scene prompts.

Soft room hint:
{room_hint}

Soft scenario hint:
{scenario_hint}

Requirements:
- Use exact counts for all mentioned objects.
- Avoid vague wording.
- Avoid colors, materials, textures, decoration, or style language.
- Avoid viewpoint-dependent wording.
- Avoid occlusion and visibility wording.
- Use clear geometric spatial relations.
- Keep prompts suitable for SceneSmith generation and later VQA.

{spec.examples}

Return JSON with this shape:
{{
  "prompts": [
    {{"prompt": "..."}},
    {{"prompt": "..."}}
  ]
}}
""".strip()
