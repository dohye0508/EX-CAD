"""Execution modes and workflow routing."""

MODES = {
    "SCRATCH_SCULPT": "Default. Start from a primitive and sculpt; no external meshes or generators.",
    "REFERENCE_SCULPT": "Scratch sculpt driven by user/online references (images, video, drawings). No mesh downloads.",
    "EXISTING_MESH_EDIT": "Edit a mesh the user supplied themselves.",
    "ASSET_ASSISTED": "Only with explicit user permission: use licensed online 3D assets (check license/author).",
    "NEURAL_GENERATION_ASSISTED": "Only with explicit user permission: TripoSG/TripoSR etc. Never auto-selected.",
}
ASSISTED = {"ASSET_ASSISTED", "NEURAL_GENERATION_ASSISTED"}

ORGANIC = {"face", "head", "human", "body", "animal", "creature", "monster", "insect", "plant", "organ", "rock",
           "terrain", "tree", "얼굴", "머리", "동물", "괴물", "생물", "식물", "바위", "지형"}
HARD = {"machine", "vehicle", "tool", "robot", "gear", "part", "bolt", "building", "architecture", "furniture",
        "기계", "차량", "도구", "로봇", "부품", "기어", "건축"}


class ModeGuard:
    def __init__(self, mode="SCRATCH_SCULPT", user_permissions=()):
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode}")
        self.mode = mode
        self.permissions = set(user_permissions)

    def require(self, op):
        if op in ("import_mesh", "download_asset") and self.mode not in ("EXISTING_MESH_EDIT", "ASSET_ASSISTED"):
            raise PermissionError(f"{op} is blocked in {self.mode}")
        if op == "neural_generate" and (self.mode != "NEURAL_GENERATION_ASSISTED" or "neural" not in self.permissions):
            raise PermissionError("neural generation requires NEURAL_GENERATION_ASSISTED mode and explicit permission")
        if self.mode in ASSISTED and self.mode.split("_")[0].lower() not in self.permissions:
            raise PermissionError(f"{self.mode} needs explicit user permission")


def route(subject):
    s = subject.lower()
    if any(k in s for k in HARD):
        return {"workflow": "hard_surface", "tools": ["edit_mode", "modifiers", "boolean", "bevel", "curves"],
                "start": "cube_or_cylinder"}
    if any(k in s for k in ORGANIC):
        return {"workflow": "organic_sculpt", "tools": ["sculpt", "sdf", "multires"], "start": "icosphere"}
    return {"workflow": "organic_sculpt", "tools": ["sculpt", "edit_mode"], "start": "icosphere"}
