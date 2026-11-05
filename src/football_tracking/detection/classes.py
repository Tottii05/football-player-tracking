CLASS_TO_ID = {
    "player": 0,
    "goalkeeper": 1,
    "referee": 2,
    "ball": 3,
}

ID_TO_CLASS = {
    class_id: class_name
    for class_name, class_id in CLASS_TO_ID.items()
}


def get_class_name(class_id: int) -> str:
    """Return the canonical class name for an ID."""
    if isinstance(class_id, bool) or not isinstance(class_id, int):
        raise TypeError("class_id must be an integer.")

    if class_id not in ID_TO_CLASS:
        raise ValueError(f"Unknown class ID: {class_id}")

    return ID_TO_CLASS[class_id]


def get_class_id(class_name: str) -> int:
    """Return the canonical ID for a class name."""
    if not isinstance(class_name, str):
        raise TypeError("class_name must be a string.")

    normalized_name = class_name.strip().lower()

    if normalized_name not in CLASS_TO_ID:
        raise ValueError(f"Unknown class name: {class_name}")

    return CLASS_TO_ID[normalized_name]
