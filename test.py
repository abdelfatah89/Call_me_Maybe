import json


def get_schema(file_path: str) -> str:
    with open(file_path, "r") as f:
        schema = json.load(f)
    return schema


print("Too many \" closing brackets")
