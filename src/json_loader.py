import json


def load_json(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found")
    except json.JSONDecodeError:
        raise json.JSONDecodeError(f"Error decoding JSON from {file_path}",
                                   doc="", pos=0)
    except Exception as e:
        raise Exception(f"Error loading JSON from {file_path}: {e}")


def save_json(data: dict, file_path: str) -> None:
    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found")
    except json.JSONDecodeError:
        raise json.JSONDecodeError(f"Error decoding JSON from {file_path}",
                                   doc="", pos=0)
    except Exception as e:
        raise Exception(f"Error saving JSON to {file_path}: {e}")
