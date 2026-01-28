import yaml

def load_yaml_config(file_path: str):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)
