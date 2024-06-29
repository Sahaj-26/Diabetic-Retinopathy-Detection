from pathlib import Path

def get_config():
    return {
        "batch_size": 64,
        "num_epochs": 20,
        "lr": 10**-4,
        "model_folder": "weights",
        "model_basename": "IVPUnet",
        "preload": "latest",
        "experiment_name": "runs/IVPUnet",
        "root_dir":"./EOphtha_45k_Dataset"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['model_basename']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def latest_weights_file_path(config):
    model_folder = f"{config['model_basename']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
