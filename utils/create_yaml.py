# Creación del archivo yaml del dataset
import yaml
import os
import argparse
from pathlib import Path
from constants import FINDING_CLASS_MAP
"""

python create_yaml.py \
    --output /home/ivan/Downloads/ss_kvasir_dataset_access/out
"""
parser = argparse.ArgumentParser()
parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Carpeta base que contiene las subcarpetas train y val.",
    )

args = parser.parse_args()

yolo_names_dict = {i: value for i, value in enumerate(FINDING_CLASS_MAP.values())}

print(yolo_names_dict)


dataset_dir = args.output
filename_yml = "kvasir_class"
train_images_path = os.path.join(dataset_dir, 'images/train')
validation_images_path = os.path.join(dataset_dir, 'images/val')

data_yaml = {
    'path': str(dataset_dir),
    'train': train_images_path,
    'val': validation_images_path,
    'test': None,
    'nc': 14,
    'names': yolo_names_dict
}

yaml_path = str(dataset_dir) + f'/{filename_yml}.yaml'
with open(yaml_path, 'w') as f:
    yaml.dump(data_yaml, f, sort_keys=False)

print(f"Archivo YAML creado en {yaml_path}")