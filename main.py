import json
import argparse
from trainer import train
import ipdb

def main():
    cmd_args = setup_parser().parse_args()
    param = load_json(cmd_args.config)
    args = vars(cmd_args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json
    
    # If --seed is explicitly passed via command line, override the json config
    if cmd_args.seed is not None:
        args['seed'] = [cmd_args.seed]
        
    train(args)


import os

def load_json(settings_path):
    candidates = [
        settings_path,
        os.path.join(os.path.dirname(__file__), settings_path),
        os.path.join("configs", os.path.basename(settings_path)),
        os.path.join("configs", "configs", os.path.basename(settings_path)),
        os.path.basename(settings_path)
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            with open(p) as data_file:
                return json.load(data_file)

    existing_files = os.listdir("configs") if os.path.exists("configs") else []
    raise FileNotFoundError(
        f"Cannot find config file '{settings_path}'. "
        f"Existing files in 'configs/': {existing_files}"
    )


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/finetune.json',
                        help='Json file of settings.')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed to override the config file.')

    # # optim
    # parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'sgd'])
    return parser


if __name__ == '__main__':
    main()
