import os
import json
import yaml
import argparse


class LoadFromFile(argparse.Action):
    """Load a configuration file and update the namespace"""

    def __call__(self, parser, namespace, values, option_string=None):
        if values.name.endswith("yaml") or values.name.endswith("yml"):
            with values as f:
                namespace.__dict__.update(yaml.load(f, Loader=yaml.FullLoader))
                return

        with values as f:
            input = f.read()
            input = input.rstrip()
            for lines in input.split("\n"):
                k, v = lines.split("=")
                typ = type(namespace.__dict__[k])
                v = typ(v) if typ is not None else v
                namespace.__dict__[k] = v

def get_args(arguments=None):
    """Get the arguments from the command line"""
    parser = argparse.ArgumentParser(description="Confgen", prefix_chars="--")
    parser.add_argument(
        "--conf",
        type=open,
        action=LoadFromFile,
        help="Use a configuration file, e.g. python conformer_generator.py --conf input.yaml",
    )
    args = parser.parse_args(args=arguments)
    return args

def save_argparse(args, filename):
    if filename.endswith("yaml") or filename.endswith("yml"):
        args = args.__dict__.copy()
        yaml.dump(args, open(filename, "w"))
    else:
        raise ValueError("Configuration file should end with yaml or yml")


def readPDBs(pdbFileList):
    pdblist = []
    with open(pdbFileList, "r") as f:
        for line in f:
            pdblist.append(line.strip())
    return sorted(pdblist)