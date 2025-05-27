import pathlib
from addict import Dict
import minydra
from minydra.dict import MinyDict
import yaml
import subprocess
import datetime

@minydra.parse_args(verbose=0, allow_overwrites=False)
def parse_and_load_config(args: MinyDict):
    """
    Parses the arguments.
    """
    args.resolve().pretty_print()
    return Dict(args)


def iterate_until_unique(output_dir):
    """
    Checks if the directory exists. If so, creates a unique one. 
    """
    name, idx = output_dir.name, 1
    while output_dir.exists():
        output_dir = output_dir.parent / f"{name}_{idx}"
        idx += 1
    
    return output_dir

def make_output_dir(config: Dict):
    """
    Establishes a suitable output directory name.
    """
    N, beta_target = config.N, config.beta_target

    # DIRECTORY NAMING
    output_dir  = pathlib.Path(config.output_dir).resolve()
    output_dir = output_dir / f"N-{N}-beta_target-{beta_target}" 
    output_dir = output_dir / f"SIGMA-{config.SIGMA}-alpha-{config.alpha}-lr-{config.lr}-min_sqrt-{config.minimize_sqrt}"
    output_dir = output_dir / f"beta_start-{config.beta_model}-seed-{config.seed}"

    output_dir = iterate_until_unique(output_dir)
    output_dir.mkdir(parents=True)
    return output_dir


def get_git_revision_hash():
    """
    Get current git hash of the current code base

    Returns:
        str: git hash
    """
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()


def save_config(config: Dict):
    """
    Saves config. Adds: git commit,
    """
    config['git_commit'] = get_git_revision_hash()
    to_save = config.to_dict()
    output_path = config.output_path
    with open(pathlib.Path(output_path) / "config.yaml", "w") as f:
        yaml.safe_dump(to_save, f)


def log(str, logfile=None):
    str = f'[{datetime.datetime.now()}] {str}'
    print(str)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            print(str, file=f)