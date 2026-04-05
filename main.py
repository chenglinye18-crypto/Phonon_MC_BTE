import argparse
from pathlib import Path

from phonon_mc import main as solver_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the phonon MC solver.")
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="",
        help="Input directory, or a base directory containing an input/ subdirectory.",
    )
    parser.add_argument(
        "--input-dir",
        dest="input_dir_flag",
        default="",
        help="Input directory, or a base directory containing an input/ subdirectory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    base_dir = args.input_dir_flag or args.input_dir
    solver_main(Path(base_dir).expanduser() if base_dir else Path(__file__).resolve().parent)
