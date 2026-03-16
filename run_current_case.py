from pathlib import Path

from phonon_mc import run_current_case


if __name__ == "__main__":
    run_current_case(base_dir=Path(__file__).resolve().parent)
