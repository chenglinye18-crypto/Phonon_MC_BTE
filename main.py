from pathlib import Path

from phonon_mc import main as solver_main


if __name__ == "__main__":
    solver_main(Path(__file__).resolve().parent)
