import argparse

from .newtons import main as newton_main
from .projected_gd import main as pgd_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="assignment2", description="Assignment 2 for ECE 490 SP2023"
    )
    # Mutex group currently broken: https://github.com/python/cpython/issues/103711
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--newton",
        action="store_true",
        help=(
            "Run the Newton Script, generates a fractal showing a color coding of which"
            " stationary point the algorithm arrives at based upon the starting point."
        ),
    )
    group.add_argument(
        "--pgd",
        action="store_true",
        help="Run the Projected Gradient Descent Script, solves a basic box constraint problem",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.newton:
        newton_main()
    elif args.pgd:
        pgd_main()
    else:
        # This shouldn't happen
        raise RuntimeError("Argparse failed to enforce Mutex group!")
