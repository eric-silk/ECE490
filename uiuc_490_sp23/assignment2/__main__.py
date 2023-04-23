import argparse

from .newtons import main as newton_main
from .projected_gd import main as pgd_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Mutex group currently broken: https://github.com/python/cpython/issues/103711
    # parser.add_mutually_exclusive_group(required=True)
    parser.add_argument("--newton", action="store_true", help="Run the Newton Script")
    parser.add_argument(
        "--pgd",
        action="store_true",
        help="Run the Projected Gradient Descent Script",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.newton ^ args.pgd:
        raise ValueError("One (and only one) of --newton or --pgd is required")
    if args.newton:
        newton_main()
    elif args.pgd:
        pgd_main()
    else:
        # This shouldn't happen
        raise RuntimeError("Argparse failed to enforce Mutex group!")
