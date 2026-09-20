import argparse
from .runner import run
from .report import write_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "report"])
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args.root, args.output)
    print(write_report(args.output))


if __name__ == "__main__":
    main()
