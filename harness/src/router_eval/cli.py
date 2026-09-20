import argparse
from .runner import run
from .report import write_report
from .live_runner import run_live
from .live_report import write_live_report
from .validate_live import validate_live_tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "report", "live-run", "live-report", "validate-live"])
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cap", type=float, default=2.0)
    args = parser.parse_args()
    if args.command == "run":
        run(args.root, args.output)
        print(write_report(args.output))
    elif args.command == "report":
        print(write_report(args.output))
    elif args.command == "validate-live":
        validate_live_tasks(args.root, args.output)
        print(args.output)
    elif args.command == "live-run":
        run_live(args.root, args.output, args.cap)
        print(write_live_report(args.output))
    else:
        print(write_live_report(args.output))


if __name__ == "__main__":
    main()
