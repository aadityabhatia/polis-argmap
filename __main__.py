from dotenv import load_dotenv
import argparse
import sys
import signal
from tasks import getTask, taskList
import datetime

load_dotenv()

if __name__ == "__main__":

    # trap SIGINT and SIGTERM to ensure graceful exit
    def signal_handler(sig, frame):
        print(f"{datetime.datetime.now()} Signal {sig} received. Exiting.\n\n")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(
        prog="polis-argmap",
        description="Run tasks on Polis datasets using ArgMap",
        epilog="Source: https://github.com/aadityabhatia/polis-argmap"
    )

    parser.add_argument(
        '--datasets', '-d',
        help="Comma-separated list of datasets to process",
        type=str,
        required=True,
    )

    parser.add_argument(
        '--output', '-o',
        help="Output file for console log; default is stdout",
        type=argparse.FileType('a'),
        default=sys.stdout
    )

    parser.add_argument(
        'tasks',
        help="List of tasks to run, space-separated",
        type=str,
        nargs='+',
        choices=taskList,
    )

    args = parser.parse_args()
    datasets = args.datasets.split(",")
    sys.stdout = args.output

    print("=" * 80)
    print(f"{datetime.datetime.now()} Starting Polis ArgMap Tasks...", flush=True)

    for dataset in datasets:
        print(f"{datetime.datetime.now()} Dataset: {dataset}", flush=True)

        for task in args.tasks:
            try:
                print(f"{datetime.datetime.now()} Task: {task}", flush=True)
                Task = getTask(task)
                Task.run(dataset)
                print(f"{datetime.datetime.now()} Task Complete: {task}", flush=True)

            except Exception as e:
                print(f"{datetime.datetime.now()} ERROR: {e}")
                print(f"{datetime.datetime.now()} ERROR: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()

            print()

    print(f"{datetime.datetime.now()} All Datasets Processed.\n")
