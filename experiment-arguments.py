import datetime
print(f"{datetime.datetime.now()} Initializing...")

import math
import signal
import sys
import os
from tqdm import tqdm
import argparse
import logging

import polars as pl
from dotenv import load_dotenv

from argmap.helpers import getTorchDeviceVersion, printCUDAMemory, loadLanguageModel, requireGPU
from argmap.dataModel import DataModel, Summary, Comments, Topics
from argmap.guidance import generate_line, generate_phrase

import guidance
from guidance import instruction, user, assistant

# logging.basicConfig(
    # format='%(asctime)s %(levelname)s [%(name)s] %(message)s', level=logging.INFO)


load_dotenv()

# this allows categorical data from various sources to be combined and handled gracefully; performance cost is acceptable
pl.enable_string_cache()


@guidance
def generate_topic_arguments(lm, summary, topic, agreeableComments, argumentCount, arguments, temperature=0, progress_bar=None):

    if progress_bar:
        lm.echo = False

    with instruction():
        lm += f"""\
        First, identify unique areas of improvement from the statements.
        Then for each area, list the associated problems and the requested action from the statements. If no ACTIONABLE SOLUTIONS are present, output None.
        Using the problem, proposed action, and the statements, create unique, short, and convincing one-sentence arguments that urges the leaders for change or improvement.
        Avoid repetitive phrases.

        DISCUSSION QUESTION: {summary.get('conversation-description')}

        """

        lm += f"""\
        TOPIC: {topic['Title']}
        KEYWORDS: {', '.join(topic['Representation'])}

        STATEMENTS:
        """

        for commentId, commentText, agrees, disagrees in agreeableComments.select('commentId', 'commentText', 'agrees', 'disagrees').iter_rows():
            lm += f"{commentId}. {commentText} ({int(agrees * 100 / (agrees + disagrees))}% voters agree)\n"

        lm += f"""
        ---
        PROBLEMS IDENTIFIED: <comma-separated list of problems from the statements>
        ACTIONABLE SOLUTIONS: <comma-separated list of actionable solution from the statements, if any>
        ARGUMENT: <make a compelling argument in one sentence that urges the need for action>
        ARGUMENT LABEL: <short three word label that describes the argument>
        """

    with user():
        lm += f"List the {argumentCount} most important areas of improvements from these statements, each on a new line.\n"

    areasOfImprovement = []
    with assistant():
        for i in range(argumentCount):
            lm += "- " + generate_line('areaOfImprovement', temperature) + "\n"
            areasOfImprovement.append(lm['areaOfImprovement'])

    # for each keyword, generate an argument
    for argumentId, area in enumerate(areasOfImprovement):
        with user():
            lm += f"""\
            AREA OF IMPROVEMENT: {area}
            """

    # for i in range(argumentCount):
        with assistant():
            lm += f"""\
            PROBLEMS IDENTIFIED: {generate_phrase('problem', temperature, 100)}
            ACTIONABLE SOLUTIONS: {generate_phrase('solution', temperature, 100)}
            ARGUMENT: {generate_line('argument', temperature, 100)}
            ARGUMENT LABEL: {generate_phrase('argumentTitle', temperature, 20)}
            """

            arguments.addRow({
                'topicId': topic['Topic'],
                'argumentId': argumentId,
                'argumentTitle': lm['argumentTitle'],
                'argumentContent': lm['argument'],
                'thoughts': [[
                    f"AREA: {area}",
                    f"PROBLEMS: {lm['problem']}",
                    f"SOLUTIONS: {lm['solution']}"
                ]]
            })

            if progress_bar:
                progress_bar.update()

    return lm


argumentSchema = {
    'topicId': pl.Int64,
    'argumentId': pl.Int64,
    'argumentTitle': pl.String,
    'argumentContent': pl.String,
    'thoughts': pl.List(pl.String),
}


def run_generate_arguments(languageModel, summary, topic, agreeableComments, argumentCount, arguments):
    progress_bar = tqdm(
        total=argumentCount,
        desc=f"Topic {topic['Topic']}",
        unit="arguments",
        smoothing=0.1,
    )

    args = {
        'summary': summary,
        'topic': topic,
        'agreeableComments': agreeableComments,
        'argumentCount': argumentCount,
        'arguments': arguments,
        'progress_bar': progress_bar
    }

    languageModel + generate_topic_arguments(**args)

    arguments.save()
    progress_bar.close()


if __name__ == "__main__":

    # trap SIGINT and SIGTERM to ensure graceful exit
    def signal_handler(sig, frame):
        print(f"{datetime.datetime.now()} Exiting...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    OPENDATA_REPO_PATH = os.getenv("OPENDATA_REPO_PATH")
    DATA_PATH = os.getenv("DATA_PATH")
    CUDA_MINIMUM_MEMORY_GB = os.getenv("CUDA_MINIMUM_MEMORY_GB")
    MODEL_ID = os.getenv("MODEL_ID")
    MODEL_REVISION = os.getenv("MODEL_REVISION")

    if not os.path.exists(OPENDATA_REPO_PATH):
        logging.error(
            "Polis Dataset not found. Please clone from https://github.com/compdemocracy/openData and set OPENDATA_REPO_PATH environment variable.")
        sys.exit(2)

    parser = argparse.ArgumentParser(
        prog="ArgMap Experiment Runner",
        description="Run experiments on Polis datasets using ArgMap",
        epilog="Source: https://github.com/aadityabhatia/polis-argmap"
    )

    parser.add_argument(
        '--start', '-s',
        help="Start Topic",
        type=int,
        default=0
    )

    parser.add_argument(
        '--end', '-e',
        help="End Topic",
        type=int,
    )

    parser.add_argument(
        '--datasets', '-d',
        help="Comma-separated list of datasets to process",
        type=str,
        required=True,
        # default="american-assembly.bowling-green"
    )

    parser.add_argument(
        '--agreeabilityThreshold', '-a',
        help="Minimum agreeability",
        type=float,
        default=0.0
    )

    args = parser.parse_args()

    start = args.start
    end = args.end
    datasets = args.datasets.split(",")
    agreeabilityThreshold = args.agreeabilityThreshold

    if MODEL_ID is None:
        logging.error(
            "Required: HuggingFace Model ID using --model or MODEL_ID environment variable")
        sys.exit(2)

    print(getTorchDeviceVersion())
    printCUDAMemory()

    languageModel = loadLanguageModel(
        MODEL_ID, MODEL_REVISION, CUDA_MINIMUM_MEMORY_GB)

    # # list of all datasets
    # if os.path.exists(OPENDATA_REPO_PATH):
    #     datasets = [d for d in os.listdir(
    #         OPENDATA_REPO_PATH) if os.path.exists(f"{OPENDATA_REPO_PATH}/{d}/comments.csv")]

    # loop through datasets and experiments
    for dataset in datasets:
        print(f"{datetime.datetime.now()} Loading dataset: {dataset}...")

        try:
            comments = Comments(
                dataset, dataPath=DATA_PATH).load_from_parquet()
            agreeableCommentsDf = comments.df.filter(
                pl.col('moderated') == 1, pl.col('agreeability') > agreeabilityThreshold)
            topics = Topics(dataset, dataPath=DATA_PATH).load_from_parquet()
            summary = Summary(OPENDATA_REPO_PATH, dataset)

            arguments = DataModel(
                dataset,
                dataPath=DATA_PATH,
                schema=argumentSchema
            ).initialize()

            if start is None or start < 0:
                start = 0

            maxTopic = agreeableCommentsDf.get_column('topicId').max()

            if end is None or end > maxTopic:
                end = maxTopic

            for topicId in range(start, end + 1):

                topicCommentsDf = agreeableCommentsDf.filter(
                    pl.col('topicId') == topicId
                )

                args = {
                    'languageModel': languageModel,
                    'summary': summary,
                    'topic': topics.get(topicId),
                    'agreeableComments': topicCommentsDf,
                    'argumentCount': math.ceil(math.log(topicCommentsDf.height) * 2),
                    'arguments': arguments,
                }

                run_generate_arguments(**args)

        except Exception as e:
            logging.error(f"Error processing dataset: {dataset}")
            logging.error(e)

    print(f"{datetime.datetime.now()} All Experiments Complete.")
