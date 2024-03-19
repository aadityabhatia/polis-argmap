from .task import Task

import guidance
from guidance import instruction, user, assistant, select

import datetime
from argmap.helpers import loadLanguageModel
from argmap.dataModel import DataModel, Summary, Comments, Topics, Votes, Arguments, ArgumentCommentMap
from argmap.guidance import generate_line, generate_phrase
import math
from tqdm import tqdm
import os
import polars as pl
from pprint import pprint

class ArgumentGeneration(Task):

    @staticmethod
    def run(dataset):
        try:
            comments = Comments(dataset).load_from_parquet()
            topics = Topics(dataset).load_from_parquet()
            summary = Summary(dataset)
        except FileNotFoundError as e:
            print(f"{datetime.datetime.now()} Error: {e}")
            return

        agreeableComments = (
            comments
            .getAgreeableComments()
        )

        arguments = Arguments(dataset, 'arguments').initialize()

        def countArguments(commentCount):
            return math.ceil(math.log(commentCount) * 2)

        commentCountByTopic = (
            agreeableComments
            .get_column('topicId')
            .value_counts()
            .sort('topicId')
            .get_column('count')
            .to_list())

        argumentCount = sum(countArguments(x) for x in commentCountByTopic)

        print(f"{datetime.datetime.now()} Generating {argumentCount} arguments from {agreeableComments.height} comments across {len(commentCountByTopic)} topics.", flush=True)

        languageModel = loadLanguageModel()

        progress_bar = tqdm(
            total=sum(countArguments(x) for x in commentCountByTopic),
            desc=f"Generating Arguments",
            unit="arguments",
            smoothing=0.1,
        )

        for topicId in range(comments.df.get_column('topicId').max()+1):

            topicComments = agreeableComments.filter(
                pl.col('topicId') == topicId
            )

            if topicComments.height == 0:
                continue

            progress_bar.set_description(f"Generating Arguments for Topic {topicId}")
            print(f"{datetime.datetime.now()} Topic {topicId}, {topicComments.height} comments, {countArguments(topicComments.height)} arguments.")

            args = {
                'summary': summary,
                'topic': topics.get(topicId),
                'agreeableComments': topicComments,
                'argumentCount': countArguments(topicComments.height),
                'arguments': arguments,
                'progress_bar': progress_bar,
            }

            languageModel + guidance_topic_arguments(**args)
            arguments.save()

        arguments.save()
        progress_bar.close()


@guidance
def guidance_topic_arguments(lm, summary, topic, agreeableComments, argumentCount, arguments, progress_bar=None):

    if progress_bar is not None:
        lm.echo = False

    temperature = int(os.getenv('MODEL_TEMPERATURE', 0))

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

    for argumentId, area in enumerate(areasOfImprovement):
        with user():
            lm += f"""\
            AREA OF IMPROVEMENT: {area}
            """

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

            if progress_bar is not None:
                progress_bar.update()

    return lm
