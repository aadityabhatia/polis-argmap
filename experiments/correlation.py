from .experiment import Experiment
from argmap.guidance import generate_line

import os
import datetime
import polars as pl
import guidance
from guidance import instruction, user, assistant, select
from tqdm import tqdm
from argmap.helpers import loadLanguageModel
from argmap.dataModel import Comments, Arguments, ArgumentCommentMap


class Correlation(Experiment):

    @staticmethod
    def run(dataset):

        try:
            comments = Comments(dataset).load_from_parquet()
            arguments = Arguments(dataset).load_from_parquet()
            agreeableComments = comments.getAgreeableComments()
        except FileNotFoundError as e:
            print(f"{datetime.datetime.now()} Error: {e}")
            return

        argumentStatementPairCount = (
            agreeableComments
            .lazy()
            .group_by('topicId').len()
            .join(
                arguments.df.lazy()
                .group_by('topicId').len(), on='topicId')
            .select(
                pl.col('len')*pl.col('len_right'))
            .sum()
        ).collect().row(0)[0]

        if argumentStatementPairCount == 0:
            print(
                f"{datetime.datetime.now()} No argument-statement pairs found. Nothing to do.")
            return

        argumentCommentMap = ArgumentCommentMap(dataset, 'argumentCommentMap').initialize()
        languageModel = loadLanguageModel()

        topicList = agreeableComments.get_column(
            'topicId').unique().sort().to_list()

        progress_bar = tqdm(
            total=argumentStatementPairCount,
            desc="Correlating Argument-Statement Pairs",
            unit="pairs",
        )

        for topicId in topicList:

            topicComments = topicComments = agreeableComments.filter(
                pl.col('topicId') == topicId
            )
            topicArguments = arguments.df.filter(pl.col('topicId') == topicId)

            print(f"{datetime.datetime.now()} Correlating Topic {topicId}, {topicComments.height} comments, {topicArguments.height} arguments, {topicComments.height * topicArguments.height} pairs...")
            progress_bar.set_description(f"Correlating Topic {topicId}")

            args = {
                'topicId': topicId,
                'topicComments': topicComments,
                'topicArguments': topicArguments,
                'argumentCommentMap': argumentCommentMap,
                'thought': True,
                'reason': False,
                'progress_bar': progress_bar,
            }

            languageModel + guidance_topic_correlate(**args)

            argumentCommentMap.save()

        progress_bar.close()


@guidance
def guidance_topic_correlate(lm, topicId, topicComments, topicArguments, argumentCommentMap, thought=False, reason=False, context_reset=True, progress_bar=None):

    if progress_bar is not None:
        lm.echo = False

    temperature = int(os.getenv('MODEL_TEMPERATURE', 0))

    with instruction():
        lm += f"""\
        You will be presented a statement and an argument. Statement is a user-generated comment from a discussion. Argument is an actionable solution.

        TASK: Determine whether the statement supports, refutes, or is unrelated to the argument.
        SUPPORT: The argument is consistent with the statement. A person who agrees with the statement will definitely support the argument.
        REFUTE: The argument goes against the statement. A person who agrees with the statement will definitely with the argument.
        UNRELATED: The statement and argument are not directly related. Implementing the argument will not directly address the underlying issue.

        ---
        OUTPUT FORMAT
        THOUGHT: Deliberate on how strongly a person who agrees with the statement will support the argument.
        RELATIONSHIP: One of the following: SUPPORT, REFUTE, UNRELATED
        REASON: Provide a reason for your choice.
        """

    # iterate over each argument
    for argumentId, argumentTitle, argumentContent in topicArguments.select('argumentId', 'argumentTitle', 'argumentContent').iter_rows():
        with user():
            lm_argument = lm + f"""\
            ARGUMENT {argumentId}: {argumentTitle}
            {argumentContent}
            """

        # iterate over each comment
        for commentId, commentText, in topicComments.select('commentId', 'commentText').iter_rows():
            if context_reset:
                lm_argument + guidance_argument_correlate(topicId, argumentId, commentId, commentText,
                                                          argumentCommentMap, thought=thought, reason=reason)
            else:
                lm_argument = lm_argument + guidance_argument_correlate(
                    topicId, argumentId, commentId, commentText, argumentCommentMap, thought=thought, reason=reason)

            if progress_bar is not None:
                progress_bar.update()

    return lm


@guidance
def guidance_argument_correlate(lm, topicId, argumentId, commentId, commentText, argumentCommentMap, thought=False, reason=False):

    temperature = int(os.getenv('MODEL_TEMPERATURE', 0))

    with user():
        lm += f"""\
        STATEMENT {commentId}: {commentText}
        """

    with assistant():

        reasoning = []
        if thought:
            lm += f"THOUGHT: {generate_line('thought', temperature, 100)}\n"
            reasoning.append(f"THOUGHT: {lm['thought']}")

        lm += f"RELATIONSHIP: {select(['SUPPORT', 'REFUTE', 'UNRELATED'], name='relationship')}\n"
        relationship = lm['relationship']

        if reason and not relationship == 'UNRELATED':
            lm += f"REASON: {generate_line('reasoning', temperature, 100)}"
            reasoning.append(f"REASON: {lm['reasoning']}")

        argumentCommentMap.addRow({
            'commentId': commentId,
            'topicId': topicId,
            'argumentId': argumentId,
            'relationship': lm['relationship'],
            'reasoning': [reasoning],
        })

    return lm
