from argmap.dataModel import DataModel, Summary, Comments
from argmap.helpers import loadLanguageModel
from argmap.guidance import generate_line

import polars as pl
from tqdm import tqdm
import guidance
from guidance import models, gen, select, instruction, user, assistant

import datetime
import os

from .task import Task


class Moderation(Task):
    def __init__(self, param):
        self.param = param

    @staticmethod
    def run(dataset):
        summary = Summary(dataset)
        comments = Comments(dataset).load_from_csv()

        print(f"""\
Dataset: {dataset}
Topic: {summary.topic}
Comments: {comments.df.height}""")

        languageModel = loadLanguageModel()

        MODERATION_REASON = os.getenv("MODERATION_REASON", False)

        experiments = []

        classify_options_simple = ['ACCEPT', 'UNSURE', 'REJECT']
        classify_options_detailed = [
            'ACCEPT', 'IRRELEVANT', 'SPAM', 'UNPROFESSIONAL', 'SCOPE', 'COMPLEX', 'UNSURE']

        i = 1

        for semantic_extraction in [False, True]:
            for thought in [False, True]:
                for classify_options in [classify_options_simple, classify_options_detailed]:
                    args = {
                        'summary': summary,
                        'comments_df': comments.df.sort('timestamp'),
                        'classify_options': classify_options,
                        'semantic_extraction': semantic_extraction,
                        'thought': thought,
                        'reason': MODERATION_REASON,
                    }

                    run_experiment(dataset, languageModel, i, args)
                    i += 1


results_schema = {
    'commentId': pl.UInt16,
    'languageModelModerated': pl.Int8,
    'classification': pl.Categorical,
    'thoughts': pl.List(pl.String),
}


def run_experiment(dataset, languageModel, experiment, args):
    print()
    print(f"{datetime.datetime.now()} Running Experiment {experiment}...")

    progress_bar = tqdm(
        total=args['comments_df'].height,
        desc=f'Moderation Experiment {experiment}',
        unit="comments",
        smoothing=0.1,
    )

    moderationResults = DataModel(
        dataset, f'moderation-results-{experiment}', schema=results_schema).initialize()

    args['results'] = moderationResults
    args['progress_bar'] = progress_bar

    languageModel + guidance_moderation(**args)

    moderationResults.save()
    progress_bar.close()

    print(f"{datetime.datetime.now()} Experiment {experiment} Complete.")


@guidance
def guidance_moderation(lm, summary, comments_df, classify_options, results, semantic_extraction=False, thought=False, reason=False, temperature=0, progress_bar=None):

    if progress_bar is not None:
        lm.echo = False

    # output the task instructions
    with instruction():
        lm += create_instructions(summary, classify_options,
                                  semantic_extraction, thought, reason)

    # iterate through the comments and ask the assistant to moderate each one
    for commentId, commentText in comments_df.select('commentId', 'commentText').iter_rows():
        lm + guidance_moderation_comment(commentId, commentText, classify_options,
                                         results, semantic_extraction, thought, reason, temperature)

        progress_bar.update() if progress_bar is not None else None

    return lm


@guidance
def guidance_moderation_comment(lm, commentId, commentText, classify_options, results, semantic_extraction, thought, reason, temperature):
    with user():
        lm += f"COMMENT {commentId}: {commentText}"

    with assistant():
        moderated = None
        thoughts = []

        if semantic_extraction:
            lm += "PROBLEM: " + generate_line('problem', temperature) + "\n"
            lm += "ACTION: " + generate_line('suggestion', temperature) + "\n"
            lm += "HOW MANY IDEAS: " + \
                generate_line('complexity', temperature) + "\n"
            thoughts += [
                f"PROBLEM: {lm['problem']}",
                f"ACTION: {lm['suggestion']}",
                f"IDEAS: {lm['complexity']}",
            ]

        if thought:
            lm += "THOUGHT: " + generate_line("thought", temperature) + "\n"
            thoughts += [f"THOUGHT: {lm['thought']}"]

        lm += "CLASSIFICATION: " + \
            select(classify_options, name="classification") + "\n"
        classification = lm['classification']

        if classification == classify_options[0]:
            moderated = 1
        else:
            moderated = 0 if classification == classify_options[1] else -1
            if reason:
                lm += "REASON: " + \
                    generate_line("classification_reason", temperature) + "\n"
                thoughts += [f"REASON: {lm['classification_reason']}"]

        results.addRow({
            'commentId': commentId,
            'languageModelModerated': moderated,
            'classification': classification,
            'thoughts': [thoughts],
        })

    return lm


guidelines = {
    'ACCEPT': [
        'mentions a real problem related to the discussion',
        'recommends a realistic and actionable solution related to the discussion',
        'makes a sincere suggestion related to the discussion',
    ],
    'REJECT': [
        'frivolous, irrelevant, unrelated to the discussion',
        'does not contribute to the discussion in a meaningful way',
        'incoherent or lacks seriousness',
        'provides neither a problem nor a solution',
        'the language is informal, colloquial, disrespectful or distasteful',
        'cannot be addressed within the scope of original question',
        'introduces multiple ideas, even if they are related to the discussion',
        'discusses distinct problems, making it difficult to determine where another person would agree or disagree',
    ],
    'IRRELEVANT': [
        'frivolous, irrelevant, unrelated to the discussion',
        'does not contribute to the discussion in a meaningful way',
    ],
    'SPAM': [
        'incoherent or lacks seriousness',
        'provides neither a problem nor a solution',
    ],
    'UNPROFESSIONAL': [
        'the language is informal, colloquial, disrespectful or distasteful',
    ],
    'SCOPE': [
        'cannot be addressed within the scope of original question',
    ],
    'COMPLEX': [
        'introduces multiple ideas, even if they are related to the discussion',
        'discusses distinct problems, making it difficult to determine where another person would agree or disagree',
    ],
    'UNSURE': [
        'may be accepted if it appears somewhat related to the discussion',
    ],
}


def create_instructions(summary, classify_options, thought=False, semantic_extraction=False, reason=False):
    global guidelines

    instructions = f"""\
Discussion Title: {summary.topic}
Discussion Question: {summary.get('conversation-description')}

---
Classify each comment objectively based on the following guidelines:
"""

    # for each item in classify_options, add the classification and the guidelines
    for classification in classify_options:
        items = guidelines[classification]
        for item in items:
            instructions += f"- {classification}: {item}.\n"

    instructions += f"""\
---
Output format:
"""

    if semantic_extraction:
        instructions += f"""\
PROBLEM: The specific problem mentioned in the comment. If only an action is suggested and no problem is explicitly mentioned, state None.
ACTION: What suggestion or change is proposed. If only a problem is mentioned and no action is suggested, state None.
HOW MANY IDEAS: Number of distinct ideas introduced in the comment.
"""

    if thought:
        instructions += f"THOUGHT: Deliberate about how the comment should be classified." + "\n"

    instructions += f"""\
CLASSIFICATION: One of the following based on given guidelines: {", ".join(classify_options)}.
"""

    if reason:
        instructions += f"REASON: Provide an explanation for the classification." + "\n"

    return instructions
