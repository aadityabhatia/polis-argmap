from pprint import pprint
from dotenv import load_dotenv
from dataModel import DataModel, Summary, Comments
import polars as pl
from tqdm import tqdm
import os
import sys
import guidance
from guidance import models, gen, select, instruction, user, assistant
import torch
import signal
import datetime
print(f"{datetime.datetime.now()} Initializing...")


load_dotenv()

# this allows categorical data from various sources to be combined and handled gracefully; performance cost is acceptable
pl.enable_string_cache()

# Define Data Model for Results

results_schema = {
    'commentId': pl.UInt16,
    'languageModelModerated': pl.Int8,
    'classification': pl.Categorical,
    'thoughts': pl.List(pl.String),
}


def cuda_print_memory():
    free_memory = sum([torch.cuda.mem_get_info(i)[0]
                      for i in range(torch.cuda.device_count())])
    total_memory = sum([torch.cuda.mem_get_info(i)[1]
                       for i in range(torch.cuda.device_count())])
    allocated_memory = sum([torch.cuda.memory_allocated(i)
                           for i in range(torch.cuda.device_count())])

    print(f"CUDA Memory: {round(free_memory/1024**3,1)} GB free, {round(allocated_memory/1024**3,1)} GB allocated, {round(total_memory/1024**3,1)} GB total")


def cuda_ensure_memory(required_memory_gb):
    required_memory = required_memory_gb * 1024**3
    free_memory = sum([torch.cuda.mem_get_info(i)[0]
                      for i in range(torch.cuda.device_count())])
    if free_memory < required_memory:
        print(
            f"Insufficient CUDA memory: {round(free_memory/1024**3,1)} GB free, {required_memory_gb} GB required")
        sys.exit(2)


# Grammar Specification for Text Generation

@guidance(stateless=True)
def generate_line(lm, name: str, temperature=0, max_tokens=50, list_append=False):
    return lm + gen(name=name, max_tokens=max_tokens, temperature=temperature, list_append=list_append, stop=['\n'])


@guidance(stateless=True)
def generate_phrase(lm, name: str, temperature=0, max_tokens=50, list_append=False):
    return lm + gen(name=name, max_tokens=max_tokens, temperature=temperature, list_append=list_append, stop=['\n', '.'])


@guidance(stateless=True)
def generate_number(lm, name: str, min: int, max: int, list_append=False):
    return lm + select(list(range(min, max+1)), name=name, list_append=list_append)

# Initialize Language Model


@guidance
def guidance_moderation(lm, instructions, classify_options, comments_df, results, examples_accepted=[], examples_rejected=[], reject_reasons=[], two_step_strategy=False, thought=False, decompose_comments=False, explain=False, temperature=0, progress_bar=None):

    if progress_bar is not None:
        lm.echo = False

    # output the task instructions
    with instruction():
        lm += instructions

    # output some examples to help the assistant understand the task
    i = 1
    for example in examples_accepted:
        with user():
            lm += f"COMMENT EXAMPLE {i}: {example}"
            i += 1

        with assistant():
            lm += classify_options[0]

    for example, reason, explanation in examples_rejected:
        with user():
            lm += f"COMMENT EXAMPLE {i}:\n{example}"
            i += 1

        with assistant():
            lm += f"""\
            {classify_options[2]}
            COMMENT EXAMPLE {i} CLASSIFICATION: {reason}
            EXPLANATION: {explanation}
            """

    # iterate through the comments and ask the assistant to moderate each one
    for commentId, commentText in comments_df.select('commentId', 'commentText').iter_rows():
        if two_step_strategy:
            lm + guidance_moderation_comment_two_step(
                commentId, commentText, classify_options, results, thought, explain, temperature, reject_reasons)
        else:
            lm + guidance_moderation_comment_one_step(
                commentId, commentText, classify_options, results, thought, decompose_comments, explain, temperature)

        progress_bar.update() if progress_bar is not None else None

    return lm


@guidance
def guidance_moderation_comment_two_step(lm, commentId, commentText, classify_options, results, thought, explain, temperature, reject_reasons=[]):
    with user():
        lm += f"COMMENT {commentId}: {commentText}"

    with assistant():
        lm += "CLASSIFICATION: " + \
            select(classify_options, name="classification") + "\n"
        classification = lm['classification']
        reason = None
        classificationCode = 0
        thoughts = []

        if classification == classify_options[1] and explain:
            lm += "EXPLANATION: " + \
                generate_line("classification_explanation", temperature) + "\n"
            thoughts += [f"EXPLANATION: {lm['classification_explanation']}"]

        # if comment is REJECTED and second thought is enabled, allow model to reconsider
        if classification == classify_options[2] and thought:
            lm += f"""\
            THOUGHT: This comment is {generate_phrase("classification_thought", temperature)}. I should classify it as {generate_phrase("classification_thought_2", temperature)}.
            """
            thoughts += [
                f"THOUGHT: This comment is {lm['classification_thought']}.",
                f"I should classify it as {lm['classification_thought_2']}.",
            ]

            lm += f"CLASSIFICATION: " + \
                select(reject_reasons, name="classification_reason") + "\n"
            reason = lm['classification_reason']

            lm += f"Based on the reasoning, should this comment still be {classify_options[2]}? " + select(
                ["YES", "NO"], name="classification_certainty") + "\n"

            thoughts += [
                f"CERTAIN: {lm['classification_certainty']}",
            ]

            if lm['classification_certainty'] == "NO":
                classification = classify_options[1]
                lm += classify_options[1]

        # if comment is still REJECTED after second thought, or if thought is disabled
        if classification == classify_options[2]:
            classificationCode = -1

            # this indicates that second thought was not enabled and we never got the classification of rejected comment
            if reason is None:
                lm += f"CLASSIFICATION: " + \
                    select(reject_reasons, name="classification_reason") + "\n"
                reason = lm['classification_reason']

            if explain:
                lm += "EXPLANATION: " + \
                    generate_line("classification_explanation",
                                  temperature) + "\n"
                thoughts += [f"EXPLANATION: {lm['classification_explanation']}"]

        # if comment is ACCEPTED
        if classification == classify_options[0]:
            classificationCode = 1

        if classification == classify_options[2]:
            classification = reason

        results.addRow({
            'commentId': commentId,
            'languageModelModerated': classificationCode,
            'classification': classification,
            'thoughts': [thoughts]
        })

    return lm


@guidance
def guidance_moderation_comment_one_step(lm, commentId, commentText, classify_options, results, thought, decompose_comments, explain, temperature):
    with user():
        lm += f"COMMENT {commentId}: {commentText}"

    with assistant():
        moderated = None
        thoughts = []

        if decompose_comments:
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
            if explain:
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


def create_experiments(comments, summary):
    experiments = []

    # Two-Step Moderation with Second-Thought Reasoning
    #
    # First, determine whether the comment would be rejected. Then deduce the reason for rejection. Following is the language model prompt:

    classify_options = ['ACCEPT', 'UNSURE', 'REJECT']
    reject_reasons = ['SPAM', 'COMPLEX']

    instructions_two_step = f"""\
    Discussion Title: {summary.get('topic')}
    Discussion Question: {summary.get('conversation-description')}

    ---
    You will be presented with comments posted on Polis discussion platform.
    Classify each comment objectively based on whether it meets the given guidelines.

    ---
    Classifications:
    - ACCEPT: Comment is coherent, makes a suggestion, or presents a real problem or issue.
    - REJECT: Comment should definitely be rejected for one of the reasons listed below.
    - UNSURE: may be accepted if it appears somewhat related to the discussion.

    ---
    Reasons for {classify_options[2]}:
    - SPAM: Does not either present a problem or discuss a solution. Cannot possibly be related to the discussion question.
    - COMPLEX: State more than one idea. Difficult to determine the where another person would agree or disagree.

    ---
    Output format:
    CLASSIFICATION: One of the following based on given guidelines: {", ".join(classify_options)}.
    THOUGHT: Deliberate about how the comment should be classified and why.
    Based on the reasoning, should this comment still be {classify_options[2]}? Answer with YES or NO.
    REASON: One of the following based on given guidelines: {", ".join(reject_reasons)}
    EXPLANATION: Provide an explanation for the classification.
    """

    examples_accepted = [
        "The impact of climate on our situation requires some serious thought and reform from our elected leaders.",
        "We need better traffic management.",
        "Every individual should have a voice and be able to make a difference.",
    ]

    examples_rejected = [
        (
            "lol why am i even here",
            "SPAM",
            "Incoherent statement that does not contribute to the discussion in any way."
        ),
        (
            "I agree with the previous comment",
            "SPAM",
            "Does not introduce a problem or propose a solution."
        ),
        (
            "We need better policy for public welfare and economic development. Also, we should consider the impact of civic activities on the environment. Toledo road needs a new bridge. Our water supply is under threat.",
            "COMPLEX",
            "Introduces multiple unrelated problems. It is difficult to determine where another person would agree or disagree."
        ),
    ]

    # Run 1: Baseline Technique

    args = {
        'instructions': instructions_two_step,
        'two_step_strategy': True,
        'classify_options': classify_options,
        'comments_df': comments.df.sort('timestamp'),
        'reject_reasons': reject_reasons,
    }

    experiments.append((1, args))

    # Run 2: Examples provided

    args = {
        'instructions': instructions_two_step,
        'two_step_strategy': True,
        'classify_options': classify_options,
        'comments_df': comments.df.sort('timestamp'),
        'examples_accepted': examples_accepted,
        'examples_rejected': examples_rejected,
        'reject_reasons': reject_reasons,
    }

    experiments.append((2, args))

    # Run 3: Thought statement enabled; no examples
    args = {
        'instructions': instructions_two_step,
        'two_step_strategy': True,
        'classify_options': classify_options,
        'comments_df': comments.df.sort('timestamp'),
        'reject_reasons': reject_reasons,
        'thought': True,
    }

    experiments.append((3, args))

    # Run 4: Thought statement enabled; examples provided

    args = {
        'instructions': instructions_two_step,
        'two_step_strategy': True,
        'classify_options': classify_options,
        'comments_df': comments.df.sort('timestamp'),
        'examples_accepted': examples_accepted,
        'examples_rejected': examples_rejected,
        'reject_reasons': reject_reasons,
        'thought': True,
    }

    experiments.append((4, args))

    # Run 5: Multiple Category Classification

    classify_options_multiple = [
        'ACCEPT', 'UNSURE', 'SPAM', 'IRRELEVANT', 'UNPROFESSIONAL', 'SCOPE', 'COMPLEX'
    ]

    instructions_one_step = f"""\
    Discussion Title: {summary.topic}
    Discussion Question: {summary.get('conversation-description')}

    ---
    Classify each comment objectively based on the following guidelines:
    - ACCEPT: mentions a real problem related to the discussion.
    - ACCEPT: recommends a realistic and actionable solution related to the discussion.
    - ACCEPT: makes a sincere suggestion related to the discussion.
    - IRRELEVANT: frivolous, irrelevant, unrelated to the discussion.
    - IRRELEVANT: does not contribute to the discussion in a meaningful way.
    - SPAM: incoherent or lacks seriousness.
    - SPAM: provides neither a problem nor a solution.
    - UNPROFESSIONAL: the language is informal, colloquial, disrespectful or distasteful.
    - SCOPE: cannot be addressed within the scope of original question.
    - COMPLEX: introduces multiple ideas, even if they are related to the discussion.
    - COMPLEX: discusses distinct problems, making it difficult to determine where another person would agree or disagree.
    - UNSURE: may be accepted if it appears somewhat related to the discussion.

    ---
    Output format:
    CLASSIFICATION: One of the following based on given guidelines: {", ".join(classify_options_multiple)}.
    EXPLANATION: Provide an explanation for the classification.
    """

    args = {
        'instructions': instructions_one_step,
        'classify_options': classify_options_multiple,
        'comments_df': comments.df.sort('timestamp'),
        'thought': False,
        'decompose_comments': False,
    }

    experiments.append((5, args))

    # Run 6: Multiple Category Classification with Thought Statement

    args = {
        'instructions': instructions_one_step,
        'classify_options': classify_options_multiple,
        'comments_df': comments.df.sort('timestamp'),
        'thought': True,
        'decompose_comments': False,
    }

    experiments.append((6, args))

    # Decompose Comments
    #
    # Decomposing each comment into its requisite components might further help improve the analysis and evaluation abilities of the model. While we do not care about the accuracy or agreeability of the posted comments, we do expect either a problem or a potential solution. Also, we want the comment to present a single idea.
    #
    # In the following prompt, we ask the model to idenfity the problem the comments address and their proposed solution. If neither is present, the comment is considered spam.

    # Run 7: Multiple Category Classification with Comment Decomposition and Thought Statement

    instructions_decompose_multi_label = f"""\
    Discussion Title: {summary.topic}
    Discussion Question: {summary.get('conversation-description')}

    ---
    Classify each comment objectively based on the following guidelines:
    - ACCEPT: mentions a real problem related to the discussion.
    - ACCEPT: recommends a realistic and actionable solution related to the discussion.
    - ACCEPT: makes a sincere suggestion related to the discussion.
    - IRRELEVANT: frivolous, irrelevant, unrelated to the discussion.
    - IRRELEVANT: does not contribute to the discussion in a meaningful way.
    - SPAM: incoherent or lacks seriousness.
    - SPAM: provides neither a problem nor a solution.
    - UNPROFESSIONAL: the language is informal, colloquial, disrespectful or distasteful.
    - SCOPE: cannot be addressed within the scope of original question.
    - COMPLEX: introduces multiple ideas, even if they are related to the discussion.
    - COMPLEX: discusses distinct problems, making it difficult to determine where another person would agree or disagree.
    - UNSURE: may be accepted if it appears somewhat related to the discussion.

    ---
    Output format:
    PROBLEM: The specific problem mentioned in the comment. If only an action is suggested and no problem is explicitly mentioned, state None.
    ACTION: What suggestion or change is proposed. If only a problem is mentioned and no action is suggested, state None.
    HOW MANY IDEAS: Number of distinct ideas introduced in the comment.
    THOUGHT: Deliberate about how the comment should be classified.
    CLASSIFICATION: One of the following based on given guidelines: {", ".join(classify_options_multiple)}.
    EXPLANATION: Provide an explanation for the classification.
    """

    args = {
        'instructions': instructions_decompose_multi_label,
        'classify_options': classify_options_multiple,
        'comments_df': comments.df.sort('timestamp'),
        'thought': True,
        'decompose_comments': True,
    }

    experiments.append((7, args))

    args = {
        'instructions': instructions_decompose_multi_label,
        'classify_options': classify_options_multiple,
        'comments_df': comments.df.sort('timestamp'),
        'thought': False,
        'decompose_comments': True,
    }

    experiments.append((8, args))

    # Run 9: Simple Classification with Comment Decomposition and Thought Statement

    instructions_decompose = f"""\
    Discussion Title: {summary.topic}
    Discussion Question: {summary.get('conversation-description')}

    ---
    Classify each comment objectively based on the following guidelines.

    Mark the comment as:
    - ACCEPT: mentions a real problem related to the discussion.
    - ACCEPT: recommends a realistic and actionable solution related to the discussion.
    - ACCEPT: makes a sincere suggestion related to the discussion.
    - REJECT: frivolous, irrelevant, unrelated to the discussion.
    - REJECT: does not contribute to the discussion in a meaningful way.
    - REJECT: incoherent or lacks seriousness.
    - REJECT: provides neither a problem nor a solution.
    - REJECT: the language is informal, colloquial, disrespectful or distasteful.
    - REJECT: cannot be addressed within the scope of original question.
    - REJECT: introduces multiple ideas, even if they are related to the discussion.
    - REJECT: discusses distinct problems, making it difficult to determine where another person would agree or disagree.
    - UNSURE: may be accepted if it appears somewhat related to the discussion.

    ---
    Output format:
    PROBLEM: The specific problem mentioned in the comment. If only an action is suggested and no problem is explicitly mentioned, state None.
    ACTION: What suggestion or change is proposed. If only a problem is mentioned and no action is suggested, state None.
    HOW MANY IDEAS: Number of distinct ideas introduced in the comment.
    THOUGHT: Deliberate about how the comment should be classified.
    CLASSIFICATION: One of the following based on given guidelines: {", ".join(classify_options)}.
    EXPLANATION: Provide an explanation for the classification.
    """

    args = {
        'instructions': instructions_decompose,
        'classify_options': classify_options,
        'comments_df': comments.df.sort('timestamp'),
        'thought': True,
        'decompose_comments': True,
    }

    experiments.append((9, args))

    return experiments


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
        dataset, f'moderation-results-{experiment}', DATA_PATH, schema=results_schema).initialize()

    args['results'] = moderationResults
    args['progress_bar'] = progress_bar

    languageModel + guidance_moderation(**args)

    moderationResults.save()
    progress_bar.close()

    print(f"{datetime.datetime.now()} Experiment {experiment} Complete.")


# for experiment, args in experiments:
#     run_experiment(experiment, args)


if __name__ == "__main__":

    # trap SIGINT and SIGTERM to ensure graceful exit
    def signal_handler(sig, frame):
        print(f"{datetime.datetime.now()} Exiting...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    OPENDATA_REPO_PATH = os.getenv("OPENDATA_REPO_PATH")
    DATA_PATH = os.getenv("DATA_PATH")

    # Verify GPU Availability

    if not torch.cuda.is_available():
        print("No CUDA device found")
        sys.exit(2)

    print(f"""
    Device: {torch.cuda.get_device_name(0)}
    Python: {sys.version}
    PyTorch: {torch.__version__}
    CUDA: {torch.version.cuda}
    CUDNN: {torch.backends.cudnn.version()}
    """)

    cuda_print_memory()

    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()

    CUDA_MINIMUM_MEMORY_GB = os.getenv("CUDA_MINIMUM_MEMORY_GB")
    MODEL_ID = os.getenv("MODEL_ID")

    if MODEL_ID is None:
        print("MODEL_ID environment variable is required.")
        sys.exit(3)

    if CUDA_MINIMUM_MEMORY_GB is not None:
        cuda_ensure_memory(int(CUDA_MINIMUM_MEMORY_GB))

    print(f"{datetime.datetime.now()} Initializing language model: {MODEL_ID}...")
    languageModel = models.TransformersChat(MODEL_ID, device_map="auto")

    print(f"{datetime.datetime.now()} Language model initialized.")
    cuda_print_memory()

    # specify default dataset
    datasets = ["american-assembly.bowling-green"]

    # allow user to specify datasets at runtime
    if "--datasets" in sys.argv:
        datasets = sys.argv[sys.argv.index("--datasets")+1].split(",")

    # loop through datasets and experiments
    for dataset in datasets:
        print(f"{datetime.datetime.now()} Loading dataset: {dataset}...")

        comments = Comments(dataset, dataPath=DATA_PATH).load_from_csv(f'{OPENDATA_REPO_PATH}/{dataset}/comments.csv')
        summary = Summary(OPENDATA_REPO_PATH, dataset)

        print(f"""
Dataset: {dataset}
Topic: {summary.topic}
Comments: {comments.df.height}
""")

        # pprint(comments.df)

        experiments = create_experiments(comments, summary)

        start = 1
        end = len(experiments)

        if "--start" in sys.argv:
            start = int(sys.argv[sys.argv.index("--start")+1])

        if "--end" in sys.argv:
            end = int(sys.argv[sys.argv.index("--end")+1])

        for experiment, args in experiments[start-1:end]:
            run_experiment(dataset, languageModel, experiment, args)

        print()

    print(f"{datetime.datetime.now()} All Experiments Complete.")
