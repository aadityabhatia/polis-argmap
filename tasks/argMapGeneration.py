from .task import Task
import datetime
import os
from argmap.argdown import argdown_heading, argdown_topic, argdown_argument, argdown_comment, argdown_template, argdown_supported_by
from argmap.dataModel import DataModel, Summary, Comments, Topics, Arguments, ArgumentCommentMap
import polars as pl


class ArgumentMapGeneration(Task):

    @staticmethod
    def run(dataset):
        try:
            summary = Summary(dataset)
            comments = Comments(dataset).load_from_parquet()
            topics = Topics(dataset).load_from_parquet()
            arguments = Arguments(dataset).load_from_parquet()
            argumentCommentMap = ArgumentCommentMap(
                dataset, 'argumentCommentMap').load_from_parquet()
            argumentTopicSupport = DataModel(
                dataset, 'argumentTopicSupport').load_from_parquet()
        except FileNotFoundError as e:
            print(f"{datetime.datetime.now()} Error: {e}")
            return

        print(f"{datetime.datetime.now()} Generating Argument Map for top arguments...")
        output = mapTopArgumentsComments(
            summary, comments, topics, argumentCommentMap, argumentTopicSupport)
        dataPath = os.getenv("DATA_PATH")
        path = os.path.join(dataPath, dataset, 'arguments-top.argdown')
        with open(path, 'w') as f:
            f.write(argdown_template(output))
        print(f"{datetime.datetime.now()} Saved to {path}")

        print(f"{datetime.datetime.now()} Generating Argument Map for all arguments...")
        output = mapAllArguments(
            summary, comments, topics, arguments, argumentCommentMap, argumentTopicSupport)
        path = os.path.join(dataPath, dataset, 'arguments-all.argdown')
        with open(path, 'w') as f:
            f.write(argdown_template(output))
        print(f"{datetime.datetime.now()} Saved to {path}")


def mapTopArgumentsComments(summary, comments, topics, argumentCommentMap, argumentTopicSupport, TOP_N_ARGUMENTS=3, TOP_N_COMMENTS=3):

    # shorthand
    summary.description = summary.get('conversation-description')

    # output discussion title and question
    output = argdown_heading(summary.topic, summary.description)
    # output += argdown_comment(summary.topic, summary.description)

    # argumentTopicSupport is a view that joins arguments and topics, and adds support and agreeability scores

    # filter to only the top argument for each topic
    topArguments = (
        argumentTopicSupport.df.lazy()
        .sort('agreeability', descending=True)
        .group_by('topicId', maintain_order=True).head(TOP_N_ARGUMENTS)
    ).collect()

    # output each topic once
    # get unique topicId from argumentTopicSupport

    topicIds = argumentTopicSupport.df.get_column(
        'topicId').unique().sort().to_list()

    for topicId in topicIds:
        topic = topics.get(topicId)
        heading = topic['Heading']
        title = topic['Title']
        output += argdown_topic(heading, title)

        # output += argdown_supports()

    # link topic to discussion title?

    # output top-n arguments and link to topic

    for topicId, topicHeading, argumentId, argumentTitle, argumentContent, agreeability, agrees, disagrees in (
        topArguments
        .select('topicId', 'topicHeading', 'argumentId', 'argumentTitle', 'argumentContent', 'agreeability', 'agrees', 'disagrees')
            .iter_rows()):

        body = f"{argumentContent} _({agrees + disagrees} voters approximated, {agreeability*100:.1f}% tend to agree_)"
        output += argdown_argument(argumentTitle, body, topicHeading)

    # find the top `n` comments for each argument

    argumentCommentMapSupport = argumentCommentMap.df.filter(
        relationship='SUPPORT')
    topArgumentComments = (
        topArguments
        .select('topicId', 'argumentId', 'argumentTitle', 'thoughts')
        .join(argumentCommentMapSupport, on=['argumentId', 'topicId'], how='left')
        .join(comments.df.select('commentId', 'commentText', 'agrees', 'disagrees', agreeability=(pl.col('agrees')/(pl.col('agrees')+pl.col('disagrees')))), on='commentId', how='left')
        .sort('agrees', descending=True)
        .group_by('topicId', 'argumentId', maintain_order=True)
        .head(TOP_N_COMMENTS)
    )

    for topicId, argumentId, argumentTitle, thoughts, commentId, commentText, agrees, disagrees, agreeability in (
        topArgumentComments
        .select('topicId', 'argumentId', 'argumentTitle', 'thoughts', 'commentId', 'commentText', 'agrees', 'disagrees', 'agreeability')
            .iter_rows()):

        body = f"{commentText} _({agrees + disagrees} votes, {agreeability*100:.1f}% agree_)"
        output += argdown_comment(f'Comment {commentId}',
                                  body, supportsArgument=argumentTitle)

    return output


def mapAllArguments(summary, comments, topics, arguments, argumentCommentMap, argumentTopicSupport):
    # output discussion title and question
    output = argdown_heading(summary.topic, summary.description)

    # argumentTopicSupport is a view that joins arguments and topics, and adds support and agreeability scores

    # get unique topicId from argumentTopicSupport
    # output each topic once
    for topicId in argumentTopicSupport.df.get_column('topicId').unique().sort().to_list():
        topic = topics.get(topicId)
        heading = topic['Heading']
        title = topic['Title']
        output += argdown_topic(heading, title)

    # # topic hierarchy
    # for Parent_ID, Parent_Name, topicList, Child_Left_ID, Child_Left_Name, Child_Right_ID, Child_Right_Name, Distance in hierarchicalTopics.df.iter_rows():
    #     output += "\n"
    #     output += f"[Topic {Parent_ID}]: Node #AI \n"
    #     output += f"  + [Topic {Child_Left_ID}]\n"
    #     output += f"  + [Topic {Child_Right_ID}]\n"

    # output all arguments and link to topic
    for topicId, topicHeading, argumentId, argumentTitle, argumentContent, agreeability, agrees, disagrees in (
        argumentTopicSupport.df
        .select('topicId', 'topicHeading', 'argumentId', 'argumentTitle', 'argumentContent', 'agreeability', 'agrees', 'disagrees')
            .iter_rows()):

        body = f"{argumentContent} ({agrees + disagrees} voters approximated, {agreeability*100:.1f}% tend to agree)"
        output += argdown_argument(argumentTitle, body, topicHeading)

    argumentCommentMapSupport = argumentCommentMap.df.filter(
        relationship='SUPPORT')

    # output all unique comments present in the argumentCommentMap
    for commentId, commentText, agrees, disagrees, agreeability in (
        argumentCommentMapSupport
        .select('commentId').unique()
        .join(comments.df, on='commentId', how='left')
        .select('commentId', 'commentText', 'agrees', 'disagrees', 'agreeability')
        .iter_rows()
    ):
        body = f"{commentText} ({agrees + disagrees} votes, {agreeability*100:.1f}% agree)"
        output += argdown_comment(f'Comment {commentId}', body)

    # enumerate argument-comment relationships
    for argumentId, topicId, argumentTitle, commentIds in (
        argumentCommentMapSupport.lazy()
        # add argumentTitle
        .join(arguments.df.lazy(), on=['topicId', 'argumentId'], how='left')
        .select('argumentId', 'topicId', 'argumentTitle', 'commentId')
        # aggregate commentIds
        .group_by(['argumentId', 'topicId', 'argumentTitle']).all()
    ).collect().iter_rows():
        output += f"\n<{argumentTitle}>\n"
        for commentId in commentIds:
            output += argdown_supported_by(f'Comment {commentId}')

    return output
