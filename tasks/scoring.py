from .task import Task
import datetime
from argmap.dataModel import DataModel, Topics, Arguments, ArgumentCommentMap, Votes

import polars as pl


class Scoring(Task):

    @staticmethod
    def run(dataset):

        try:
            topics = Topics(dataset).load_from_parquet()
            arguments = Arguments(dataset).load_from_parquet()
            argumentCommentMap = ArgumentCommentMap(
                dataset, 'argumentCommentMap').load_from_parquet()
            votes = Votes(dataset).load_from_csv()
        except FileNotFoundError as e:
            print(f"{datetime.datetime.now()} Error: {e}")
            return

        argumentTopicSupport = score(
            topics,
            arguments,
            argumentCommentMap,
            votes
        )

        DataModel(
            dataset,
            'argumentTopicSupport',
            df=argumentTopicSupport
        ).save()


def score(topics, arguments, argumentCommentMap, votes):

    argumentSupportByComment = (
        argumentCommentMap.df.lazy()
        .filter(pl.col('relationship') == 'SUPPORT')
        .join(votes.df.lazy().filter(pl.col('vote') != 0), 'commentId', 'inner')
        .select('topicId', 'argumentId', 'commentId', 'voterId', 'vote')
    )

    argumentSupportByVoter = (
        argumentSupportByComment
        .group_by(['topicId', 'argumentId', 'voterId'])
        .agg(vote=pl.sum('vote'))
        .with_columns(
            vote=(
                pl.when(pl.col('vote') > 0).then(1)
                .when(pl.col('vote') < 0).then(-1)
                .otherwise(0)
            ),
            agree=pl.col('vote') > 0,
            disagree=pl.col('vote') < 0,
        )
    )

    argumentSupport = (
        argumentSupportByVoter
        .group_by('topicId', 'argumentId')
        .agg(
            support=pl.sum('vote'),
            agrees=pl.sum('agree'),
            disagrees=pl.sum('disagree'),
        )
        .with_columns(
            agreeability=pl.col('agrees') /
            (pl.col('agrees') + pl.col('disagrees')),
        )
        .sort('agreeability', descending=True)
    )

    argumentTopicSupport = (
        argumentSupport
        .join(arguments.df.lazy(), ['topicId', 'argumentId'], 'left')
        .join(topics.df.lazy().select(topicId=pl.col('Topic'), topicTitle='Title', topicHeading='Heading'), 'topicId', 'left')
        .sort('agreeability', descending=True)
    ).collect()

    return argumentTopicSupport
