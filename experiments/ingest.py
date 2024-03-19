import datetime
import os

from .experiment import Experiment
from dotenv import load_dotenv
from argmap.dataModel import Summary, Comments
from argmap.helpers import loadEmbeddingModel

import polars as pl

class Ingestion(Experiment):

    @staticmethod
    def run(dataset):
        summary = Summary(dataset)
        comments = Comments(dataset)

        embedModel = loadEmbeddingModel()

        EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID")

        if os.path.exists(comments.filename):
            comments.load_from_parquet()
            print(
                f"Loaded {comments.df.height} comments from Parquet DataFrame.")
        else:
            comments.load_from_csv()
            print(
                f"Loaded {comments.df.height} comments from original dataset CSV.")

        print(f"Dataset: {dataset}")
        print(f"Topic: {summary.get('topic')}")
        print(f"Comments: {comments.df.height}")

        embeddings = calculate_embeddings(
            embedModel,
            comments,
            show_progress_bar=True
        )
        comments.addColumns(
            pl.Series(embeddings).alias(f'embedding-{EMBED_MODEL_ID}')
        )
        comments.save_to_parquet()

        print(f"Embeddings: {len(embeddings)}")
        print(f"Dimensions: {len(embeddings[0])}")
        print()


def calculate_embeddings(embedModel, comments, show_progress_bar=False):
    documents = comments.df.get_column('commentText').to_list()
    embeddings = embedModel.encode(
        documents,
        show_progress_bar=show_progress_bar
    )
    return embeddings
