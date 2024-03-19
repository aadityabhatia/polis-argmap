from .experiment import Experiment
from argmap.guidance import generate_phrase
from argmap.dataModel import Summary, Comments, Topics, HierarchicalTopics
import datetime
import logging
import re
import os
import math
import numpy as np
from tqdm import tqdm
import polars as pl
from pprint import pprint

import guidance
from guidance import user, assistant, instruction

from sklearn.metrics import silhouette_score


class TopicModeling(Experiment):

    @staticmethod
    def run(dataset, languageModel=None):

        try:
            topicModelTask = TopicModeling(dataset)
        except FileNotFoundError as e:
            print(f"{datetime.datetime.now()} Error: {e}")
            return

        print(f"{datetime.datetime.now()} Importing dependencies... ", end="", flush=True)
        import_heavy_libraries()
        print("done.", flush=True)

        topic_model = topicModelTask.createTopicModel()
        topic_assignments = topicModelTask.fit_transform()

        print(f"""\
Comments: {len(topicModelTask.documents)}
Topics: {len(topic_model.get_topic_freq()) - 1}
Outliers: {topic_assignments.count(-1)}
Largest Cluster: {topic_assignments.count(0)}
Silhouette Score: {silhouette_score(topicModelTask.embeddings, topic_assignments, metric='cosine')}
Relative Validity: {topic_model.hdbscan_model.relative_validity_}\
""")

        topic_assignments = topicModelTask.assignOutliers(topic_assignments)

        print("Creating hierarchical topics...")
        topicModelTask.createHierarchicalTopics()

        try:
            print("Generating topic headings...")
            from argmap.helpers import loadLanguageModel
            languageModel = loadLanguageModel()
            topicModelTask.generateTopicHeadings(languageModel)
        except Exception as e:
            logging.error(f"Error processing dataset: {dataset}")
            logging.error(e)
            import traceback
            traceback.print_exc()

        print("Saving comments...")
        topicModelTask.saveComments(topic_assignments)

    def __init__(self, dataset) -> None:
        EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID")
        if EMBED_MODEL_ID is None:
            raise Exception("Required: EMBED_MODEL_ID environment variable")

        self.dataset = dataset

        self.comments = Comments(dataset).load_from_parquet()
        self.summary = Summary(dataset)

        self.comments_df = (
            self.comments.df
            .filter(pl.col('moderated') >= 0)
            .select('commentId', 'commentText', embedding=f'embedding-{EMBED_MODEL_ID}')
        )

        self.documents = self.comments_df.get_column('commentText').to_list()

        # Convert list of numpy arrays to 2D numpy array
        embeddings = self.comments_df.get_column('embedding').to_numpy()
        self.embeddings = np.array([np.array(embedding)
                                    for embedding in embeddings])

    def createTopicModel(self):
        from bertopic import BERTopic
        from bertopic.representation import MaximalMarginalRelevance, PartOfSpeech
        from bertopic.vectorizers import ClassTfidfTransformer
        from sklearn.feature_extraction.text import CountVectorizer

        import spacy
        from spacy.lang.en.stop_words import STOP_WORDS
        import numpy as np

        from umap import UMAP
        from hdbscan import HDBSCAN

        import torch
        import re

        umap_params = dict(
            n_neighbors=math.floor(math.log(self.comments_df.height)),
            min_dist=0.0,
            n_components=32,
            metric='cosine',
            random_state=42,
            densmap=True,
        )
        pprint(f"UMAP Parameters: {umap_params}")
        self.umap_model = UMAP(**umap_params)

        hdbscan_params = dict(
            # 2% of the statement count or 5, whichever is higher
            min_cluster_size=max(math.floor(self.comments_df.height / 50), 4),
            # min_cluster_size=math.ceil(math.log(self.comments_df.height)),  # natural log of the statement count
            min_samples=1,  # a higher default value makes clustering more conservative
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True,
            gen_min_span_tree=True,
        )
        pprint(f"HDBSCAN Parameters: {hdbscan_params}")
        self.hdbscan_model = HDBSCAN(**hdbscan_params)

        summary_stop_words = re.split(r'\W+', self.summary.topic.lower())

        self.vectorizer_model = CountVectorizer(stop_words=(
            list(STOP_WORDS) + summary_stop_words), ngram_range=(1, 2))

        self.ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

        pos_patterns = [
            [{'POS': 'ADJ'}, {'POS': 'NOUN'}],
            [{'POS': 'NOUN'}]
        ]

        self.representation_model = [
            PartOfSpeech("en_core_web_lg", pos_patterns=pos_patterns),
            MaximalMarginalRelevance(diversity=0.3),
        ]

        self.topic_model = BERTopic(
            umap_model=self.umap_model,					 # Reduce dimensionality
            hdbscan_model=self.hdbscan_model,				# Cluster reduced embeddings
            vectorizer_model=self.vectorizer_model,		 # Tokenize topics
            ctfidf_model=self.ctfidf_model,				 # Extract topic words
            representation_model=self.representation_model,  # Fine-tune topic representations
            nr_topics="auto",
        )

        return self.topic_model

    def fit_transform(self):
        topic_assignments, _ = self.topic_model.fit_transform(
            self.documents, self.embeddings)
        return topic_assignments

    def generateTopicHeadings(self, languageModel):
        topics = Topics(self.dataset, df=pl.from_pandas(
            self.topic_model.get_topic_info()))

        progress_bar = tqdm(
            total=topics.df.height,
            desc='Topic Titles',
            unit="topics",
            smoothing=0.1,
        )

        languageModel + generate_topic_headings_titles(
            self.summary,
            topics,
            generate_titles=True,
            progress_bar=progress_bar
        )

        topics.save_to_parquet()
        progress_bar.close()

        self.topic_model.set_topic_labels(
            topics.df.get_column('Heading').to_list()
        )

    def assignOutliers(self, topic_assignments):
        if topic_assignments.count(-1) > 0:
            print(
                f"Assigning {topic_assignments.count(-1)} outliers to topics using embeddings...")
            topic_assignments = self.topic_model.reduce_outliers(
                self.documents,
                topic_assignments,
                strategy='embeddings',
                embeddings=self.embeddings
            )

        if topic_assignments.count(-1) > 0:
            print(
                f"Assigning {topic_assignments.count(-1)} outliers to topics using c-TF-IDF based probability distributions...")
            topic_assignments = self.topic_model.reduce_outliers(
                self.documents,
                topic_assignments,
                strategy='distributions'
            )

        if topic_assignments.count(-1) > 0:
            print("Outliers remaining:", topic_assignments.count(-1))

        self.topic_model.update_topics(
            self.documents,
            topics=topic_assignments,
            ctfidf_model=self.ctfidf_model,
            vectorizer_model=self.vectorizer_model,
            representation_model=self.representation_model
        )

        return topic_assignments

    def saveComments(self, topic_assignments):
        self.comments_df = self.comments_df.with_columns(
            topicId=pl.Series(topic_assignments).cast(pl.Int16))
        self.comments.join_in_place(
            self.comments_df.select('commentId', 'topicId'),
            'commentId',
            dropColumns='topicId'
        ).save_to_parquet()

    def createHierarchicalTopics(self):
        hierarchical_topics = self.topic_model.hierarchical_topics(
            self.documents)
        hTopics = HierarchicalTopics(self.dataset, df=hierarchical_topics)
        hTopics.save_to_parquet()


def import_heavy_libraries():
    from bertopic import BERTopic
    import spacy
    from umap import UMAP
    from hdbscan import HDBSCAN
    import torch

    # prefer GPU for spacy if available
    if torch.cuda.is_available():
        spacy.prefer_gpu()


@guidance
def generate_topic_headings_titles(lm, summary, topics, generate_titles=False, progress_bar=None):

    temperature = int(os.getenv('MODEL_TEMPERATURE', 0))

    if progress_bar is not None:
        lm.echo = False

    # avoid repeating anything from conversation title
    taboo_words = re.split(r'\W+', summary.topic)

    with instruction():
        lm += f"""\
Assign a detailed title and a short heading to best represent each given topic.
Start with a noun or adjective.
Avoid repetitive words or phrases such as "Enhancing" or "Improving".
Avoid using these words: {', '.join(taboo_words)}

KEYWORDS: [a set of keywords that describe the topic]
STATEMENTS: [a set of statements that best represent the topic]
TITLE: [a descriptive sentence that represents the topic and starts with a noun]
HEADING: [terse phrase]
"""

    topic_titles = []
    topic_headings = []

    for topic, keywords, docs in topics.df.select('Topic', 'Representation', 'Representative_Docs').iter_rows():
        if topic == -1:
            topic_titles.append("Outliers")
            topic_headings.append("Outliers")
            progress_bar.update() if progress_bar is not None else None
            continue

        with user():
            lm_topic = lm + f"""
            # Topic {topic}
            KEYWORDS: {', '.join(keywords)}
            STATEMENTS: {'; '.join(docs)}
            """
        with assistant():
            if generate_titles:
                lm_topic += f"TITLE: " + \
                    generate_phrase('title', temperature, 50) + '\n'
            lm_topic += f"HEADING: " + \
                generate_phrase('heading', temperature, 12) + '\n'

        if generate_titles:
            topic_titles.append(lm_topic['title'])

        topic_headings.append(lm_topic['heading'])

        progress_bar.update() if progress_bar is not None else None

    if generate_titles:
        topics.addColumns(pl.Series('Title', topic_titles))

    topics.addColumns(pl.Series('Heading', topic_headings))

    return lm
