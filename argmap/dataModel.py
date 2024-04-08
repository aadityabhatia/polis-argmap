import polars as pl
import os

# this allows categorical data from various sources to be combined and handled gracefully; performance cost is acceptable
pl.enable_string_cache()


class Summary:
    def __init__(self, dataset):
        self.dataset = dataset
        import csv
        openDataRepoPath = os.getenv("OPENDATA_REPO_PATH")
        csv_path = os.path.join(openDataRepoPath, dataset, 'summary.csv')
        with open(csv_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            self.summary = {row[0]: row[1] for row in reader}
            for key, value in self.summary.items():
                setattr(self, key, value)

    def get(self, key):
        return self.summary[key]

    def getAll(self):
        return self.summary


class DataModel:
    df = None
    dbTable = None
    filename = None
    dbURI = None
    schema = None
    dataset = None
    table = None

    def __init__(self, dataset, table=None, dataPath=None, schema=None, df=None):

        dataPath = dataPath or os.getenv("DATA_PATH")
        self.dataset = dataset
        self.table = table = self.table or table or self.__class__.__name__.lower()
        self.dbTable = f"{dataset}-{table}".replace('.', '-')
        self.dbURI = os.getenv("DB_CONNECTION_URI")

        if schema is not None:
            self.schema = schema

        if df is not None:
            if not isinstance(df, pl.DataFrame):
                df = pl.from_pandas(df)
            self.df = df

            if self.schema is None:
                self.schema = df.schema
            else:
                self.df = self.df.cast(self.schema)

        if dataPath is not None:
            self.filename = f'{dataPath}/{dataset}/{table}.parquet'
            os.makedirs(f'{dataPath}/{dataset}', exist_ok=True)

    def initialize(self):
        if self.df is None and self.schema is not None:
            self.df = pl.DataFrame(schema=self.schema)
        return self

    def preprocess(self, lazyFrame: pl.LazyFrame) -> pl.LazyFrame:
        """Override this method to preprocess the dataframe after loading from CSV"""
        return lazyFrame

    def save(self):
        if self.dbURI is not None:
            self.save_to_database()
        if self.filename is not None:
            self.df.write_parquet(self.filename)
        return self

    def load_from_csv(self, **kwargs):
        openDataRepoPath = os.getenv("OPENDATA_REPO_PATH")
        csv_path = os.path.join(
            openDataRepoPath, self.dataset, f'{self.table}.csv')

        lazyFrame = pl.scan_csv(csv_path)

        lazyFrame = self.preprocess(lazyFrame)

        if self.schema is not None:
            lazyFrame = lazyFrame.cast(self.schema)
        else:
            self.schema = lazyFrame.schema

        self.df = lazyFrame.collect()
        return self

    def load_from_database(self):
        self.df = pl.read_database_uri(
            query=f'SELECT * from "{self.dbTable}"',
            uri=self.dbURI,
            schema_overrides=self.schema,
            engine='adbc',
        )
        return self

    def save_to_database(self, **kwargs):
        self.df.write_database(
            self.dbTable, self.dbURI, if_table_exists='replace', engine='adbc', **kwargs)
        return self

    def load_from_parquet(self, **kwargs):
        self.df = pl.read_parquet(self.filename, **kwargs)
        if self.schema is None:
            self.schema = self.df.schema
        else:
            self.df = self.df.cast(self.schema)
        return self

    def save_to_parquet(self, **kwargs):
        self.df.write_parquet(self.filename, **kwargs)
        return self

    def addRow(self, row_dict):
        """Add a row to the dataframe. The row_dict must have the same keys as the schema."""
        df_new = pl.DataFrame(row_dict, schema=self.schema)
        self.df.vstack(df_new, in_place=True)

    def addColumns(self, *args, **kwargs):
        """Add a column to the dataframe"""
        self.df = self.df.with_columns(*args, **kwargs)
        self.schema = self.df.schema

    def join(self, df_new, on, how='left'):
        """Shorthand to left join with another dataframe on specified column"""
        return self.df.join(df_new, on=on, how=how)

    def join_in_place(self, df_new, on, dropColumns=None):
        """Shorthand to left join with another dataframe on specified column"""
        if dropColumns is not None:
            self.df = self.df.drop(dropColumns)
        self.df = self.join(df_new, on=on, how='left')
        return self

    def get(self, *args, **kwargs):
        raise Exception("Method not implemented")

    def glimpse(self, **kwargs):
        """shorthand for df.glimpse()"""
        return self.df.glimpse(**kwargs)


class Arguments(DataModel):
    schema = {
        'topicId': pl.Int16,
        'argumentId': pl.UInt16,
        'argumentTitle': pl.String,
        'argumentContent': pl.String,
        'thoughts': pl.List(pl.String),
    }

    def get(self, topicId):
        return self.df.filter(pl.col('topicId') == topicId)

    def stack(self, topicId, argumentTitles, argumentContents):
        countTitles = len(argumentTitles)
        countContents = len(argumentContents)
        if countTitles != countContents:
            raise Exception(
                f"Number of argument titles ({countTitles}) does not match number of argument contents ({countContents})")
        df_new = pl.DataFrame({
            "topicId": topicId,
            "argumentId": range(1, countTitles + 1),
            "argumentTitle": argumentTitles,
            "argumentContent": argumentContents,
        }, schema=self.schema)
        self.df = self.df.filter(pl.col('topicId') != topicId).vstack(df_new)
        return self


class Comments(DataModel):

    schema = {
        'timestamp': pl.Int64,
        'commentId': pl.UInt16,
        'authorId': pl.UInt16,
        'agrees': pl.UInt16,
        'disagrees': pl.UInt16,
        'moderated': pl.Int8,
        'commentText': pl.String,
        'agreeability': pl.Float32,
    }

    def get(self, topicId, quantile=None, agreeabilityThreshold=None):
        """Get agreeable comments from a specific topic."""
        predicate = (pl.col('topic') == topicId)
        if agreeabilityThreshold is not None:
            predicate = predicate & (
                pl.col('agreeability') >= agreeabilityThreshold)
        if quantile is not None and agreeabilityThreshold is None:
            agreeabilityThreshold = (
                self.df
                .filter(predicate)
                .get_column('agreeability')
                .quantile(quantile))
            predicate = predicate & (
                pl.col('agreeability') >= agreeabilityThreshold)
        return self.df.filter(predicate)

    def getAgreeableComments(self, agreeabilityThreshold: float = None, quantile: float = None):
        """Get comments that have been accepted during moderation and are above specified agreeability threshold.
        If quantile is specified, agreeabilityThreshold is ignored and calculated based on agreeability distribution.
        """

        if agreeabilityThreshold is None and quantile is None:
            quantile = 0.1

        if quantile is not None:
            agreeabilityThreshold = (
                self.df
                .filter(moderated=1)
                .get_column('agreeability')
                .quantile(quantile))

        return self.df.filter(
            (pl.col('agreeability') >= agreeabilityThreshold) &
            (pl.col('moderated') == 1)
        )

    # agreeability is the proportion of votes that agrees
    def preprocess(self, lazyFrame: pl.LazyFrame) -> pl.LazyFrame:
        return (
            lazyFrame
            .rename({'comment-body': 'commentText', 'author-id': 'authorId', 'comment-id': 'commentId'})
            .drop('datetime')
            .with_columns(
                agreeability=(
                    pl.when(pl.col('agrees') +
                            pl.col('disagrees') == 0).then(0)
                    .otherwise(
                        pl.col('agrees') / (pl.col('agrees') + pl.col('disagrees'))))
            )
        )


class Votes(DataModel):
    schema = {
        'commentId': pl.UInt16,
        'voterId': pl.UInt16,
        'vote': pl.Int8,
    }

    def preprocess(self, lazyFrame: pl.LazyFrame) -> pl.LazyFrame:
        return (
            lazyFrame
            .rename({'comment-id': 'commentId', 'voter-id': 'voterId'})
            .filter(pl.col('vote') != 0)
            .sort('timestamp')
            .group_by('commentId', 'voterId')
            .last()
        )


class ArgumentCommentMap(DataModel):
    table = 'argumentCommentMap'

    schema = {
        'commentId': pl.UInt16,
        'topicId': pl.Int16,
        'argumentId': pl.UInt16,
        'relationship': pl.Categorical,
        'reasoning': pl.List(pl.String),
    }

    def get(self, topicId):
        predicate = (pl.col('topicId') == topicId)
        return self.df.filter(predicate)


class Topics(DataModel):

    schema = {
        'Topic': pl.Int16,
    }

    def get(self, topicId):
        predicate = (pl.col('Topic') == topicId)
        return self.df.row(by_predicate=predicate, named=True)


class HierarchicalTopics(DataModel):

    def get(self, topicId):
        predicate = (pl.col('Parent_ID') == topicId)
        return self.df.row(by_predicate=predicate, named=True)
