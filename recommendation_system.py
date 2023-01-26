import pandas as pd
from sklearn.neighbors import NearestNeighbors
from env import Env
from sqlalchemy import create_engine
from preprocessing import preprocessing
import pymysql.cursors


class RecommendationSystem:
    def __init__(self, env: Env, n_neighbors=20, metric="cosine", batch_size=1000):
        self.n_neighbors = n_neighbors
        self.env = env
        self.df = self.read_data()
        self.df_preprocessed = preprocessing(self.df)
        self.nearest_neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, n_jobs=-1).fit(
            self.df_preprocessed)
        self.batch_size = batch_size

    @staticmethod
    def read_data():
        df = pd.read_csv("winemag-data-130k-v2.csv.zip")
        df = df.drop(columns=["Unnamed: 0", "designation", "taster_name", "taster_twitter_handle", "region_2"])
        df = df.dropna(subset=['country', 'variety'])
        df["region_1"] = df["region_1"].fillna("")
        df["price"] = df["price"].fillna(df["price"].median())
        df = df.drop_duplicates(subset=["title"]) \
            .drop_duplicates() \
            .reset_index(drop=True)
        return df

    def remove_data_in_db(self):
        connection = pymysql.connect(host=self.env.HOST_MYSQL,
                                     user=self.env.USER_MYSQL,
                                     password=self.env.PWD_MYSQL,
                                     database=self.env.DB_MYSQL,
                                     ssl={"fake_flag_to_enable_tls": True})
        cursor = connection.cursor()
        cursor.execute("TRUNCATE `similarities`;")
        cursor.fetchall()

    def write_in_db(self, df):
        engine = create_engine(
            f'mysql+pymysql://{self.env.USER_MYSQL}:{self.env.PWD_MYSQL}@{self.env.HOST_MYSQL}/{self.env.DB_MYSQL}',
            ssl={"fake_flag_to_enable_tls": True})
        df.to_sql('similarities', engine, if_exists='append', index=False)

    def create_dataframe_similarities(self, indexes):
        wine_ids = []
        wine_similarity = []
        same_country = []
        ranks = []
        for i in range(len(indexes)):
            id_ = indexes[i, 0]
            country = self.df.iloc[id_]["country"]

            index_same_country = self.df.iloc[indexes[i, 1:]].loc[self.df["country"] == country].index.tolist()

            if len(index_same_country) > self.n_neighbors:
                index_same_country = index_same_country[:self.n_neighbors]

            index_world = self.df.iloc[indexes[i, 1:]].loc[self.df["country"] != country].index[
                          :self.n_neighbors].tolist()

            wine_ids += (len(index_world) + len(index_same_country)) * [id_]
            wine_similarity += index_same_country + index_world
            same_country += len(index_same_country) * [True] + len(index_world) * [False]
            ranks += list(range(1, len(index_same_country) + 1)) + list(range(1, len(index_world) + 1))

        df_similarities = pd.DataFrame({"wine_id": wine_ids,
                                        "wine_id_similarity": wine_similarity,
                                        "rank": ranks,
                                        "same_country": same_country})
        return df_similarities

    def run(self):
        self.remove_data_in_db()
        for batch in range(0, len(self.df_preprocessed), self.batch_size):
            print(f"Batch n°{batch}")
            distances, indexes = self.nearest_neighbors.kneighbors(
                self.df_preprocessed.iloc[batch:batch + self.batch_size],
                n_neighbors=self.df_preprocessed.shape[0] // 2)
            df_similarities = self.create_dataframe_similarities(indexes)
            self.write_in_db(df_similarities)
