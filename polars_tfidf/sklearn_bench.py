import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from time import perf_counter



def sklearn_bench():
    df = pl.read_parquet('mb.parquet')
    column = "title"

    series = df.select(column).fill_null("").to_series()

    vectorizer = TfidfVectorizer()

    init = perf_counter()
    vectorizer.fit_transform(series)
    fit_time = perf_counter() - init
    print(f"fit transform time: {fit_time}")
    print(f"KDocs per second: {len(series) * 0.001 / fit_time}")


if __name__ == "__main__":
    sklearn_bench()
