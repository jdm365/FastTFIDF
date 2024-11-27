import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from time import perf_counter



def sklearn_bench():
    df = pl.read_parquet('mb.parquet')
    titles = df.select("title").fill_null("").to_series()

    vectorizer = TfidfVectorizer()

    init = perf_counter()
    ## vectorizer.fit(titles)
    vectorizer.fit_transform(titles)
    fit_time = perf_counter() - init
    print(f"fit time: {fit_time}")
    print(f"KDocs per second: {len(titles) * 0.001 / fit_time}")


if __name__ == "__main__":
    sklearn_bench()
