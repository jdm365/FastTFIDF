import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from time import perf_counter

from polars_tfidf import TfidfVectorizer as TfidfVectorizerPolars



def sklearn_bench(filename: str):
    df = pl.read_parquet(filename)
    column = "title"

    series = df.select(column).fill_null("").to_series()

    vectorizer = TfidfVectorizer(ngram_range=(1, 1))

    init = perf_counter()
    vectorizer.fit_transform(series)
    fit_time = perf_counter() - init
    print(f"fit transform time: {fit_time}")
    print(f"KDocs per second: {len(series) * 0.001 / fit_time}")
    print(f"Vocab size: {len(vectorizer.vocabulary_) / 1000:.2f}K\n\n")


def polars_bench(filename: str):
    df = pl.read_parquet(filename)
    column = "title"

    series = df.select(column).fill_null("").to_series()

    vectorizer = TfidfVectorizerPolars()

    init = perf_counter()
    X = vectorizer.fit_transform(series, ngram_range=(1, 1), lowercase=True, return_csr=True)
    fit_time = perf_counter() - init
    print(f"fit transform time: {fit_time}")
    print(f"KDocs per second: {len(series) * 0.001 / fit_time}\n\n")
    ## print(f"Vocab size: {len(vectorizer.vocabulary_) / 1000:.2f}K")

    print(X)


if __name__ == "__main__":
    FILENAME = "mb_small.parquet"

    sklearn_bench(filename=FILENAME)
    polars_bench(filename=FILENAME)


## fit transform time: 307.3444125908427
## KDocs per second: 63.04002677866569
## Vocab size: 56879.62K

## Time: 46.365886422s
## KDocs per second: 417.8719
## Vocab size: 55975K
