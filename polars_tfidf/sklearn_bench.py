import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from time import perf_counter



def sklearn_bench():
    df = pl.read_parquet('mb_small.parquet')
    column = "title"

    series = df.select(column).fill_null("").to_series()

    vectorizer = TfidfVectorizer(ngram_range=(1, 1))

    init = perf_counter()
    vectorizer.fit_transform(series)
    fit_time = perf_counter() - init
    print(f"fit transform time: {fit_time}")
    print(f"KDocs per second: {len(series) * 0.001 / fit_time}")
    print(f"Vocab size: {len(vectorizer.vocabulary_) / 1000:.2f}K")


if __name__ == "__main__":
    sklearn_bench()


## fit transform time: 307.3444125908427
## KDocs per second: 63.04002677866569
## Vocab size: 56879.62K

## Time: 46.365886422s
## KDocs per second: 417.8719
## Vocab size: 55975K
