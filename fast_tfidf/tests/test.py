from fast_tfidf import FastTFIDF
import polars as pl
import os
from time import perf_counter


def test_sklearn(csv_filename: str, search_col: str = 'name'):
    from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer

    names = pl.read_csv(csv_filename).select(search_col).to_pandas()[search_col].tolist()

    init = perf_counter()
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(names)
    print(f"Sklearn TfidfVectorizer time: {perf_counter() - init:.2f} seconds")

    init = perf_counter()
    vectorizer = HashingVectorizer()
    X = vectorizer.fit_transform(names)
    print(f"Sklearn HashingVectorizer time: {perf_counter() - init:.2f} seconds")


def test_documents(csv_filename: str, search_col: str):
    ## names = pd.read_csv(csv_filename)[search_col].tolist()
    names = pl.read_csv(csv_filename).select(search_col).to_pandas()[search_col].tolist()

    model = FastTFIDF()

    init = perf_counter()
    X = model.fit_transform(names)
    print(f"FastTFIDF time: {perf_counter() - init:.2f} seconds")


if __name__ == '__main__':
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_FILENAME = os.path.join(CURRENT_DIR, '../../../SearchApp/data', 'companies_sorted.csv')

    test_sklearn(CSV_FILENAME, search_col='name')
    test_documents(CSV_FILENAME, search_col='name')
