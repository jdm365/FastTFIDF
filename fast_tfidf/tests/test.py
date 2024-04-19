from fast_tfidf import FastTFIDF
import polars as pl
import numpy as np
import os
from time import perf_counter


def test_sklearn(csv_filename: str, search_col: str = 'name'):
    from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer

    names = pl.read_csv(csv_filename).select(search_col).to_pandas()[search_col].str.upper().tolist()


    init = perf_counter()
    vectorizer = HashingVectorizer()
    X = vectorizer.fit_transform(names)
    print(f"Sklearn HashingVectorizer time: {perf_counter() - init:.2f} seconds")

    init = perf_counter()
    vectorizer = TfidfVectorizer(dtype=np.float32, lowercase=False)
    X = vectorizer.fit_transform(names)
    print(f"Sklearn TfidfVectorizer time: {perf_counter() - init:.2f} seconds")

    print(X.shape, X.dtype, X.nnz)

    return X


def test_documents(csv_filename: str, search_col: str):
    names = pl.read_csv(csv_filename).select(search_col).to_pandas()[search_col].tolist()

    ## model = FastTFIDF(max_df=0.001)
    model = FastTFIDF()

    init = perf_counter()
    X = model.fit_transform(names)
    print(f"FastTFIDF time: {perf_counter() - init:.2f} seconds")
    print(X.shape, X.dtype, X.nnz)

    return X


if __name__ == '__main__':
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_FILENAME = os.path.join(CURRENT_DIR, '../../../SearchApp/data', 'companies_sorted_100M.csv')
    ## CSV_FILENAME = os.path.join(CURRENT_DIR, '../../../SearchApp/data', 'companies_sorted.csv')

    X2 = test_documents(CSV_FILENAME, search_col='name')
    X1 = test_sklearn(CSV_FILENAME, search_col='name')
    ## print(X1, X2)
    ## print(X1.shape, X2.shape)
    ## print(X1 == X2)
