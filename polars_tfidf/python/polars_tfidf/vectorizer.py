from typing import Optional, Tuple, Union, List
import polars as pl
import scipy.sparse as sp
from random import random

from polars_tfidf._rust import TfidfVectorizer as RustTfidfVectorizer

class TfidfVectorizer:
    def __init__(self):
        self._vectorizers = {}

        self.lowercase = True
        self.ngram_range = (1, 1)
        self.min_df = 1
        self.max_df = None
        self.whitespace_tokenization = True
        self.boost_factors = None

        ## Include random hash in default col string name to avoid collisions.
        self.hash = str(hash(random()))

    def _fit_transform(
        self,
        text: pl.Series,
        vectorizer: RustTfidfVectorizer,
        ) -> sp.csr_matrix:

        X = vectorizer.fit_transform(
                text, 
                lowercase=self.lowercase, 
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df,
                whitespace_tokenization=self.whitespace_tokenization,
                )
        self.dim = int(max(X[1]) + 1)

        return self.to_csr(*X)

    def fit_transform(
        self,
        text: Union[pl.Series, pl.DataFrame],
        col_names: Optional[List[str]] = None,
        boost_factors: Optional[List[float]] = None,
        lowercase: bool = True,
        ngram_range: Tuple[int, int] = (1, 1),
        min_df: Optional[int] = None,
        max_df: Optional[int] = None,
        whitespace_tokenization: bool = True,
    ) -> sp.csr_matrix:
        self.lowercase   = lowercase
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.whitespace_tokenization = whitespace_tokenization

        ## TODO: Add lazyframe support.
        if isinstance(text, pl.DataFrame):
            assert col_names is not None, "col_names must be provided when text is a DataFrame"

            if boost_factors is None:
                boost_factors = [1.0] * len(col_names)

            boost_factors = [x / sum(boost_factors) for x in boost_factors]
            self.boost_factors = {
                    col: boost for col, boost in zip(col_names, boost_factors)
                    }

            Xs = []
            for col in col_names:
                vectorizer = RustTfidfVectorizer()
                self._vectorizers[col] = vectorizer

                col_data = text.select(col).fill_null("").to_series()
                X = self._fit_transform(
                    col_data,
                    vectorizer,
                    ) * self.boost_factors[col]
                Xs.append(X)

            return sp.hstack(Xs, format="csr")

        ## Series
        self.boost_factors = {
                f"default_series_{hash}": 1.0,
                }
        vectorizer = RustTfidfVectorizer()
        self._vectorizers[f"default_series_{hash}"] = vectorizer

        return self._fit_transform(
                text,
                vectorizer,
                )

    def transform(
            self, 
            text: Union[pl.Series, pl.DataFrame],
            col_names: Optional[List[str]] = None,
            ) -> sp.csr_matrix:
        if isinstance(text, pl.DataFrame):
            assert col_names is not None, "col_names must be provided when text is a DataFrame"

        Xs = []
        for col, vectorizer in self._vectorizers.items():
            if isinstance(text, pl.DataFrame):
                text_col = text.select(col).fill_null("").to_series()
            else:
                text_col = text.fill_null("")

            X = vectorizer.transform(
                    text_col,
                    self.lowercase,
                    self.ngram_range,
                    self.whitespace_tokenization,
                    )
            X = self.to_csr(*X) * self.boost_factors[col]
            Xs.append(X)

        X = sp.hstack(Xs, format="csr")

        return X

    def to_csr(self, data, indices, indptr) -> sp.csr_matrix:
        if len(data) == 0:
            return sp.csr_matrix((0, 0))

        return sp.csr_matrix(
                arg1=(data, indices, indptr), 
                shape=(len(indptr) - 1, self.dim),
                )

    def get_vocab(self) -> dict:
        vocabs = {}
        for col, vectorizer in self._vectorizers.items():
            vocabs[col] = vectorizer.get_vocab()
        return vocabs
