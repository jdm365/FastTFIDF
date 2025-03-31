from typing import Optional, Tuple
import polars as pl

try:
    import scipy.sparse as sp
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from polars_tfidf._rust import TfidfVectorizer as RustTfidfVectorizer

class TfidfVectorizer:
    def __init__(self):
        self._vectorizer = RustTfidfVectorizer()

        self.lowercase = True
        self.ngram_range = (1, 1)
        self.min_df = 1
        self.max_df = None
        self.whitespace_tokenization = True

    def fit_transform(
        self,
        text: pl.Series,
        lowercase: bool = True,
        ngram_range: Tuple[int, int] = (1, 1),
        min_df: Optional[int] = None,
        max_df: Optional[int] = None,
        whitespace_tokenization: bool = True,
        return_csr: bool = True,
    ):
        self.lowercase = lowercase
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.whitespace_tokenization = whitespace_tokenization

        X = self._vectorizer.fit_transform(
                text, 
                lowercase=lowercase, 
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
                whitespace_tokenization=whitespace_tokenization,
                )

        if return_csr:
            return self.to_csr(*X)

        return X

    def transform(self, text: pl.Series, return_csr: bool = True):
        X = self._vectorizer.transform(
                text,
                self.lowercase,
                self.ngram_range,
                self.whitespace_tokenization,
                )

        if return_csr:
            return self.to_csr(*X)

    @staticmethod
    def to_csr(data, indices, indptr):
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "scipy is required for to_csr(). "
                "Install with 'pip install scipy'"
            )
            
        return sp.csr_matrix((data, indices, indptr))

    def get_vocab(self):
        return self._vectorizer.get_vocab()
