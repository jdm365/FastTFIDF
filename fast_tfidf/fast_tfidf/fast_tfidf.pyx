
# cython: language_level=3

cimport cython
from cython.view cimport array as cvarray

from libc.stdint cimport uint32_t, uint64_t
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.pair cimport pair
from libcpp cimport bool

import numpy as np
cimport numpy as cnp
from scipy.sparse import csr_matrix
from time import perf_counter
import os


cdef extern from "engine.h":
    cdef struct CSRData:
        vector[float] values
        vector[uint32_t]   idxs
        vector[uint32_t]   ptrs

    cdef cppclass _TFIDF:
        _TFIDF(int min_df, float max_df) nogil
        CSRData fit_transform(const vector[string]& docs) nogil
        CSRData transform(const vector[string]& docs) nogil


cdef class FastTFIDF:
    cdef _TFIDF* tfidf
    cdef int     min_df
    cdef float   max_df


    def __init__(self, int   min_df = 1, float max_df = 1.0):
        self.min_df = min_df
        self.max_df = max_df

    def fit_transform(self, list documents):
        init = perf_counter()
        self.tfidf = new _TFIDF(
                self.min_df,
                self.max_df
                )
        cdef vector[string] docs
        docs.reserve(len(documents))
        cdef str doc
        for doc in documents:
            docs.push_back(doc.upper().encode("utf-8"))

        print(f"init: {perf_counter() - init:.2f}")

        init = perf_counter()
        cdef CSRData data = self.tfidf.fit_transform(docs)
        print(f"fit_transform: {perf_counter() - init:.2f}")

        init = perf_counter()

        cdef float[:] values_view = <float[:data.values.size()]>&data.values[0]
        cdef cnp.ndarray[cnp.float32_t, ndim=1] values = np.asarray(values_view, dtype=np.float32)

        cdef uint32_t[:] idxs_view = <uint32_t[:data.idxs.size()]>&data.idxs[0]
        cdef cnp.ndarray[cnp.uint32_t, ndim=1] idxs = np.asarray(idxs_view, dtype=np.uint32)

        cdef uint32_t[:] ptrs_view = <uint32_t[:data.ptrs.size()]>&data.ptrs[0]
        cdef cnp.ndarray[cnp.uint32_t, ndim=1] ptrs = np.asarray(ptrs_view, dtype=np.uint32)

        X = csr_matrix((values, idxs, ptrs), dtype=np.float32)

        print(f"np.array: {perf_counter() - init:.2f}")
        return X


    def transform(self, list documents):
        cdef vector[string] docs
        docs.reserve(len(documents))
        cdef str doc
        for doc in documents:
            docs.push_back(doc.upper().encode("utf-8"))

        cdef CSRData data = self.tfidf.transform(docs)
        return csr_matrix((data.values, data.idxs, data.ptrs), dtype=np.float32)
