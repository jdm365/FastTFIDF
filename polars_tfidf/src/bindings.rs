use pyo3::prelude::*;
use polars::prelude::*;
use numpy::{PyArray1, ToPyArray};
use pyo3_polars::PySeries;
use rustc_hash::FxHashMap;

use std::collections::HashMap;
use pyo3::types::PyDict; 

use crate::{
    TfidfVectorizer, 
    Vocab, CSRMatrix, 
    fit_transform as _fit_transform,
    transform as _transform,
};



#[pyclass(name = "TfidfVectorizer")]
pub struct PyTfidfVectorizer {
    inner: TfidfVectorizer<f32>,
}

#[pymethods]
impl PyTfidfVectorizer {
    #[new]
    fn new() -> Self {
        PyTfidfVectorizer {
            inner: TfidfVectorizer {
                vocab: Vocab {
                    vocab: FxHashMap::default(),
                    higher_grams: [
                        FxHashMap::default(),
                        FxHashMap::default(),
                        FxHashMap::default(),
                        FxHashMap::default(),
                        FxHashMap::default(),
                        FxHashMap::default(),
                        FxHashMap::default(),
                    ],
                    num_tokens: 0,
                },
                dfs: Vec::new(),
                csr_mat: CSRMatrix {
                    values: Vec::new(),
                    col_idxs: Vec::new(),
                    row_start_pos: Vec::new(),
                },
            }
        }
    }

    #[pyo3(signature = (series, lowercase=true, ngram_range=(1, 1), min_df=None, max_df=None, whitespace_tokenization=true))]
    fn fit_transform(
        &mut self,
        series: PySeries,
        lowercase: bool,
        ngram_range: (usize, usize),
        min_df: Option<usize>,
        max_df: Option<usize>,
        whitespace_tokenization: bool,
        py: Python<'_>,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<u32>>, Py<PyArray1<u64>>)> {
        let series: Series = series.into();
        
        match _fit_transform(
            &series,
            lowercase,
            ngram_range,
            min_df,
            max_df,
            whitespace_tokenization,
        ) {
            Ok(vectorizer) => {
                self.inner = vectorizer;
                Ok(self.to_csr(py)?)
            },
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Error in fit_transform: {}", e)
            )),
        }
    }

    #[pyo3(signature = (series, lowercase=true, ngram_range=(1, 1), whitespace_tokenization=true))]
    fn transform(
        &self,
        series: PySeries,
        lowercase: bool,
        ngram_range: (usize, usize),
        whitespace_tokenization: bool,
        py: Python<'_>,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<u32>>, Py<PyArray1<u64>>)> {
        let series: Series = series.into();
        
        match _transform(
            &series,
            &self.inner,
            lowercase,
            ngram_range,
            whitespace_tokenization,
        ) {
            Ok(csr_mat) => {
                Ok((
                    csr_mat.values.as_slice().to_pyarray(py).to_owned(),
                    csr_mat.col_idxs.as_slice().to_pyarray(py).to_owned(),
                    csr_mat.row_start_pos.as_slice().to_pyarray(py).to_owned()
                ))
            },
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Error in transform: {}", e)
            )),
        }
    }

    #[pyo3(signature = ())]
    fn get_vocab(&self) -> PyResult<Py<PyDict>> {
        Python::with_gil(|py| {
            let vocab: HashMap<String, u32> = self.inner.vocab.vocab.iter().map(|(k, v)| (k.clone(), *v)).collect();
            let dict = PyDict::new(py);
            for (key, value) in vocab.iter() {
                dict.set_item(key, value)?;
            }
            Ok(dict.into_py(py))
        })
    }

    #[pyo3(name = "to_csr")]
    fn to_csr<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<u32>>, Py<PyArray1<u64>>)> {
        Ok((
            self.inner.csr_mat.values.as_slice().to_pyarray(py).to_owned(),
            self.inner.csr_mat.col_idxs.as_slice().to_pyarray(py).to_owned(),
            self.inner.csr_mat.row_start_pos.as_slice().to_pyarray(py).to_owned()
        ))
    }
}

#[pymodule]
fn _rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTfidfVectorizer>()?;
    Ok(())
}
