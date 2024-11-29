use std::collections::HashMap;
use polars::prelude::*;
// use sprs::{CsMat, TriMat, CsMatView};


#[allow(dead_code)]
#[inline]
fn get_lf_columns(lf: &LazyFrame) -> Result<Vec<String>, PolarsError> {
    Ok(lf.clone().collect_schema()?
        .iter()
        .map(|(x, _)| (x.clone().to_string()))
        .collect())
}


#[derive(Clone)]
struct TFToken {
    tf: u32,
    term_id: u32,
}

struct CSRMatrix<T> {
    // data
    values: Vec<T>,

    // indices
    col_idxs: Vec<u32>,

    // indptr
    row_start_pos: Vec<u64>,
}

struct TfidfVectorizer<T> {
    vocab: HashMap<String, u32>,
    dfs: Vec<u32>,
    csr_mat: CSRMatrix<T>,
}


fn fit_tfidf(text_series: &Column) -> Result<TfidfVectorizer<f32>, PolarsError> {
    let mut vectorizer: TfidfVectorizer<f32> = TfidfVectorizer {
        vocab: HashMap::new(),
        dfs: Vec::new(),
        csr_mat: CSRMatrix {
            values: Vec::new(),
            col_idxs: Vec::new(),
            row_start_pos: Vec::new(),
        },
    };
    let count = text_series.len();
    vectorizer.dfs.reserve(count);

    let mut idx: usize = 0;
    let mut doc_tokens: Vec<TFToken> = Vec::new();
    text_series.str()?.into_iter().for_each(|v: Option<&str>| {
        let _v: &str = match v {
            Some(v) => v,
            None => { idx += 1; return; },
        };

        doc_tokens.clear();
        for token in _v.split_whitespace() {
            if let Some(&term_id) = vectorizer.vocab.get(token) {
                match doc_tokens.iter_mut().find(|x| x.term_id == term_id) {
                    Some(item) => item.tf += 1,
                    None => {
                        vectorizer.dfs[term_id as usize] += 1;
                        doc_tokens.push(TFToken{
                            tf: 1,
                            term_id,
                        });
                    },
                }
            } else {
                vectorizer.vocab.insert(token.to_string(), vectorizer.dfs.len() as u32);
                doc_tokens.push(TFToken{
                    tf: 1,
                    term_id: vectorizer.dfs.len() as u32,
                });
                vectorizer.dfs.push(1);
            }
        }

        vectorizer.csr_mat.row_start_pos.push(vectorizer.csr_mat.values.len() as u64);

        for token in doc_tokens.iter() {
            vectorizer.csr_mat.values.push(token.tf as f32);
            vectorizer.csr_mat.col_idxs.push(token.term_id);
        }

        idx += 1;
    });

    let start_time = std::time::Instant::now();
    for (idx, v) in vectorizer.csr_mat.values.iter_mut().enumerate() {
        let df = vectorizer.dfs[vectorizer.csr_mat.col_idxs[idx] as usize];
        let idf = (count as f32/ (1 + df) as f32).ln();
        *v *= idf;
    };
    let end_time = start_time.elapsed();
    println!("SPRS construction time: {:?}", end_time);
    Ok(vectorizer)
}

fn main() -> Result<(), PolarsError> {
    const FILENAME: &str = "mb.parquet";

    let lf = LazyFrame::scan_parquet(FILENAME, Default::default())?;
    // let columns = get_lf_columns(&lf.clone())?;

    let titles = lf.clone().select([col("title")]).collect()?;

    let start_time = std::time::Instant::now();
    _ = fit_tfidf(lf.clone().select([col("title")]).collect()?.column("title")?);
    println!("Time: {:?}", start_time.elapsed());
    println!("KDocs per second: {:?}", 0.001 * titles.height() as f32 / start_time.elapsed().as_secs_f32());

    println!("{:?}", lf.clone().collect());
    println!("{:?}", lf.clone().collect_schema()?);
    // println!("Columns: {:?}", &columns);
    // println!("Titles: {:?}", &titles);

    Ok(())
}
