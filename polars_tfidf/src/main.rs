use std::collections::HashMap;
use polars::prelude::*;
use sprs::{CsMat, TriMat, CsMatView};


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
    doc_id: u32,
}

struct CSRMatrix<T> {
    // data
    values: Vec<T>,

    // indices
    col_idxs: Vec<u32>,

    // indptr
    row_start_pos: Vec<u64>,
}

#[derive(Clone)]
struct Tokens {
    tf: Vec<u32>,
    term_id: Vec<u32>,
    doc_id: Vec<u32>,
}

#[allow(dead_code)]
impl Tokens {
    #[inline]
    fn push(&mut self, tf: u32, term_id: u32, doc_id: u32) {
        self.tf.push(tf);
        self.term_id.push(term_id);
        self.doc_id.push(doc_id);
    }

    #[inline]
    fn push_token(&mut self, token: &TFToken) {
        self.tf.push(token.tf);
        self.term_id.push(token.term_id);
        self.doc_id.push(token.doc_id);
    }

    #[inline]
    fn get(&self, idx: usize) -> (u32, u32, u32) {
        (self.tf[idx], self.term_id[idx], self.doc_id[idx])
    }

    #[inline]
    fn len(&self) -> usize {
        self.tf.len()
    }

    fn new() -> Self {
        Tokens {
            tf: Vec::new(),
            term_id: Vec::new(),
            doc_id: Vec::new(),
        }
    }
}

#[allow(dead_code)]
#[derive(Clone)]
struct TfidfVectorizer {
    vocab: HashMap<String, u32>,
    dfs: Vec<u32>,
    tokens: Tokens,
}


fn fit_tfidf(text_series: &Column) -> Result<TfidfVectorizer, PolarsError> {
    let mut vectorizer = TfidfVectorizer {
        vocab: HashMap::new(),
        dfs: Vec::new(),
        tokens: Tokens::new(),
    };

    let count = text_series.len();
    vectorizer.dfs.reserve(count);

    // TODO: Calc indptrs through as cumsum of dfs
    // data: tfidfs
    // indices: term_ids
    let mut csr_matrix: CSRMatrix<f32> = CSRMatrix {
        values: Vec::new(),
        col_idxs: Vec::new(),
        row_start_pos: Vec::new(),
    };

    let mut idx: usize = 0;
    let mut doc_tokens: Vec<TFToken> = Vec::new();
    text_series.str()?.into_iter().for_each(|v: Option<&str>| {
        let _v = match v {
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
                            doc_id: idx as u32,
                        });
                    },
                }
            } else {
                vectorizer.vocab.insert(token.to_string(), vectorizer.dfs.len() as u32);
                doc_tokens.push(TFToken{
                    tf: 1,
                    term_id: vectorizer.dfs.len() as u32,
                    doc_id: idx as u32,
                });
                vectorizer.dfs.push(1);
            }
        }

        for token in doc_tokens.iter() {
            vectorizer.tokens.push_token(token);
        }

        idx += 1;
    });

    let start_time = std::time::Instant::now();

    let mut trimat: TriMat<f32> = TriMat::new((count, vectorizer.vocab.len()));
    for i in 0..vectorizer.tokens.len() {
        let (tf, term_id, doc_id) = vectorizer.tokens.get(i);

        let tfidf = tf as f32 * (count as f32 / vectorizer.dfs[term_id as usize] as f32).ln();
        trimat.add_triplet(doc_id as usize, term_id as usize, tfidf);
    };

    // let cs_mat_view = CsMatView::new(
        // (count, vectorizer.vocab.len()),
        // vectorizer.tokens.tf.as_slice(),
        // vectorizer.tokens.term_id.as_slice(),
        // vectorizer.tokens.doc_id.as_slice(),
    // );

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
