use rustc_hash::FxHashMap;
use polars::prelude::*;


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
    vocab: FxHashMap<String, u32>,
    dfs: Vec<u32>,
    csr_mat: CSRMatrix<T>,
}

#[allow(dead_code)]
#[derive(Debug)]
enum TokenizationError {
    InvalidToken,
}

#[inline]
fn process_doc_whitespace_hashmap(
    text: &str,
    vectorizer: &mut TfidfVectorizer<f32>,
    doc_tokens_hashmap: &mut FxHashMap<u32, u32>,
    token_buffer: &mut [u8; 4096],
) -> Result<(), TokenizationError> {

    let mut buffer_idx: usize = 0;

    doc_tokens_hashmap.clear();
    for char in text.as_bytes().iter() {
        match char {
            b' ' | b'\t' | b'\n' | b'\r' => {
                let str_ref = std::str::from_utf8(&token_buffer[0..buffer_idx]).unwrap();

                if let Some(&term_id) = vectorizer.vocab.get(str_ref) {
                    match doc_tokens_hashmap.get_mut(&term_id) {
                        Some(item) => *item += 1,
                        None => {
                            vectorizer.dfs[term_id as usize] += 1;
                            doc_tokens_hashmap.insert(term_id, 1);
                        },
                    }
                } else {
                    vectorizer.vocab.insert(str_ref.to_string(), vectorizer.dfs.len() as u32);
                    doc_tokens_hashmap.insert(vectorizer.dfs.len() as u32, 1);
                    vectorizer.dfs.push(1);
                }

                buffer_idx = 0;
            },
            _ => {
                token_buffer[buffer_idx] = *char;
                buffer_idx += 1;
            },
        }
    }

    vectorizer.csr_mat.row_start_pos.push(vectorizer.csr_mat.values.len() as u64);

    for (term_id, tf) in doc_tokens_hashmap.iter() {
        vectorizer.csr_mat.values.push(*tf as f32);
        vectorizer.csr_mat.col_idxs.push(*term_id);
    }

    Ok(())
}

#[inline]
fn process_doc_whitespace(
    text: &str,
    vectorizer: &mut TfidfVectorizer<f32>,
    doc_tokens: &mut Vec<TFToken>,
    token_buffer: &mut [u8; 4096],
) -> Result<(), TokenizationError> {

    let mut buffer_idx: usize = 0;

    doc_tokens.clear();
    for char in text.as_bytes().iter() {
        match char {
            b' ' | b'\t' | b'\n' | b'\r' => {
                let str_ref = std::str::from_utf8(&token_buffer[0..buffer_idx]).unwrap();

                if let Some(&term_id) = vectorizer.vocab.get(str_ref) {
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
                    vectorizer.vocab.insert(str_ref.to_string(), vectorizer.dfs.len() as u32);
                    doc_tokens.push(TFToken{
                        tf: 1,
                        term_id: vectorizer.dfs.len() as u32,
                    });
                    vectorizer.dfs.push(1);
                }

                buffer_idx = 0;
            },
            _ => {
                token_buffer[buffer_idx] = *char;
                buffer_idx += 1;
            },
        }
    }

    vectorizer.csr_mat.row_start_pos.push(vectorizer.csr_mat.values.len() as u64);

    for token in doc_tokens.iter() {
        vectorizer.csr_mat.values.push(token.tf as f32);
        vectorizer.csr_mat.col_idxs.push(token.term_id);
    }

    Ok(())
}


fn fit_transform(text_series: &Column) -> Result<TfidfVectorizer<f32>, PolarsError> {
    let start_time = std::time::Instant::now();

    let count = text_series.len();

    let mut vectorizer: TfidfVectorizer<f32> = TfidfVectorizer {
        vocab: FxHashMap::with_capacity_and_hasher(count / 6, Default::default()),
        dfs: Vec::new(),
        csr_mat: CSRMatrix {
            values: Vec::new(),
            col_idxs: Vec::new(),
            row_start_pos: Vec::new(),
        },
    };
    vectorizer.dfs.reserve(count);

    const N: usize = 4096;
    let mut token_buffer: [u8; N] = [0; N];

    let mut idx: usize = 0;
    let mut doc_tokens: Vec<TFToken> = Vec::new();
    let mut doc_tokens_hashmap: FxHashMap<u32, u32> = FxHashMap::with_capacity_and_hasher(256, Default::default());

    text_series.str()?.into_iter().for_each(|v: Option<&str>| {
        let _v: &str = match v {
            Some(v) => v,
            None => { idx += 1; return; },
        };

        if _v.len() > 256 {
            process_doc_whitespace_hashmap(
                _v,
                &mut vectorizer,
                &mut doc_tokens_hashmap,
                &mut token_buffer,
            ).unwrap();
        } else {
            process_doc_whitespace(
                _v,
                &mut vectorizer,
                &mut doc_tokens,
                &mut token_buffer,
            ).unwrap();
        }

        idx += 1;
    });

    // Add the last row. Necessary for csr indptr format.
    vectorizer.csr_mat.row_start_pos.push(vectorizer.csr_mat.values.len() as u64);

    for (idx, v) in vectorizer.csr_mat.values.iter_mut().enumerate() {
        let df = vectorizer.dfs[vectorizer.csr_mat.col_idxs[idx] as usize];
        let idf = (count as f32/ (1 + df) as f32).ln();
        *v *= idf;
    };

    let elapsed = start_time.elapsed();
    println!("Time: {:?}", elapsed);
    println!("KDocs per second: {:?}", 0.001 * count as f32 / elapsed.as_secs_f32());
    Ok(vectorizer)
}

fn main() -> Result<(), PolarsError> {
    const FILENAME: &str = "mb.parquet";
    const N_ROWS: u32 = std::u32::MAX;

    let lf = LazyFrame::scan_parquet(FILENAME, Default::default())?.limit(N_ROWS);
    // println!("{:?}", lf.clone().collect());
    // println!("{:?}", lf.clone().collect_schema()?);

    let column = "title";
    _ = fit_transform(lf.clone().select([col(column)]).collect()?.column(column)?);

    Ok(())
}
