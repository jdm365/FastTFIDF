use rustc_hash::FxHashMap;
use polars::prelude::*;


#[derive(Clone)]
struct TFToken {
    tf: u32,
    term_id: u32,
}

#[derive(Eq, PartialEq, Hash)]
enum NGramKey {
    Bigram([u32; 2]),
    Trigram([u32; 3]),
    Quadgram([u32; 4]),
    Pentagram([u32; 5]),
    Hexagram([u32; 6]),
    Heptagram([u32; 7]),
    Octagram([u32; 8]),
}

struct Vocab {
    vocab: FxHashMap<String, u32>,
    higher_grams: [FxHashMap<NGramKey, u32>; 7],
}

impl Vocab {
    #[inline]
    fn insert(&mut self, key: String, value: u32) {
        self.vocab.insert(key, value);
    }

    #[inline]
    fn insert_ngram(&mut self, key: NGramKey, value: u32) {
        match key {
            NGramKey::Bigram(_) => {
                self.higher_grams[0].insert(key, value);
            },
            NGramKey::Trigram(_) => {
                self.higher_grams[1].insert(key, value);
            },
            NGramKey::Quadgram(_) => {
                self.higher_grams[2].insert(key, value);
            },
            NGramKey::Pentagram(_) => {
                self.higher_grams[3].insert(key, value);
            },
            NGramKey::Hexagram(_) => {
                self.higher_grams[4].insert(key, value);
            },
            NGramKey::Heptagram(_) => {
                self.higher_grams[5].insert(key, value);
            },
            NGramKey::Octagram(_) => {
                self.higher_grams[6].insert(key, value);
            },
        }
    }

    #[inline]
    fn get(&self, key: &str) -> Option<&u32> {
        self.vocab.get(key)
    }
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
    // vocab: FxHashMap<String, u32>,
    vocab: Vocab,
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
    lowercase: bool,
    n_gram_range: (usize, usize),
) -> Result<(), TokenizationError> {

    let mut buffer_idx: usize = 0;
    let mut n_gram_buffer: [u32; 8] = [0; 8];

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
                if lowercase {
                    token_buffer[buffer_idx] = char.to_ascii_lowercase();
                } else {
                    token_buffer[buffer_idx] = *char;
                }
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
    lowercase: bool,
    n_gram_range: (usize, usize),
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
                if lowercase {
                    token_buffer[buffer_idx] = char.to_ascii_lowercase();
                } else {
                    token_buffer[buffer_idx] = *char;
                }
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


fn fit_transform(
    text_series: &Column,
    lowercase: bool,
    n_gram_range: (usize, usize),
    ) -> Result<TfidfVectorizer<f32>, PolarsError> {
    let start_time = std::time::Instant::now();

    assert!(n_gram_range.0 <= n_gram_range.1, "n_gram_range.0 must be less than or equal to n_gram_range.1.");
    assert!(n_gram_range.0 > 0, "n_gram_range.0 must be greater than 0.");
    assert!(n_gram_range.1 - n_gram_range.0 <= 7, "width of n_gram_range must be less than 8.");

    let count = text_series.len();
    assert!(count > 0);

    let mut vectorizer: TfidfVectorizer<f32> = TfidfVectorizer {
        vocab: Vocab {
            vocab: FxHashMap::with_capacity_and_hasher(count / 6, Default::default()),
            higher_grams: [
                FxHashMap::with_capacity_and_hasher((count / 6) * (n_gram_range.1 > 1) as usize, Default::default()),
                FxHashMap::with_capacity_and_hasher((count / 6) * (n_gram_range.1 > 2) as usize, Default::default()),
                FxHashMap::with_capacity_and_hasher((count / 6) * (n_gram_range.1 > 3) as usize, Default::default()),
                FxHashMap::with_capacity_and_hasher((count / 6) * (n_gram_range.1 > 4) as usize, Default::default()),
                FxHashMap::with_capacity_and_hasher((count / 6) * (n_gram_range.1 > 5) as usize, Default::default()),
                FxHashMap::with_capacity_and_hasher((count / 6) * (n_gram_range.1 > 6) as usize, Default::default()),
                FxHashMap::with_capacity_and_hasher((count / 6) * (n_gram_range.1 > 7) as usize, Default::default()),
            ],
        },
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
                lowercase,
                n_gram_range,
            ).unwrap();
        } else {
            process_doc_whitespace(
                _v,
                &mut vectorizer,
                &mut doc_tokens,
                &mut token_buffer,
                lowercase,
                n_gram_range,
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
    _ = fit_transform(
        lf.clone().select([col(column)]).collect()?.column(column)?,
        true,
        (1, 1),
        );

    Ok(())
}
