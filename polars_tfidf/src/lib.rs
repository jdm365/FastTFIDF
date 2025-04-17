use rustc_hash::FxHashMap;
use polars::prelude::*;

pub mod bindings;


struct FiFo {
    data: [u32; 8],
    capacity: usize,
}

impl FiFo {
    #[inline]
    fn push(&mut self, val: u32) {
        self.data.rotate_left(1);
        self.data[self.capacity] = val;
    }

    #[inline]
    fn get_last(&self, size: usize) -> &[u32] {
        &self.data[(self.capacity - size)..self.capacity]
    }
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
    num_tokens: usize,
}

impl Vocab {
    #[inline]
    fn insert(&mut self, key: String, value: u32) {
        self.vocab.insert(key, value);
    }

    #[inline]
    fn insert_ngram_arr(&mut self, key: &[u32], value: u32) {
        match key.len() {
            2 => {
                self.higher_grams[0].insert(NGramKey::Bigram(key.try_into().unwrap()), value);
            },
            3 => {
                self.higher_grams[1].insert(NGramKey::Trigram(key.try_into().unwrap()), value);
            },
            4 => {
                self.higher_grams[2].insert(NGramKey::Quadgram(key.try_into().unwrap()), value);
            },
            5 => {
                self.higher_grams[3].insert(NGramKey::Pentagram(key.try_into().unwrap()), value);
            },
            6 => {
                self.higher_grams[4].insert(NGramKey::Hexagram(key.try_into().unwrap()), value);
            },
            7 => {
                self.higher_grams[5].insert(NGramKey::Heptagram(key.try_into().unwrap()), value);
            },
            8 => {
                self.higher_grams[6].insert(NGramKey::Octagram(key.try_into().unwrap()), value);
            },
            _ => panic!("Invalid ngram size."),
        }
    }

    #[inline]
    fn get(&self, key: &str) -> Option<&u32> {
        self.vocab.get(key)
    }

    #[inline]
    fn get_ngram_arr(&self, key: &[u32]) -> Option<&u32> {
        match key.len() {
            2 => {
                self.higher_grams[0].get(&NGramKey::Bigram(key.try_into().unwrap()))
            },
            3 => {
                self.higher_grams[1].get(&NGramKey::Trigram(key.try_into().unwrap()))
            },
            4 => {
                self.higher_grams[2].get(&NGramKey::Quadgram(key.try_into().unwrap()))
            },
            5 => {
                self.higher_grams[3].get(&NGramKey::Pentagram(key.try_into().unwrap()))
            },
            6 => {
                self.higher_grams[4].get(&NGramKey::Hexagram(key.try_into().unwrap()))
            },
            7 => {
                self.higher_grams[5].get(&NGramKey::Heptagram(key.try_into().unwrap()))
            },
            8 => {
                self.higher_grams[6].get(&NGramKey::Octagram(key.try_into().unwrap()))
            },
            _ => panic!("Invalid ngram size."),
        }
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

fn l2_norm(
    values: &mut Vec<f32>,
    _: &mut Vec<u32>,
    row_start_pos: &mut Vec<u64>,
) {
    let num_rows = row_start_pos.len() - 1;

    for row in 0..num_rows {
        let start_idx = row_start_pos[row] as usize;
        let end_idx = row_start_pos[row + 1] as usize;

        // Calculate L2 norm for the row
        let mut sum_of_squares: f32 = 0.0;
        for i in start_idx..end_idx {
            sum_of_squares += values[i] * values[i];
        }

        let norm = sum_of_squares.sqrt();

        // Normalize the row
        if norm > 0.0 {
            for i in start_idx..end_idx {
                values[i] /= norm;
            }
        }
    }
}



struct TfidfVectorizer<T> {
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
fn add_ngrams_range(
    vectorizer: &mut TfidfVectorizer<f32>,
    doc_tokens_hashmap: &mut FxHashMap<u32, u32>,
    ngram_queue: &FiFo,
    min_size: usize,
    max_size: usize,
) {
    assert!(min_size > 1, "ngram_size must be greater than 1.");
    assert!(max_size <= 8, "ngram_size must be greater than 1.");

    for ngram_size in min_size..max_size+1 {
        let arr = ngram_queue.get_last(ngram_size);
        for &item in arr.iter() {
            if item == std::u32::MAX {
                continue;
            }
        }

        if let Some(&term_id) = vectorizer.vocab.get_ngram_arr(&arr) {
            // Term already exists in vocab.
            match doc_tokens_hashmap.get_mut(&term_id) {
                Some(item) => *item += 1,
                None => {
                    vectorizer.dfs[term_id as usize] += 1;
                    doc_tokens_hashmap.insert(term_id, 1);
                },
            }
        } else {
            // Term does not exist in vocab.
            let term_id = vectorizer.vocab.num_tokens as u32;
            doc_tokens_hashmap.insert(term_id, 1);

            vectorizer.vocab.insert_ngram_arr(arr, term_id);
            doc_tokens_hashmap.insert(term_id, 1);
            vectorizer.dfs.push(1);

            vectorizer.vocab.num_tokens += 1;
        }
    }
}

#[inline]
fn process_doc_whitespace_hashmap_queue(
    text: &str,
    vectorizer: &mut TfidfVectorizer<f32>,
    doc_tokens_hashmap: &mut FxHashMap<u32, u32>,
    token_buffer: &mut [u8; 4096],
    lowercase: bool,
    ngram_range: (usize, usize),
    max_df: usize,
) -> Result<(), TokenizationError> {

    let ngram_width = 1 + ngram_range.1 - ngram_range.0;
    let mut ngram_queue: FiFo = FiFo {
        data: [0; 8],
        capacity: ngram_width,
    };

    /////////////////////////////////////////////////////////
    //                    Queue Adding                     //
    //                                                     //
    // First iter  - Add all first -> max grams in window  //
    // Other iters - Add only final grams for all ranges   //
    //               in window.                            //
    /////////////////////////////////////////////////////////

    let mut buffer_idx: usize = 0;
    doc_tokens_hashmap.clear();
    for char in text.as_bytes().iter() {
        match char {
            0..=47 | 58..=64 | 91..=96 | 123..=126 => {

                if buffer_idx == 0 {
                    continue;
                }
                let str_ref = std::str::from_utf8(&token_buffer[0..buffer_idx]).unwrap();

                if let Some(&term_id) = vectorizer.vocab.get(str_ref) {
                    match doc_tokens_hashmap.get_mut(&term_id) {
                        Some(item) => *item += 1,
                        None => {
                            vectorizer.dfs[term_id as usize] += 1;
                            if vectorizer.dfs[term_id as usize] <= max_df as u32 {
                                doc_tokens_hashmap.insert(term_id, 1);
                            }

                            ngram_queue.push(term_id);
                        },
                    }
                } else {
                    let term_id = vectorizer.vocab.num_tokens as u32;

                    vectorizer.vocab.insert(str_ref.to_string(), term_id);
                    doc_tokens_hashmap.insert(term_id, 1);
                    vectorizer.dfs.push(1);

                    vectorizer.vocab.num_tokens += 1;

                    ngram_queue.push(term_id);
                }

                buffer_idx = 0;

                if ngram_range.1 > 1 {
                    add_ngrams_range(
                        vectorizer, 
                        doc_tokens_hashmap, 
                        &ngram_queue, 
                        ngram_range.0.max(2),
                        ngram_range.1,
                        );
                }
            },
            _ => {
                if lowercase {
                    token_buffer[buffer_idx] = char.to_ascii_lowercase();
                } else {
                    token_buffer[buffer_idx] = *char;
                }
                if *char < 128 {
                    buffer_idx += 1;
                }
            },
        }
    }

    if buffer_idx > 0 {
        let str_ref = std::str::from_utf8(&token_buffer[0..buffer_idx]).unwrap();

        if let Some(&term_id) = vectorizer.vocab.get(str_ref) {
            match doc_tokens_hashmap.get_mut(&term_id) {
                Some(item) => *item += 1,
                None => {
                    vectorizer.dfs[term_id as usize] += 1;
                    if vectorizer.dfs[term_id as usize] <= max_df as u32 {
                        doc_tokens_hashmap.insert(term_id, 1);
                    }

                    ngram_queue.push(term_id);
                },
            }
        } else {
            let term_id = vectorizer.vocab.num_tokens as u32;

            vectorizer.vocab.insert(str_ref.to_string(), term_id);
            doc_tokens_hashmap.insert(term_id, 1);
            vectorizer.dfs.push(1);

            vectorizer.vocab.num_tokens += 1;

            ngram_queue.push(term_id);
        }

        if ngram_range.1 > 1 {
            add_ngrams_range(
                vectorizer, 
                doc_tokens_hashmap, 
                &ngram_queue, 
                ngram_range.0.max(2),
                ngram_range.1,
                );
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
fn process_doc_char_hashmap_queue(
    text: &str,
    vectorizer: &mut TfidfVectorizer<f32>,
    doc_tokens_hashmap: &mut FxHashMap<u32, u32>,
    token_buffer: &mut [u8; 4096],
    lowercase: bool,
    ngram_range: (usize, usize),
    max_df: usize,
) -> Result<(), TokenizationError> {

    let ngram_width = 1 + ngram_range.1 - ngram_range.0;
    let mut ngram_queue: FiFo = FiFo {
        data: [0; 8],
        capacity: ngram_width,
    };

    /////////////////////////////////////////////////////////
    //                    Queue Adding                     //
    //                                                     //
    // First iter  - Add all first -> max grams in window  //
    // Other iters - Add only final grams for all ranges   //
    //               in window.                            //
    /////////////////////////////////////////////////////////
    let mut cntr: usize = ngram_range.0;

    let mut buffer_idx: usize = 0;
    doc_tokens_hashmap.clear();
    for char in text.as_bytes().iter() {
        match cntr {
            0 => {
                cntr = ngram_range.0;

                if buffer_idx == 0 {
                    continue;
                }

                let str_ref = std::str::from_utf8(&token_buffer[0..buffer_idx]).unwrap();

                if let Some(&term_id) = vectorizer.vocab.get(str_ref) {
                    match doc_tokens_hashmap.get_mut(&term_id) {
                        Some(item) => *item += 1,
                        None => {
                            vectorizer.dfs[term_id as usize] += 1;
                            if vectorizer.dfs[term_id as usize] <= max_df as u32 {
                                doc_tokens_hashmap.insert(term_id, 1);
                            }

                            ngram_queue.push(term_id);
                        },
                    }
                } else {
                    let term_id = vectorizer.vocab.num_tokens as u32;

                    vectorizer.vocab.insert(str_ref.to_string(), term_id);
                    doc_tokens_hashmap.insert(term_id, 1);
                    vectorizer.dfs.push(1);

                    vectorizer.vocab.num_tokens += 1;

                    ngram_queue.push(term_id);
                }

                buffer_idx = 0;

                if ngram_range.1 > 1 {
                    add_ngrams_range(
                        vectorizer, 
                        doc_tokens_hashmap, 
                        &ngram_queue, 
                        ngram_range.0.max(2),
                        ngram_range.1,
                        );
                }
            },
            _ => {
                if lowercase {
                    token_buffer[buffer_idx] = char.to_ascii_lowercase();
                } else {
                    token_buffer[buffer_idx] = *char;
                }

                if *char < 128 {
                    buffer_idx += 1;
                    cntr -= 1;
                }
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

fn fit_transform(
    text_series: &Series,
    lowercase: bool,
    ngram_range: (usize, usize),
    _min_df: Option<usize>,
    _max_df: Option<usize>,
    whitespace_tokenization: bool,
    ) -> Result<TfidfVectorizer<f32>, PolarsError> {
    let min_df = _min_df.unwrap_or(0);
    let max_df = _max_df.unwrap_or(std::usize::MAX);

    assert!(ngram_range.0 <= ngram_range.1, "ngram_range.0 must be less than or equal to ngram_range.1.");
    assert!(ngram_range.0 > 0, "ngram_range.0 must be greater than 0.");
    assert!(ngram_range.1 - ngram_range.0 <= 7, "width of ngram_range must be less than 8.");

    let count = text_series.len();
    assert!(count > 0);

    let mut vectorizer: TfidfVectorizer<f32> = TfidfVectorizer {
        vocab: Vocab {
            vocab: FxHashMap::with_capacity_and_hasher(count / 6, Default::default()),
            higher_grams: [
                FxHashMap::with_capacity_and_hasher((count / 6) * (ngram_range.1 > 1) as usize, Default::default()),
                FxHashMap::with_capacity_and_hasher((count / 6) * (ngram_range.1 > 2) as usize, Default::default()),
                FxHashMap::with_capacity_and_hasher((count / 6) * (ngram_range.1 > 3) as usize, Default::default()),
                FxHashMap::with_capacity_and_hasher((count / 6) * (ngram_range.1 > 4) as usize, Default::default()),
                FxHashMap::with_capacity_and_hasher((count / 6) * (ngram_range.1 > 5) as usize, Default::default()),
                FxHashMap::with_capacity_and_hasher((count / 6) * (ngram_range.1 > 6) as usize, Default::default()),
                FxHashMap::with_capacity_and_hasher((count / 6) * (ngram_range.1 > 7) as usize, Default::default()),
            ],
            num_tokens: 0,
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
    let mut doc_tokens_hashmap: FxHashMap<u32, u32> = FxHashMap::with_capacity_and_hasher(256, Default::default());

    text_series.utf8()?.into_iter().for_each(|v: Option<&str>| {
        let _v: &str = match v {
            Some(v) => v,
            None => { idx += 1; return; },
        };

        if whitespace_tokenization {
            process_doc_whitespace_hashmap_queue(
                _v,
                &mut vectorizer,
                &mut doc_tokens_hashmap,
                &mut token_buffer,
                lowercase,
                ngram_range,
                max_df,
            ).unwrap();
        } else {
            process_doc_char_hashmap_queue(
                _v,
                &mut vectorizer,
                &mut doc_tokens_hashmap,
                &mut token_buffer,
                lowercase,
                ngram_range,
                max_df,
            ).unwrap();
        }

        idx += 1;
    });

    let mut remove_idxs: Vec<usize> = Vec::new();
    for (idx, v) in vectorizer.csr_mat.values.iter_mut().enumerate() {
        let val = vectorizer.csr_mat.col_idxs[idx] as usize;
        let df = vectorizer.dfs[val];

        if (df < min_df as u32) || (df > max_df as u32) {
            remove_idxs.push(idx);
            continue;
        }

        let idf = (count as f32 / (1 + df) as f32).ln();
        *v *= idf;
    };

    for &index in remove_idxs.iter().rev() {
        vectorizer.csr_mat.values.swap_remove(index);
        vectorizer.csr_mat.col_idxs.swap_remove(index);
    }
    vectorizer.csr_mat.values.shrink_to_fit();
    vectorizer.csr_mat.col_idxs.shrink_to_fit();

    let mut prev_idx: u32 = 0;
    for (idx, v) in vectorizer.csr_mat.col_idxs.iter().enumerate() {
        if *v < prev_idx {
            prev_idx = *v;
            vectorizer.csr_mat.row_start_pos.push(idx as u64);
        }
    }

    // Add the last row. Necessary for csr indptr format.
    vectorizer.csr_mat.row_start_pos.push(vectorizer.csr_mat.values.len() as u64);

    Ok(vectorizer)
}


#[inline]
fn add_ngrams_range_transform(
    vectorizer: &TfidfVectorizer<f32>,
    doc_tokens_hashmap: &mut FxHashMap<u32, u32>,
    ngram_queue: &FiFo,
    min_size: usize,
    max_size: usize,
) {
    assert!(
        min_size > 1,
        "ngram_size must be greater than 1."
    );
    assert!(
        max_size <= 8,
        "ngram_size must be less than or equal to 8."
    );

    for ngram_size in min_size..=max_size {
        let arr = ngram_queue.get_last(ngram_size);
        if arr.iter().any(|&item| item == std::u32::MAX) {
            continue;
        }

        if let Some(&term_id) = vectorizer.vocab.get_ngram_arr(arr) {
            *doc_tokens_hashmap.entry(term_id).or_insert(0) += 1;
        }
    }
}

#[inline]
fn process_doc_whitespace_transform(
    text: &str,
    vectorizer: &TfidfVectorizer<f32>,
    doc_tokens_hashmap: &mut FxHashMap<u32, u32>,
    token_buffer: &mut [u8; 4096],
    lowercase: bool,
    ngram_range: (usize, usize),
) {
    let ngram_width = 1 + (ngram_range.1 - ngram_range.0);
    let mut ngram_queue: FiFo = FiFo {
        data: [0; 8],
        capacity: ngram_width,
    };

    let mut buffer_idx: usize = 0;

    for &byte in text.as_bytes().iter() {
        match byte {
            0..=47 | 58..=64 | 91..=96 | 123..=126 => {
                if buffer_idx == 0 {
                    continue;
                }
                let token = std::str::from_utf8(&token_buffer[0..buffer_idx]).unwrap();

                if let Some(&term_id) = vectorizer.vocab.get(token) {
                    *doc_tokens_hashmap.entry(term_id).or_insert(0) += 1;
                    ngram_queue.push(term_id);
                }
                buffer_idx = 0;
                if ngram_range.1 > 1 {
                    add_ngrams_range_transform(
                        vectorizer,
                        doc_tokens_hashmap,
                        &ngram_queue,
                        ngram_range.0.max(2),
                        ngram_range.1,
                    );
                }
            }
            _ => {
                token_buffer[buffer_idx] = if lowercase {
                    byte.to_ascii_lowercase()
                } else {
                    byte
                };
                if byte < 128 {
                    buffer_idx += 1;
                }
            }
        }
    }
    // Process any token remaining in the buffer.
    if buffer_idx > 0 {
        let token = std::str::from_utf8(&token_buffer[0..buffer_idx]).unwrap();
        if let Some(&term_id) = vectorizer.vocab.get(token) {
            *doc_tokens_hashmap.entry(term_id).or_insert(0) += 1;
            ngram_queue.push(term_id);
        }
        if ngram_range.1 > 1 {
            add_ngrams_range_transform(
                vectorizer,
                doc_tokens_hashmap,
                &ngram_queue,
                ngram_range.0.max(2),
                ngram_range.1,
            );
        }
    }
}

#[inline]
fn process_doc_char_transform(
    text: &str,
    vectorizer: &TfidfVectorizer<f32>,
    doc_tokens_hashmap: &mut FxHashMap<u32, u32>,
    token_buffer: &mut [u8; 4096],
    lowercase: bool,
    ngram_range: (usize, usize),
) {
    let ngram_width = 1 + (ngram_range.1 - ngram_range.0);
    let mut ngram_queue: FiFo = FiFo {
        data: [0; 8],
        capacity: ngram_width,
    };

    let mut cntr: usize = ngram_range.0;
    let mut buffer_idx: usize = 0;

    for &byte in text.as_bytes().iter() {
        if cntr == 0 {
            // Time to “flush” the token.
            cntr = ngram_range.0;
            if buffer_idx > 0 {
                let token =
                    std::str::from_utf8(&token_buffer[0..buffer_idx]).unwrap();
                if let Some(&term_id) = vectorizer.vocab.get(token) {
                    *doc_tokens_hashmap.entry(term_id).or_insert(0) += 1;
                    ngram_queue.push(term_id);
                }
                buffer_idx = 0;
                if ngram_range.1 > 1 {
                    add_ngrams_range_transform(
                        vectorizer,
                        doc_tokens_hashmap,
                        &ngram_queue,
                        ngram_range.0.max(2),
                        ngram_range.1,
                    );
                }
            }
        }
        token_buffer[buffer_idx] = if lowercase {
            byte.to_ascii_lowercase()
        } else {
            byte
        };
        if byte < 128 {
            buffer_idx += 1;
            cntr -= 1;
        }
    }
    if buffer_idx > 0 {
        let token =
            std::str::from_utf8(&token_buffer[0..buffer_idx]).unwrap();
        if let Some(&term_id) = vectorizer.vocab.get(token) {
            *doc_tokens_hashmap.entry(term_id).or_insert(0) += 1;
            ngram_queue.push(term_id);
        }
        if ngram_range.1 > 1 {
            add_ngrams_range_transform(
                vectorizer,
                doc_tokens_hashmap,
                &ngram_queue,
                ngram_range.0.max(2),
                ngram_range.1,
            );
        }
    }
}

fn transform(
    text_series: &Series,
    vectorizer: &TfidfVectorizer<f32>,
    lowercase: bool,
    ngram_range: (usize, usize),
    whitespace_tokenization: bool,
) -> Result<CSRMatrix<f32>, PolarsError> {
    let training_count = vectorizer.csr_mat.row_start_pos.len() - 1;

    let mut csr_mat: CSRMatrix<f32> = CSRMatrix {
        values: Vec::new(),
        col_idxs: Vec::new(),
        row_start_pos: Vec::new(),
    };
    csr_mat.row_start_pos.push(0);

    const N: usize = 4096;
    let mut token_buffer: [u8; N] = [0; N];

    let mut doc_tokens_hashmap: FxHashMap<u32, u32> =
        FxHashMap::with_capacity_and_hasher(256, Default::default());

    for opt_text in text_series.utf8()?.into_iter() {
        doc_tokens_hashmap.clear();

        if let Some(text) = opt_text {
            if whitespace_tokenization {
                process_doc_whitespace_transform(
                    text,
                    vectorizer,
                    &mut doc_tokens_hashmap,
                    &mut token_buffer,
                    lowercase,
                    ngram_range,
                );
            } else {
                process_doc_char_transform(
                    text,
                    vectorizer,
                    &mut doc_tokens_hashmap,
                    &mut token_buffer,
                    lowercase,
                    ngram_range,
                );
            }
        }
        for (&term_id, &tf) in doc_tokens_hashmap.iter() {
            let df = vectorizer
                .dfs
                .get(term_id as usize)
                .copied()
                .unwrap_or(0);
            let idf = ((training_count as f32) / (1.0 + df as f32)).ln();
            csr_mat.values.push(tf as f32 * idf);
            csr_mat.col_idxs.push(term_id);
        }
        csr_mat.row_start_pos.push(csr_mat.values.len() as u64);
    }
    Ok(csr_mat)
}


#[cfg(test)]
mod tests {
    use polars::prelude::*;
    use super::*;

    #[test]
    fn test_build() -> Result<(), PolarsError> {
        const FILENAME: &str = "mb.parquet";
        const N_ROWS: u32 = std::u32::MAX;

        let lf = LazyFrame::scan_parquet(FILENAME, Default::default())?.limit(N_ROWS);

        let column = "title";
        let vectorizer = fit_transform(
            lf.clone().select([col(column)]).collect()?.column(column)?,
            true,
            (1, 1),
            None,
            Some(10_000),
            false,
            ).unwrap();

        eprintln!("Vocab size: {:?}K", vectorizer.vocab.num_tokens / 1000);

        Ok(())
    }
}
