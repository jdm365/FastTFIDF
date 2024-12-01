use rustc_hash::FxHashMap;
use polars::prelude::*;


struct FiFo {
    data: [u32; 8],
    capacity: usize,
}

impl FiFo {
    fn push(&mut self, val: u32) {
        self.data.rotate_left(1);
        self.data[self.capacity] = val;
    }
}


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
    num_tokens: usize,
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

    #[inline]
    fn get_ngram(&self, key: NGramKey) -> Option<&u32> {
        match key {
            NGramKey::Bigram(key) => {
                self.higher_grams[0].get(&NGramKey::Bigram(key))
            },
            NGramKey::Trigram(key) => {
                self.higher_grams[1].get(&NGramKey::Trigram(key))
            },
            NGramKey::Quadgram(key) => {
                self.higher_grams[2].get(&NGramKey::Quadgram(key))
            },
            NGramKey::Pentagram(key) => {
                self.higher_grams[3].get(&NGramKey::Pentagram(key))
            },
            NGramKey::Hexagram(key) => {
                self.higher_grams[4].get(&NGramKey::Hexagram(key))
            },
            NGramKey::Heptagram(key) => {
                self.higher_grams[5].get(&NGramKey::Heptagram(key))
            },
            NGramKey::Octagram(key) => {
                self.higher_grams[6].get(&NGramKey::Octagram(key))
            },
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
fn add_all_ngrams(
    vectorizer: &mut TfidfVectorizer<f32>,
    doc_tokens_hashmap: &mut FxHashMap<u32, u32>,
    ngram_queue: &FiFo,
    ngram_width: usize,
) {
    for i in 0..ngram_width {
        let ngram_size = i + 2;

        for j in 0..(ngram_width - i) {
            if let Some(&term_id) = match ngram_size {
                2 => {
                    let key = NGramKey::Bigram(ngram_queue.data[j..j+1].try_into().unwrap());
                    vectorizer.vocab.get_ngram(key)
                },
                3 => {
                    let key = NGramKey::Trigram(ngram_queue.data[j..j+2].try_into().unwrap());
                    vectorizer.vocab.get_ngram(key)
                },
                4 => {
                    let key = NGramKey::Quadgram(ngram_queue.data[j..j+3].try_into().unwrap());
                    vectorizer.vocab.get_ngram(key)
                },
                5 => {
                    let key = NGramKey::Pentagram(ngram_queue.data[j..j+4].try_into().unwrap());
                    vectorizer.vocab.get_ngram(key)
                },
                6 => {
                    let key = NGramKey::Hexagram(ngram_queue.data[j..j+5].try_into().unwrap());
                    vectorizer.vocab.get_ngram(key)
                },
                7 => {
                    let key = NGramKey::Heptagram(ngram_queue.data[j..j+6].try_into().unwrap());
                    vectorizer.vocab.get_ngram(key)
                },
                8 => {
                    let key = NGramKey::Octagram(ngram_queue.data[j..j+7].try_into().unwrap());
                    vectorizer.vocab.get_ngram(key)
                },
                _ => panic!("Invalid ngram size."),
            } {
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

                match ngram_size {
                    2 => vectorizer.vocab.insert_ngram(NGramKey::Bigram(ngram_queue.data[j..j+1].try_into().unwrap()), term_id),
                    3 => vectorizer.vocab.insert_ngram(NGramKey::Trigram(ngram_queue.data[j..j+2].try_into().unwrap()), term_id),
                    4 => vectorizer.vocab.insert_ngram(NGramKey::Quadgram(ngram_queue.data[j..j+3].try_into().unwrap()), term_id),
                    5 => vectorizer.vocab.insert_ngram(NGramKey::Pentagram(ngram_queue.data[j..j+4].try_into().unwrap()), term_id),
                    6 => vectorizer.vocab.insert_ngram(NGramKey::Hexagram(ngram_queue.data[j..j+5].try_into().unwrap()), term_id),
                    7 => vectorizer.vocab.insert_ngram(NGramKey::Heptagram(ngram_queue.data[j..j+6].try_into().unwrap()), term_id),
                    8 => vectorizer.vocab.insert_ngram(NGramKey::Octagram(ngram_queue.data[j..j+7].try_into().unwrap()), term_id),
                    _ => panic!("Invalid ngram size."),
                }
                doc_tokens_hashmap.insert(term_id, 1);
                vectorizer.dfs.push(1);

                vectorizer.vocab.num_tokens += 1;
            }
        }
    }
}

#[inline]
fn add_last_ngrams(
    vectorizer: &mut TfidfVectorizer<f32>,
    doc_tokens_hashmap: &mut FxHashMap<u32, u32>,
    ngram_queue: &FiFo,
    ngram_width: usize,
) {
    for i in 0..ngram_width {
        let ngram_size = i + 2;

        for j in 0..(ngram_width - i) {
            if let Some(&term_id) = match ngram_size {
                2 => {
                    let key = NGramKey::Bigram(ngram_queue.data[j..j+1].try_into().unwrap());
                    vectorizer.vocab.get_ngram(key)
                },
                3 => {
                    let key = NGramKey::Trigram(ngram_queue.data[j..j+2].try_into().unwrap());
                    vectorizer.vocab.get_ngram(key)
                },
                4 => {
                    let key = NGramKey::Quadgram(ngram_queue.data[j..j+3].try_into().unwrap());
                    vectorizer.vocab.get_ngram(key)
                },
                5 => {
                    let key = NGramKey::Pentagram(ngram_queue.data[j..j+4].try_into().unwrap());
                    vectorizer.vocab.get_ngram(key)
                },
                6 => {
                    let key = NGramKey::Hexagram(ngram_queue.data[j..j+5].try_into().unwrap());
                    vectorizer.vocab.get_ngram(key)
                },
                7 => {
                    let key = NGramKey::Heptagram(ngram_queue.data[j..j+6].try_into().unwrap());
                    vectorizer.vocab.get_ngram(key)
                },
                8 => {
                    let key = NGramKey::Octagram(ngram_queue.data[j..j+7].try_into().unwrap());
                    vectorizer.vocab.get_ngram(key)
                },
                _ => panic!("Invalid ngram size."),
            } {
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

                match ngram_size {
                    2 => vectorizer.vocab.insert_ngram(NGramKey::Bigram(ngram_queue.data[j..j+1].try_into().unwrap()), term_id),
                    3 => vectorizer.vocab.insert_ngram(NGramKey::Trigram(ngram_queue.data[j..j+2].try_into().unwrap()), term_id),
                    4 => vectorizer.vocab.insert_ngram(NGramKey::Quadgram(ngram_queue.data[j..j+3].try_into().unwrap()), term_id),
                    5 => vectorizer.vocab.insert_ngram(NGramKey::Pentagram(ngram_queue.data[j..j+4].try_into().unwrap()), term_id),
                    6 => vectorizer.vocab.insert_ngram(NGramKey::Hexagram(ngram_queue.data[j..j+5].try_into().unwrap()), term_id),
                    7 => vectorizer.vocab.insert_ngram(NGramKey::Heptagram(ngram_queue.data[j..j+6].try_into().unwrap()), term_id),
                    8 => vectorizer.vocab.insert_ngram(NGramKey::Octagram(ngram_queue.data[j..j+7].try_into().unwrap()), term_id),
                    _ => panic!("Invalid ngram size."),
                }
                doc_tokens_hashmap.insert(term_id, 1);
                vectorizer.dfs.push(1);

                vectorizer.vocab.num_tokens += 1;
            }
        }
    }
}

#[allow(dead_code)]
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

    let ngram_width = ngram_range.1 - ngram_range.0;
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

    let mut idx = 0;
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

                if idx == 0 {
                    add_all_ngrams(vectorizer, doc_tokens_hashmap, &ngram_queue, ngram_width);
                } else {
                    add_last_ngrams(vectorizer, doc_tokens_hashmap, &ngram_queue, ngram_width);
                }
                idx += 1;
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
fn process_doc_whitespace_hashmap(
    text: &str,
    vectorizer: &mut TfidfVectorizer<f32>,
    doc_tokens_hashmap: &mut FxHashMap<u32, u32>,
    token_buffer: &mut [u8; 4096],
    lowercase: bool,
    _ngram_range: (usize, usize),
    max_df: usize,
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
                            if vectorizer.dfs[term_id as usize] <= max_df as u32 {
                                doc_tokens_hashmap.insert(term_id, 1);
                            }
                        },
                    }
                } else {
                    vectorizer.vocab.insert(str_ref.to_string(), vectorizer.vocab.num_tokens as u32);
                    doc_tokens_hashmap.insert(vectorizer.vocab.num_tokens as u32, 1);
                    vectorizer.dfs.push(1);

                    vectorizer.vocab.num_tokens += 1;
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
    _ngram_range: (usize, usize),
    max_df: usize,
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
                            if vectorizer.dfs[term_id as usize] <= max_df as u32 {
                                doc_tokens.push(TFToken{
                                    tf: 1,
                                    term_id,
                                });
                            }
                        },
                    }
                } else {
                    vectorizer.vocab.insert(str_ref.to_string(), vectorizer.vocab.num_tokens as u32);
                    doc_tokens.push(TFToken{
                        tf: 1,
                        term_id: vectorizer.vocab.num_tokens as u32,
                    });
                    vectorizer.dfs.push(1);

                    vectorizer.vocab.num_tokens += 1;
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

    for token in doc_tokens.iter() {
        vectorizer.csr_mat.values.push(token.tf as f32);
        vectorizer.csr_mat.col_idxs.push(token.term_id);
    }

    Ok(())
}


fn fit_transform(
    text_series: &Column,
    lowercase: bool,
    ngram_range: (usize, usize),
    _min_df: Option<usize>,
    _max_df: Option<usize>,
    ) -> Result<TfidfVectorizer<f32>, PolarsError> {
    let start_time = std::time::Instant::now();

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
    let mut doc_tokens: Vec<TFToken> = Vec::new();
    let mut doc_tokens_hashmap: FxHashMap<u32, u32> = FxHashMap::with_capacity_and_hasher(256, Default::default());

    text_series.str()?.into_iter().for_each(|v: Option<&str>| {
        let _v: &str = match v {
            Some(v) => v,
            None => { idx += 1; return; },
        };

        if _v.len() > 256 {
            // process_doc_whitespace_hashmap(
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
            process_doc_whitespace(
                _v,
                &mut vectorizer,
                &mut doc_tokens,
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

    let mut last_idx: u32 = 0;
    for (idx, v) in vectorizer.csr_mat.col_idxs.iter().enumerate() {
        if *v < last_idx {
            last_idx = *v;
            vectorizer.csr_mat.row_start_pos.push(idx as u64);
        }
    }

    // Add the last row. Necessary for csr indptr format.
    vectorizer.csr_mat.row_start_pos.push(vectorizer.csr_mat.values.len() as u64);

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
        (1, 2),
        None,
        // None,
        Some(10_000),
        );

    Ok(())
}
