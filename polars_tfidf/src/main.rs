use std::collections::HashMap;
use polars::prelude::*;


#[allow(dead_code)]
#[inline]
fn get_lf_columns(lf: &LazyFrame) -> Result<Vec<String>, PolarsError> {
    Ok(lf.clone().collect_schema()?
        .iter()
        .map(|(x, _)| (x.clone().to_string()))
        .collect())
}

#[derive(Copy, Clone)]
struct TFToken {
    tf: u32,
    term_id: u32,
}

#[allow(dead_code)]
#[derive(Clone)]
struct TfidfVectorizer {
    vocab: HashMap<String, u32>,
    dfs: Vec<u32>,
    tokens: Vec<TFToken>,
}


fn fit_tfidf(lf: &LazyFrame, text_column: &str) -> Result<TfidfVectorizer, PolarsError> {
    let mut vectorizer = TfidfVectorizer {
        vocab: HashMap::new(),
        dfs: Vec::new(),
        tokens: Vec::new(),
    };

    let count = lf.clone().select([col(text_column).count()]).collect()?.height();
    vectorizer.dfs.reserve(count as usize);

    let mut doc_tokens: Vec<TFToken> = Vec::new();
    lf.clone().select([col(text_column)]).collect()?.column(text_column)?.str()?.into_iter().for_each(|v: Option<&str>| {
        let _v = match v {
            Some(v) => v,
            None => return,
        };
        doc_tokens.clear();
        for token in _v.split_whitespace() {
            if let Some(&term_id) = vectorizer.vocab.get(token) {
                match doc_tokens.iter_mut().find(|x| x.term_id == term_id) {
                    Some(item) => item.tf += 1,
                    None => {
                        vectorizer.dfs[term_id as usize] += 1;
                        vectorizer.tokens.push(TFToken{
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

        for token in doc_tokens.iter() {
            vectorizer.tokens.push(*token);
        }
    });
    Ok(vectorizer)
}

fn main() -> Result<(), PolarsError> {
    const FILENAME: &str = "mb.parquet";

    let lf = LazyFrame::scan_parquet(FILENAME, Default::default())?;
    // let columns = get_lf_columns(&lf.clone())?;

    let titles = lf.clone().select([col("title")]).collect()?;

    let start_time = std::time::Instant::now();
    _ = fit_tfidf(&lf, "title");
    println!("Time: {:?}", start_time.elapsed());
    println!("KDocs per second: {:?}", 0.001 * titles.height() as f32 / start_time.elapsed().as_secs_f32());

    println!("{:?}", lf.clone().collect());
    println!("{:?}", lf.clone().collect_schema()?);
    // println!("Columns: {:?}", &columns);
    // println!("Titles: {:?}", &titles);

    Ok(())
}
