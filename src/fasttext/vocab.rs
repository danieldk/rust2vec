use std::collections::HashMap;

use crate::fasttext::subword::FastTextStr;
use crate::subword::SubwordIndices;
use crate::vocab::{Indices, Vocab};

static EOS: &str = "</s>";
const BOW: char = '<';
const EOW: char = '>';

pub struct FastTextVocab {
    words: Vec<String>,
    indices: HashMap<String, usize>,
    min_n: usize,
    max_n: usize,
    n_buckets: usize,
}

impl Vocab for FastTextVocab {
    fn word_indices(&self, word: impl AsRef<str>) -> Option<Indices> {
        if let Some(&idx) = self.indices.get(word.as_ref()) {
            return Some(Indices::SingleIndex(idx));
        }

        let mut bracketed_word = String::new();
        bracketed_word.push(BOW);
        bracketed_word.push_str(word.as_ref());
        bracketed_word.push(EOW);

        let indices = FastTextStr(bracketed_word)
            .subword_indices(self.min_n, self.max_n, self.n_buckets)
            .into_iter()
            .map(|v| v as usize + self.words.len())
            .collect();

        Some(Indices::MultiIndex(indices))
    }

    fn words(&self) -> &[String] {
        &self.words
    }
}
