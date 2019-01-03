use std::collections::HashMap;

pub enum Indices {
    SingleIndex(usize),
    MultiIndex(Vec<usize>),
}

pub trait Vocab {
    /// Get the vocabulary indices for `word`.
    fn word_indices(&self, word: impl AsRef<str>) -> Option<Indices>;

    /// Return vocabulary words.
    fn words(&self) -> &[String];
}

/// Vocabulary consisting of words.
pub struct WordVocab {
    words: Vec<String>,
    indices: HashMap<String, usize>,
}

impl WordVocab {
    pub(crate) fn new(
        words: impl Into<Vec<String>>,
        indices: impl Into<HashMap<String, usize>>,
    ) -> Self {
        WordVocab {
            words: words.into(),
            indices: indices.into(),
        }
    }
}

impl Vocab for WordVocab {
    fn word_indices(&self, word: impl AsRef<str>) -> Option<Indices> {
        self.indices
            .get(word.as_ref())
            .cloned()
            .map(Indices::SingleIndex)
    }

    fn words(&self) -> &[String] {
        &self.words
    }
}
