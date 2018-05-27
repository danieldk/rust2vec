extern crate byteorder;

#[macro_use]
extern crate failure;

extern crate itertools;

extern crate ndarray;

macro_rules! try_opt {
    ($expr:expr) => {
        match $expr {
            Some(val) => val,
            None => return None,
        }
    };
}

mod embeddings;
pub use embeddings::{Builder, BuilderError, Embeddings, Iter, WordSimilarity};

pub mod fasttext;

mod text;
pub use text::{ReadText, WriteText};

pub(crate) mod util;

mod word2vec;
pub use word2vec::{ReadWord2Vec, WriteWord2Vec};

#[cfg(test)]
mod tests;
