mod embeddings;
pub use crate::embeddings::{Embeddings, Iter};

pub mod fasttext;

pub mod similarity;

pub mod subword;

pub mod text;

pub mod vocab;

pub mod word2vec;

#[cfg(test)]
mod tests;
