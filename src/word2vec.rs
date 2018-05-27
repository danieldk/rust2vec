use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::mem;
use std::slice::from_raw_parts_mut;

use byteorder::{LittleEndian, WriteBytesExt};
use failure::{err_msg, Error};
use ndarray::{Array2, Axis};

use super::*;

/// Method to construct `Embeddings` from a word2vec binary file.
///
/// This trait defines an extension to `Embeddings` to read the word embeddings
/// from a file in word2vec binary format.
pub trait ReadWord2Vec<R>
where
    R: BufRead,
{
    /// Read the embeddings from the given buffered reader.
    fn read_word2vec_binary(reader: &mut R) -> Result<Embeddings, Error>;
}

impl<R> ReadWord2Vec<R> for Embeddings
where
    R: BufRead,
{
    fn read_word2vec_binary(reader: &mut R) -> Result<Embeddings, Error> {
        let n_words = try!(read_number(reader, b' '));
        let embed_len = try!(read_number(reader, b'\n'));

        let mut matrix = Array2::zeros((n_words, embed_len));
        let mut indices = HashMap::new();
        let mut words = Vec::with_capacity(n_words);

        for idx in 0..n_words {
            let word = try!(util::read_string(reader, ' ' as u8));
            let word = word.trim();
            words.push(word.to_owned());
            indices.insert(word.to_owned(), idx);

            let mut embedding = matrix.subview_mut(Axis(0), idx);

            {
                let mut embedding_raw = match embedding.as_slice_mut() {
                    Some(s) => unsafe { typed_to_bytes(s) },
                    None => return Err(err_msg("Matrix not contiguous")),
                };
                try!(reader.read_exact(&mut embedding_raw));
            }
        }

        Ok(super::embeddings::new_embeddings(
            matrix, embed_len, indices, words,
        ))
    }
}

fn read_number(reader: &mut BufRead, delim: u8) -> Result<usize, Error> {
    let field_str = try!(util::read_string(reader, delim));
    Ok(try!(field_str.parse()))
}

unsafe fn typed_to_bytes<T>(slice: &mut [T]) -> &mut [u8] {
    from_raw_parts_mut(
        slice.as_mut_ptr() as *mut u8,
        slice.len() * mem::size_of::<T>(),
    )
}

/// Method to write `Embeddings` to a word2vec binary file.
///
/// This trait defines an extension to `Embeddings` to write the word embeddings
/// to a file in word2vec binary format.
pub trait WriteWord2Vec<W>
where
    W: Write,
{
    /// Write the embeddings from the given writer.
    fn write_word2vec_binary(&self, w: &mut W) -> Result<(), Error>;
}

impl<W> WriteWord2Vec<W> for Embeddings
where
    W: Write,
{
    fn write_word2vec_binary(&self, w: &mut W) -> Result<(), Error>
    where
        W: Write,
    {
        write!(w, "{} {}\n", self.len(), self.embed_len())?;

        for (word, embed) in self.iter() {
            write!(w, "{} ", word)?;

            // Write embedding to a vector with little-endian encoding.
            for v in embed {
                w.write_f32::<LittleEndian>(*v)?;
            }

            w.write(&[0x0a])?;
        }

        Ok(())
    }
}
