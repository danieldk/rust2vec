use std::io::BufRead;

use byteorder::{LittleEndian, ReadBytesExt};
use failure::{err_msg, Error};

use super::*;

const FASTTEXT_FILEFORMAT_MAGIC: u32 = 793712314;
const FASTTEXT_VERSION: u32 = 12;

static EOS: &str = "</s>";
const BOW: char = '<';
const EOW: char = '>';

#[derive(Clone, Debug)]
pub enum Model {
    CBOW,
    SkipGram,
    Supervised,
}

impl Model {
    fn read_binary<R>(reader: &mut R) -> Result<Model, Error>
    where
        R: BufRead,
    {
        let model = reader.read_u32::<LittleEndian>()?;

        use self::Model::*;
        match model {
            1 => Ok(CBOW),
            2 => Ok(SkipGram),
            3 => Ok(Supervised),
            m => Err(err_msg(format!("Unknown model: {}", m))),
        }
    }
}

#[derive(Clone, Debug)]
pub enum Loss {
    HierarchicalSoftmax,
    NegativeSampling,
    Softmax,
}

impl Loss {
    fn read_binary<R>(reader: &mut R) -> Result<Loss, Error>
    where
        R: BufRead,
    {
        let loss = reader.read_u32::<LittleEndian>()?;

        use self::Loss::*;
        match loss {
            1 => Ok(HierarchicalSoftmax),
            2 => Ok(NegativeSampling),
            3 => Ok(Softmax),
            l => Err(err_msg(format!("Unknown loss: {}", l))),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Args {
    dims: u32,
    window_size: u32,
    epoch: u32,
    min_count: u32,
    neg: u32,
    word_ngrams: u32,
    loss: Loss,
    model: Model,
    bucket: u32,
    min_n: u32,
    max_n: u32,
    lr_update_rate: u32,
    sampling_threshold: f64,
}

impl Args {
    fn read_binary<R>(reader: &mut R) -> Result<Args, Error>
    where
        R: BufRead,
    {
        let dims = reader.read_u32::<LittleEndian>()?;
        let window_size = reader.read_u32::<LittleEndian>()?;
        let epoch = reader.read_u32::<LittleEndian>()?;
        let min_count = reader.read_u32::<LittleEndian>()?;
        let neg = reader.read_u32::<LittleEndian>()?;
        let word_ngrams = reader.read_u32::<LittleEndian>()?;
        let loss = Loss::read_binary(reader)?;
        let model = Model::read_binary(reader)?;
        let bucket = reader.read_u32::<LittleEndian>()?;
        let min_n = reader.read_u32::<LittleEndian>()?;
        let max_n = reader.read_u32::<LittleEndian>()?;
        let lr_update_rate = reader.read_u32::<LittleEndian>()?;
        let sampling_threshold = reader.read_f64::<LittleEndian>()?;

        Ok(Args {
            dims,
            window_size,
            epoch,
            min_count,
            neg,
            word_ngrams,
            loss,
            model,
            bucket,
            min_n,
            max_n,
            lr_update_rate,
            sampling_threshold,
        })
    }
}

pub enum EntryType {
    Word,
    Label,
}

impl EntryType {
    fn read_binary<R>(reader: &mut R) -> Result<Self, Error>
    where
        R: BufRead,
    {
        let model = reader.read_u8()?;

        use self::EntryType::*;
        match model {
            0 => Ok(Word),
            1 => Ok(Label),
            t => Err(err_msg(format!("Unknown entry type: {}", t))),
        }
    }
}

pub struct Entry {
    word: String,
    count: u64,
    entry_type: EntryType,
    subwords: Vec<u32>,
}

pub struct Dictionary {
    args: Args,
    words: Vec<Entry>,
}

impl Dictionary {
    fn read_binary<R>(args: Args, reader: &mut R) -> Result<Dictionary, Error>
    where
        R: BufRead,
    {
        let size = reader.read_u32::<LittleEndian>()?;
        let n_words = reader.read_u32::<LittleEndian>()?;
        let n_labels = reader.read_u32::<LittleEndian>()?;
        let n_tokens = reader.read_u64::<LittleEndian>()?;
        let prune_idx_size = reader.read_i64::<LittleEndian>()?;

        let mut words = Vec::with_capacity(size as usize);
        for i in 0..size {
            let word = util::read_string(reader, 0)?;
            let count = reader.read_u64::<LittleEndian>()?;
            let entry_type = EntryType::read_binary(reader)?;

            words.push(Entry {
                word,
                count,
                entry_type,
                subwords: Vec::new(),
            })
        }

        if prune_idx_size >= 0 {
            unimplemented!();
        }

        for i in 0..prune_idx_size {
            let _first = reader.read_u32::<LittleEndian>()?;
            let _second = reader.read_u32::<LittleEndian>()?;
        }

        eprintln!(
            "size: {}, n_words: {}, n_labels: {}, n_tokens: {}, pr_idx_size: {}",
            size, n_words, n_labels, n_tokens, prune_idx_size
        );

        let mut dict = Dictionary { args, words };
        dict.init_ngrams();

        Ok(dict)
    }

    fn init_ngrams(&mut self) {
        let n_words = self.words.len() as u32;

        for word in &mut self.words {
            if word.word == EOS {
                continue;
            }

            let mut bounded_word = String::new();
            bounded_word.push(BOW);
            bounded_word.push_str(&word.word);
            bounded_word.push(EOW);

            let subwords = compute_subwords(
                &bounded_word,
                self.args.min_n,
                self.args.max_n,
                n_words,
                self.args.bucket,
            );
        }
    }
}

pub struct Matrix {
    data: Vec<f32>,
}

impl Matrix {
    fn read_binary<R>(reader: &mut R) -> Result<Self, Error>
    where
        R: BufRead,
    {
        let m = reader.read_u64::<LittleEndian>()?;
        let n = reader.read_u64::<LittleEndian>()?;
        eprintln!("m: {}, n: {}", m, n);

        let mut data = vec![0.0; (m * n) as usize];
        reader.read_f32_into::<LittleEndian>(&mut data)?;

        Ok(Matrix { data })
    }
}

pub trait ReadFastText<R>
where
    R: BufRead,
{
    fn read_fasttext_binary(reader: &mut R) -> Result<Embeddings, Error>;
}

impl<R> ReadFastText<R> for Embeddings
where
    R: BufRead,
{
    fn read_fasttext_binary(reader: &mut R) -> Result<Embeddings, Error> {
        let magic = reader.read_u32::<LittleEndian>()?;
        if magic != FASTTEXT_FILEFORMAT_MAGIC {
            return Err(err_msg("Incorrect file format"));
        }

        let version = reader.read_u32::<LittleEndian>()?;
        if version > FASTTEXT_VERSION {
            return Err(err_msg(format!(
                "FastText file version unsupported: {} > {}",
                version, FASTTEXT_VERSION
            )));
        }

        let args = Args::read_binary(reader)?;
        let dict = Dictionary::read_binary(args.clone(), reader)?;

        let quant_input = reader.read_u8()?;
        if quant_input == 1 {
            unimplemented!();
        }

        let matrix = Matrix::read_binary(reader)?;

        let qout = reader.read_u8()?;
        if qout == 1 {
            unimplemented!();
        }

        eprintln!("args: {:?}", args);

        unimplemented!();
    }
}

fn compute_subwords(word: &str, min_n: u32, max_n: u32, n_words: u32, buckets: u32) -> Vec<u32> {
    assert!(min_n >= max_n);

    let chars: Vec<_> = word.chars().collect();

    let mut hashes = Vec::new();

    for ngram in util::NGrams::new(&chars, min_n as usize, max_n as usize) {
        let h = fasttext_hash(ngram) % buckets;
        hashes.push(n_words + h);
    }

    hashes
}

fn fasttext_hash(chars: &[char]) -> u32 {
    let mut h = 2166136261;

    for ch in chars {
        let mut bytes = [0; 4];
        for byte in ch.encode_utf8(&mut bytes).bytes() {
            h = h ^ byte as u32;
            h = h.wrapping_mul(16777619);
        }
    }

    h
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::BufReader;

    use super::ReadFastText;
    use Embeddings;

    #[test]
    fn read_args_test() {
        let f = File::open("/home/daniel/Downloads/cc.nl.300.bin").unwrap();
        let mut reader = BufReader::new(f);
        Embeddings::read_fasttext_binary(&mut reader).unwrap();
    }
}
