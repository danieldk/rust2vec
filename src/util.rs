use std::cmp;
use std::io::BufRead;

use failure::Error;

pub struct NGrams<'a, T>
where
    T: 'a,
{
    max_n: usize,
    min_n: usize,
    seq: &'a [T],
    ngram: &'a [T],
}

impl<'a, T> NGrams<'a, T> {
    fn new(seq: &'a [T], min_n: usize, max_n: usize) -> Self {
        assert!(min_n <= max_n);

        let upper = cmp::min(max_n, seq.len());

        NGrams {
            min_n,
            max_n,
            seq,
            ngram: &seq[..upper],
        }
    }
}

impl<'a, T> Iterator for NGrams<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.ngram.len() < self.min_n {
            if self.seq.len() <= self.min_n {
                return None;
            }

            self.seq = &self.seq[1..];

            let upper = cmp::min(self.max_n, self.seq.len());
            self.ngram = &self.seq[..upper];
        }

        let ngram = self.ngram;

        self.ngram = &self.ngram[..self.ngram.len() - 1];

        Some(ngram)
    }
}

pub fn read_string(reader: &mut BufRead, delim: u8) -> Result<String, Error> {
    let mut buf = Vec::new();
    try!(reader.read_until(delim, &mut buf));
    buf.pop();
    Ok(try!(String::from_utf8(buf)))
}

#[cfg(test)]
mod tests {
    use super::NGrams;

    #[test]
    fn ngrams_test() {
        let hello_chars: Vec<_> = "hello world".chars().collect();
        let mut hello_check: Vec<&[char]> = vec![
            &['h'],
            &['h', 'e'],
            &['h', 'e', 'l'],
            &['e'],
            &['e', 'l'],
            &['e', 'l', 'l'],
            &['l'],
            &['l', 'l'],
            &['l', 'l', 'o'],
            &['l'],
            &['l', 'o'],
            &['l', 'o', ' '],
            &['o'],
            &['o', ' '],
            &['o', ' ', 'w'],
            &[' '],
            &[' ', 'w'],
            &[' ', 'w', 'o'],
            &['w'],
            &['w', 'o'],
            &['w', 'o', 'r'],
            &['o'],
            &['o', 'r'],
            &['o', 'r', 'l'],
            &['r'],
            &['r', 'l'],
            &['r', 'l', 'd'],
            &['l'],
            &['l', 'd'],
            &['d'],
        ];

        hello_check.sort();

        let mut hello_ngrams: Vec<_> = NGrams::new(&hello_chars, 1, 3).collect();
        hello_ngrams.sort();

        assert_eq!(hello_check, hello_ngrams);
    }

    #[test]
    fn ngrams_23_test() {
        let hello_chars: Vec<_> = "hello world".chars().collect();
        let mut hello_check: Vec<&[char]> = vec![
            &['h', 'e'],
            &['h', 'e', 'l'],
            &['e', 'l'],
            &['e', 'l', 'l'],
            &['l', 'l'],
            &['l', 'l', 'o'],
            &['l', 'o'],
            &['l', 'o', ' '],
            &['o', ' '],
            &['o', ' ', 'w'],
            &[' ', 'w'],
            &[' ', 'w', 'o'],
            &['w', 'o'],
            &['w', 'o', 'r'],
            &['o', 'r'],
            &['o', 'r', 'l'],
            &['r', 'l'],
            &['r', 'l', 'd'],
            &['l', 'd'],
        ];
        hello_check.sort();

        let mut hello_ngrams: Vec<_> = NGrams::new(&hello_chars, 2, 3).collect();
        hello_ngrams.sort();

        assert_eq!(hello_check, hello_ngrams);
    }

    #[test]
    fn empty_ngram_test() {
        let check: &[&[char]] = &[];
        assert_eq!(NGrams::<char>::new(&[], 1, 3).collect::<Vec<_>>(), check);
    }

}
