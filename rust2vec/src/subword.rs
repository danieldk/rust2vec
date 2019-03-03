use std::cmp;
use std::hash::{Hash, Hasher};

use fnv::FnvHasher;

/// Iterator over n-grams in a sequence.
///
/// N-grams provides an iterator over the n-grams in a sentence between a
/// minimum and maximum length.
///
/// **Warning:** no guarantee is provided with regard to the iteration
/// order. The iterator only guarantees that all n-grams are produced.
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
    /// Create a new n-ngram iterator.
    ///
    /// The iterator will create n-ngrams of length *[min_n, max_n]*
    pub fn new(seq: &'a [T], min_n: usize, max_n: usize) -> Self {
        assert!(min_n != 0, "The minimum n-gram length cannot be zero.");
        assert!(
            min_n <= max_n,
            "The maximum length should be equal to or greater than the minimum length."
        );

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

/// Extension trait for computing subword indices.
///
/// Subword indexing assigns an identifier to each subword (n-gram) of a
/// string. A subword is indexed by computing its hash and then mapping
/// the hash to a bucket.
///
/// Since a non-perfect hash function is used, multiple subwords can
/// map to the same index.
pub trait SubwordIndices {
    /// Return the subword indices of the subwords of a string.
    ///
    /// The n-grams that are used are of length *[min_n, max_n]*, these are
    /// mapped to indices into *2^buckets_exp* buckets.
    ///
    /// The largest possible bucket exponent is 64.
    fn subword_indices(&self, min_n: usize, max_n: usize, buckets_exp: usize) -> Vec<(&str, u64)>;
}

impl SubwordIndices for str {
    fn subword_indices(&self, min_n: usize, max_n: usize, buckets_exp: usize) -> Vec<(&str, u64)> {
        assert!(
            buckets_exp <= 64,
            "The largest possible buckets exponent is 64."
        );

        let mask = if buckets_exp == 64 {
            !0
        } else {
            (1 << buckets_exp) - 1
        };

        let chars: Vec<_> = self.char_indices().collect();

        let mut indices = Vec::with_capacity((max_n - min_n + 1) * chars.len());
        for ngram in NGrams::new(&chars, min_n, max_n) {
            // Compute the index.
            let mut hasher = FnvHasher::default();
            ngram.len().hash(&mut hasher);
            for (_, ch) in ngram {
                ch.hash(&mut hasher);
            }
            let index = hasher.finish() & mask;

            // Get the string slice that the n-gram corresponds to.
            let lower = ngram.first().expect("Empty n-gram").0;
            let last = ngram.last().expect("Empty n-gram");
            let upper = last.0 + last.1.len_utf8();
            indices.push((&self[lower..upper], index));
        }

        indices
    }
}

#[cfg(test)]
mod tests {
    use lazy_static::lazy_static;
    use maplit::hashmap;
    use std::collections::HashMap;

    use super::{NGrams, SubwordIndices};

    #[test]
    fn ngrams_test() {
        let hello_chars: Vec<_> = "hellö world".chars().collect();
        let mut hello_check: Vec<&[char]> = vec![
            &['h'],
            &['h', 'e'],
            &['h', 'e', 'l'],
            &['e'],
            &['e', 'l'],
            &['e', 'l', 'l'],
            &['l'],
            &['l', 'l'],
            &['l', 'l', 'ö'],
            &['l'],
            &['l', 'ö'],
            &['l', 'ö', ' '],
            &['ö'],
            &['ö', ' '],
            &['ö', ' ', 'w'],
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

    #[test]
    #[should_panic]
    fn incorrect_min_n_test() {
        NGrams::<char>::new(&[], 0, 3);
    }

    #[test]
    #[should_panic]
    fn incorrect_max_n_test() {
        NGrams::<char>::new(&[], 2, 1);
    }

    lazy_static! {
        static ref SUBWORD_TESTS_2: HashMap<&'static str, Vec<(&'static str, u64)>> = hashmap! {
            "<Daniël>" =>
                vec![("<Da", 3), ("<Dan", 2), ("<Dani", 2), ("<Danië", 2), ("Dan", 1), ("Dani", 3), ("Danië", 1), ("Daniël", 2), ("ani", 0), ("anië", 0), ("aniël", 1), ("aniël>", 0), ("iël", 0), ("iël>", 1), ("nië", 2), ("niël", 1), ("niël>", 2), ("ël>", 3)],
            "<hallo>" =>
                vec![("<ha", 3), ("<hal", 0), ("<hall", 1), ("<hallo", 1), ("all", 3), ("allo", 3), ("allo>", 0), ("hal", 3), ("hall", 0), ("hallo", 2), ("hallo>", 3), ("llo", 1), ("llo>", 0), ("lo>", 3)],
        };
    }

    lazy_static! {
        static ref SUBWORD_TESTS_21: HashMap<&'static str, Vec<(&'static str, u64)>> = hashmap! {
            "<Daniël>" =>
                vec![("<Da", 2026735), ("<Dan", 1666166), ("<Dani", 2065822), ("<Danië", 1680294), ("Dan", 214157), ("Dani", 841219), ("Danië", 311961), ("Daniël", 1167494), ("ani", 1192256), ("anië", 741276), ("aniël", 1679745), ("aniël>", 1693100), ("iël", 233912), ("iël>", 488897), ("nië", 1644730), ("niël", 1489905), ("niël>", 620206), ("ël>", 1532271)],
            "<hallo>" =>
                vec![("<ha", 985859), ("<hal", 104120), ("<hall", 1505861), ("<hallo", 1321513), ("all", 938007), ("allo", 1163391), ("allo>", 599360), ("hal", 456131), ("hall", 1892376), ("hallo", 1006102), ("hallo>", 136555), ("llo", 722393), ("llo>", 1218704), ("lo>", 75867)],
        };
    }

    #[test]
    fn subword_indices_4_test() {
        // The goal of this test is to ensure that we are correctly bucketing
        // subwords. With a bucket exponent of 2, there are 2^2 = 4 buckets,
        // so we should see bucket numbers [0..3].

        for (word, indices_check) in SUBWORD_TESTS_2.iter() {
            let mut indices = word.subword_indices(3, 6, 2);
            indices.sort();
            assert_eq!(indices_check, &indices);
        }
    }

    #[test]
    fn subword_indices_2m_test() {
        // This test checks against precomputed bucket numbers. The goal of
        // if this test is to ensure that the subword_indices() method hashes
        // to the same buckets in the future.

        for (word, indices_check) in SUBWORD_TESTS_21.iter() {
            let mut indices = word.subword_indices(3, 6, 21);
            indices.sort();
            assert_eq!(indices_check, &indices);
        }
    }
}
