use crate::subword::{NGrams, SubwordIndices};

pub struct FastTextStr<'a>(&'a str);

impl<'a> SubwordIndices for FastTextStr<'a> {
    fn subword_indices(&self, min_n: usize, max_n: usize, n_buckets: usize) -> Vec<u64> {
        let chars: Vec<_> = self.0.chars().collect();

        NGrams::new(&chars, min_n, max_n)
            .map(|ngram| fasttext_hash(ngram) as u64 % n_buckets as u64)
            .collect()
    }
}

fn fasttext_hash(chars: &[char]) -> u32 {
    let mut h = 2166136261;

    for ch in chars {
        let mut bytes = [0; 4];
        for byte in ch.encode_utf8(&mut bytes).bytes() {
            h = h ^ (byte as i8) as u32;
            h = h.wrapping_mul(16777619);
        }
    }

    h
}

#[cfg(test)]
mod tests {
    use crate::subword::SubwordIndices;

    use super::FastTextStr;

    const FASTTEXT_BUCKETS: usize = 2000000;

    #[test]
    fn subword_indices() {
        let mut indices = FastTextStr("<groß>").subword_indices(3, 6, FASTTEXT_BUCKETS);
        indices.sort_unstable();
        assert_eq!(
            indices,
            &[112809, 205234, 309171, 478071, 606866, 838837, 968056, 1490375, 1924105, 1982818]
        );

        let mut indices =
            FastTextStr("<ノドアカハチドリ体長>").subword_indices(3, 6, FASTTEXT_BUCKETS);
        indices.sort_unstable();
        assert_eq!(
            indices,
            vec![
                42198, 101645, 189821, 215859, 287873, 289058, 296136, 399372, 502758, 510842,
                542013, 576040, 727354, 784682, 871282, 924427, 1131669, 1157786, 1159974, 1269047,
                1290342, 1338620, 1377657, 1487435, 1496901, 1557890, 1589316, 1597651, 1715065,
                1718303, 1850912, 1863663, 1920554, 1954325
            ]
        );
    }
}
