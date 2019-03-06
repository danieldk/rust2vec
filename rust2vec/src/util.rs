use std::mem::size_of;

use ndarray::ArrayViewMut1;

pub fn l2_normalize(mut v: ArrayViewMut1<f32>) -> f32 {
    let norm = v.dot(&v).sqrt();

    if norm != 0. {
        v /= norm;
    }

    norm
}

pub fn padding<T>(pos: u64) -> u64 {
    let size = size_of::<T>() as u64;
    size - (pos % size)
}
