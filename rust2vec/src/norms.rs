use std::io::{Read, Seek, SeekFrom, Write};
use std::mem::size_of;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use failure::{ensure, format_err, Error};
use ndarray::Array1;

use crate::io::private::{ChunkIdentifier, ReadChunk, TypeId, WriteChunk};
use crate::util::padding;

#[derive(Debug, PartialEq)]
pub struct Norms(pub Array1<f32>);

impl WriteChunk for Norms {
    fn chunk_identifier(&self) -> ChunkIdentifier {
        ChunkIdentifier::Norms
    }

    fn write_chunk<W>(&self, write: &mut W) -> Result<(), Error>
    where
        W: Write + Seek,
    {
        write.write_u32::<LittleEndian>(ChunkIdentifier::Norms as u32)?;
        let n_padding = padding::<f32>(write.seek(SeekFrom::Current(0))?);

        // Chunk size: type id (u32) + padding [0, 4) bytes, len (u64), norms.
        let chunk_len = size_of::<u32>()
            + size_of::<u64>()
            + n_padding as usize
            + self.0.len() * size_of::<f32>();
        write.write_u64::<LittleEndian>(chunk_len as u64)?;
        write.write_u64::<LittleEndian>(self.0.len() as u64)?;
        write.write_u32::<LittleEndian>(f32::type_id())?;

        let padding = vec![0; n_padding as usize];
        write.write_all(&padding)?;

        for elem in self.0.iter() {
            write.write_f32::<LittleEndian>(*elem)?;
        }

        Ok(())
    }
}

impl ReadChunk for Norms {
    fn read_chunk<R>(read: &mut R) -> Result<Self, Error>
    where
        R: Read + Seek,
    {
        let chunk_id = read.read_u32::<LittleEndian>()?;
        let chunk_id = ChunkIdentifier::try_from(chunk_id)
            .ok_or_else(|| format_err!("Unknown chunk identifier: {}", chunk_id))?;
        ensure!(
            chunk_id == ChunkIdentifier::Norms,
            "Cannot read chunk {:?} as Norms",
            chunk_id
        );

        // Read and discard chunk length.
        read.read_u64::<LittleEndian>()?;

        let len = read.read_u64::<LittleEndian>()? as usize;

        ensure!(
            read.read_u32::<LittleEndian>()? == f32::type_id(),
            "Expected single precision floating point matrix for Norms."
        );

        let n_padding = padding::<f32>(read.seek(SeekFrom::Current(0))?);
        read.seek(SeekFrom::Current(n_padding as i64))?;

        let mut data = vec![0f32; len];
        read.read_f32_into::<LittleEndian>(&mut data)?;

        Ok(Norms(Array1::from_vec(data)))
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Read, Seek, SeekFrom};

    use byteorder::{LittleEndian, ReadBytesExt};
    use ndarray::Array1;

    use crate::io::private::{ReadChunk, WriteChunk};
    use crate::norms::Norms;

    const N_NORMS: usize = 100;

    fn test_norms() -> Norms {
        let test_data = Array1::range(0f32, N_NORMS as f32, 1f32);
        Norms(test_data)
    }

    fn read_chunk_size(read: &mut impl Read) -> u64 {
        // Skip identifier.
        read.read_u32::<LittleEndian>().unwrap();

        // Return chunk length.
        read.read_u64::<LittleEndian>().unwrap()
    }

    #[test]
    fn norms_correct_chunk_size() {
        let check_norms = test_norms();
        let mut cursor = Cursor::new(Vec::new());
        check_norms.write_chunk(&mut cursor).unwrap();
        cursor.seek(SeekFrom::Start(0)).unwrap();

        let chunk_size = read_chunk_size(&mut cursor);
        assert_eq!(
            cursor.read_to_end(&mut Vec::new()).unwrap(),
            chunk_size as usize
        );
    }

    #[test]
    fn norms_write_read_roundtrip() {
        let check_norms = test_norms();
        let mut cursor = Cursor::new(Vec::new());
        check_norms.write_chunk(&mut cursor).unwrap();
        cursor.seek(SeekFrom::Start(0)).unwrap();
        let norms = Norms::read_chunk(&mut cursor).unwrap();
        assert_eq!(norms, check_norms);
    }
}
