use std::io::{Read, Seek, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use failure::{ensure, err_msg, Error};
use toml::Value;

use crate::io::private::{ChunkIdentifier, ReadChunk, WriteChunk};

#[derive(Clone, Debug, PartialEq)]
pub struct Metadata(pub Value);

impl ReadChunk for Metadata {
    fn read_chunk<R>(read: &mut R) -> Result<Self, Error>
    where
        R: Read + Seek,
    {
        let chunk_id = ChunkIdentifier::try_from(read.read_u32::<LittleEndian>()?)
            .ok_or_else(|| err_msg("Unknown chunk identifier"))?;
        ensure!(
            chunk_id == ChunkIdentifier::Metadata,
            "Cannot read chunk {:?} as Metadata",
            chunk_id
        );

        // Read chunk length.
        let chunk_len = read.read_u64::<LittleEndian>()? as usize;

        // Read TOML data.
        let mut buf = vec![0; chunk_len];
        read.read_exact(&mut buf)?;
        let buf_str = String::from_utf8(buf)?;

        Ok(Metadata(buf_str.parse::<Value>()?))
    }
}

impl WriteChunk for Metadata {
    fn chunk_identifier(&self) -> ChunkIdentifier {
        ChunkIdentifier::Metadata
    }

    fn write_chunk<W>(&self, write: &mut W) -> Result<(), Error>
    where
        W: Write + Seek,
    {
        let metadata_str = self.0.to_string();

        write.write_u32::<LittleEndian>(self.chunk_identifier() as u32)?;
        write.write_u64::<LittleEndian>(metadata_str.len() as u64)?;
        write.write_all(metadata_str.as_bytes())?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Read, Seek, SeekFrom};

    use byteorder::{LittleEndian, ReadBytesExt};
    use toml::{toml, toml_internal};

    use super::Metadata;
    use crate::io::private::{ReadChunk, WriteChunk};

    fn read_chunk_size(read: &mut impl Read) -> u64 {
        // Skip identifier.
        read.read_u32::<LittleEndian>().unwrap();

        // Return chunk length.
        read.read_u64::<LittleEndian>().unwrap()
    }

    fn test_metadata() -> Metadata {
        Metadata(toml! {
            [hyperparameters]
            dims = 300
            ns = 5

            [description]
            description = "Test model"
            language = "de"
        })
    }

    #[test]
    fn metadata_correct_chunk_size() {
        let check_metadata = test_metadata();
        let mut cursor = Cursor::new(Vec::new());
        check_metadata.write_chunk(&mut cursor).unwrap();
        cursor.seek(SeekFrom::Start(0)).unwrap();

        let chunk_size = read_chunk_size(&mut cursor);
        assert_eq!(
            cursor.read_to_end(&mut Vec::new()).unwrap(),
            chunk_size as usize
        );
    }

    #[test]
    fn metadata_write_read_roundtrip() {
        let check_metadata = test_metadata();
        let mut cursor = Cursor::new(Vec::new());
        check_metadata.write_chunk(&mut cursor).unwrap();
        cursor.seek(SeekFrom::Start(0)).unwrap();
        let metadata = Metadata::read_chunk(&mut cursor).unwrap();
        assert_eq!(metadata, check_metadata);
    }
}
