use std::io::BufRead;

use failure::{err_msg, Error};

pub fn read_string(reader: &mut BufRead, delim: u8) -> Result<String, Error> {
    let mut buf = Vec::new();
    try!(reader.read_until(delim, &mut buf));
    buf.pop();
    Ok(try!(String::from_utf8(buf)))
}
