//! scans for a local header, returns false if there is no more local file header.
//! The gzip file format seems to think that there can be multiple files in a gzip file
//! but none
bool CZipReader::scanGZipHeader()
{
	SZipFileEntry entry;
	entry.Offset = 0;
	memset(&entry.header, 0, sizeof(SZIPFileHeader));

	// read header
	SGZIPMemberHeader header;
	if (File->read(&header, sizeof(SGZIPMemberHeader)) == sizeof(SGZIPMemberHeader))
	{
		// check header value
		if (header.sig != 0x8b1f)
			return false;

		// now get the file info
		if (header.flags & EGZF_EXTRA_FIELDS)
		{
			// read lenth of extra data
			uint16_t dataLen;

			File->read(&dataLen, 2);

			// skip it
			File->seek(dataLen, true);
		}

		std::filesystem::path ZipFileName = "";

		if (header.flags & EGZF_FILE_NAME)
		{
			char c;
			File->read(&c, 1);
			while (c)
			{
				ZipFileName += c;
				File->read(&c, 1);
			}
		}
		else
		{
			// no file name?
			ZipFileName = Path;
			core::deletePathFromFilename(ZipFileName);

			// rename tgz to tar or remove gz extension
			if (core::hasFileExtension(ZipFileName, "tgz"))
			{
				ZipFileName.string()[ ZipFileName.string().size() - 2] = 'a';
				ZipFileName.string()[ ZipFileName.string().size() - 1] = 'r';
			}
			else if (core::hasFileExtension(ZipFileName, "gz"))
			{
				ZipFileName.string()[ ZipFileName.string().size() - 3] = 0;
			}
		}

		if (header.flags & EGZF_COMMENT)
		{
			char c='a';
			while (c)
				File->read(&c, 1);
		}

		if (header.flags & EGZF_CRC16)
			File->seek(2, true);

		// we are now at the start of the data blocks
		entry.Offset = File->getPos();

		entry.header.FilenameLength = ZipFileName.string().length();

		entry.header.CompressionMethod = header.compressionMethod;
		entry.header.DataDescriptor.CompressedSize = (File->getSize() - 8) - File->getPos();

		// seek to file end
		File->seek(entry.header.DataDescriptor.CompressedSize, true);

		// read CRC
		File->read(&entry.header.DataDescriptor.CRC32, 4);
		// read uncompressed size
		File->read(&entry.header.DataDescriptor.UncompressedSize, 4);

		// now we've filled all the fields, this is just a standard deflate block
		addItem(ZipFileName, entry.Offset, entry.header.DataDescriptor.UncompressedSize, false, 0);
		FileInfo.push_back(entry);
	}

	// there's only one block of data in a gzip file
	return false;
}


#ifdef _NBL_COMPILE_WITH_LZMA_
//! Used for LZMA decompression. The lib has no default memory management
namespace
{
	void *SzAlloc(void *p, size_t size) { p = p; return _NBL_ALIGNED_MALLOC(size,_NBL_SIMD_ALIGNMENT); }
	void SzFree(void *p, void *address) { p = p; _NBL_ALIGNED_FREE(address); }
	ISzAlloc lzmaAlloc = { SzAlloc, SzFree };
}
#endif