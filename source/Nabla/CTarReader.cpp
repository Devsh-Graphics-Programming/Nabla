//! Check if the file might be loaded by this class
/** Check might look into the file.
\param file File handle to check.
\return True if file seems to be loadable. */
bool CArchiveLoaderTAR::isALoadableFileFormat(io::IReadFile* file) const
{
	// TAR files consist of blocks of 512 bytes
	// if it isn't a multiple of 512 then it's not a TAR file.
	if (file->getSize() % 512)
		return false;

	const size_t prevPos = file->getPos();
	file->seek(0u);

	// read header of first file
	STarHeader fHead;
	file->read(&fHead, sizeof(STarHeader));
	file->seek(prevPos);

	uint32_t checksum = 0;
	sscanf(fHead.Checksum, "%o", &checksum);

	// verify checksum

	// some old TAR writers assume that chars are signed, others assume unsigned
	// USTAR archives have a longer header, old TAR archives end after linkname

	uint32_t checksum1=0;
	int32_t checksum2=0;

	// remember to blank the checksum field!
	memset(fHead.Checksum, ' ', 8);

	// old header
	for (uint8_t* p = (uint8_t*)(&fHead); p < (uint8_t*)(&fHead.Magic[0]); ++p)
	{
		checksum1 += *p;
		checksum2 += char(*p);
	}

	if (!strncmp(fHead.Magic, "ustar", 5))
	{
		for (uint8_t* p = (uint8_t*)(&fHead.Magic[0]); p < (uint8_t*)(&fHead) + sizeof(fHead); ++p)
		{
			checksum1 += *p;
			checksum2 += char(*p);
		}
	}
	return checksum1 == checksum || checksum2 == (int32_t)checksum;
}