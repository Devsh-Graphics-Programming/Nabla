// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CLimitReadFile.h"

namespace irr
{
namespace io
{


CLimitReadFile::CLimitReadFile(IReadFile* alreadyOpenedFile, const size_t& pos,
		const size_t& areaSize, const io::path& name)
	: Filename(name), AreaStart(0), AreaEnd(0), Pos(0),
	File(alreadyOpenedFile)
{
	#ifdef _IRR_DEBUG
	setDebugName("CLimitReadFile");
	#endif

	if (File)
	{
		File->grab();
		AreaStart = pos;
		AreaEnd = AreaStart + areaSize;
	}
}


CLimitReadFile::~CLimitReadFile()
{
	if (File)
		File->drop();
}


//! returns how much was read
int32_t CLimitReadFile::read(void* buffer, uint32_t sizeToRead)
{
#if 1
	if (0 == File)
		return 0;

	int32_t r = AreaStart + Pos;
	int32_t toRead = core::s32_min(AreaEnd, r + sizeToRead) - core::s32_max(AreaStart, r);
	if (toRead < 0)
		return 0;
	File->seek(r);
	r = File->read(buffer, toRead);
	Pos += r;
	return r;
#else
	const size_t pos = File->getPos();

	if (pos >= AreaEnd)
		return 0;

	if (pos + (size_t)sizeToRead >= AreaEnd)
		sizeToRead = AreaEnd - pos;

	return File->read(buffer, sizeToRead);
#endif
}


//! changes position in file, returns true if successful
bool CLimitReadFile::seek(const size_t& finalPos, bool relativeMovement)
{
#if 1
	Pos = core::s32_clamp(finalPos + (relativeMovement ? Pos : 0 ), 0, AreaEnd - AreaStart);
	return true;
#else
	const size_t pos = File->getPos();

	if (relativeMovement)
	{
		if (pos + finalPos > AreaEnd)
			finalPos = AreaEnd - pos;
	}
	else
	{
		finalPos += AreaStart;
		if (finalPos > AreaEnd)
			return false;
	}

	return File->seek(finalPos, relativeMovement);
#endif
}


//! returns size of file
size_t CLimitReadFile::getSize() const
{
	return AreaEnd - AreaStart;
}


//! returns where in the file we are.
size_t CLimitReadFile::getPos() const
{
#if 1
	return Pos;
#else
	return File->getPos() - AreaStart;
#endif
}


//! returns name of file
const io::path& CLimitReadFile::getFileName() const
{
	return Filename;
}


} // end namespace io
} // end namespace irr

