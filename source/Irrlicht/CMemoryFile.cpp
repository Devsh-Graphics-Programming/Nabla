// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CMemoryFile.h"
#include "irrString.h"

namespace irr
{
namespace io
{


CMemoryFile::CMemoryFile(void* memory, long len, const io::path& fileName, bool d)
: Buffer(memory), Len(len), Pos(0), Filename(fileName), deleteMemoryWhenDropped(d)
{
	#ifdef _DEBUG
	setDebugName("CMemoryFile");
	#endif
}


CMemoryFile::~CMemoryFile()
{
	if (deleteMemoryWhenDropped)
		delete [] (int8_t*)Buffer;
}


//! returns how much was read
int32_t CMemoryFile::read(void* buffer, uint32_t sizeToRead)
{
	int32_t amount = static_cast<int32_t>(sizeToRead);
	if (Pos + amount > Len)
		amount -= Pos + amount - Len;

	if (amount <= 0)
		return 0;

	int8_t* p = (int8_t*)Buffer;
	memcpy(buffer, p + Pos, amount);

	Pos += amount;

	return amount;
}

//! returns how much was written
int32_t CMemoryFile::write(const void* buffer, uint32_t sizeToWrite)
{
	int32_t amount = static_cast<int32_t>(sizeToWrite);
	if (Pos + amount > Len)
		amount -= Pos + amount - Len;

	if (amount <= 0)
		return 0;

	int8_t* p = (int8_t*)Buffer;
	memcpy(p + Pos, buffer, amount);

	Pos += amount;

	return amount;
}



//! changes position in file, returns true if successful
//! if relativeMovement==true, the pos is changed relative to current pos,
//! otherwise from begin of file
bool CMemoryFile::seek(long finalPos, bool relativeMovement)
{
	if (relativeMovement)
	{
		if (Pos + finalPos > Len)
			return false;

		Pos += finalPos;
	}
	else
	{
		if (finalPos > Len)
			return false;

		Pos = finalPos;
	}

	return true;
}


//! returns size of file
long CMemoryFile::getSize() const
{
	return Len;
}


//! returns where in the file we are.
long CMemoryFile::getPos() const
{
	return Pos;
}


//! returns name of file
const io::path& CMemoryFile::getFileName() const
{
	return Filename;
}


IReadFile* createMemoryReadFile(void* memory, long size, const io::path& fileName, bool deleteMemoryWhenDropped)
{
	CMemoryFile* file = new CMemoryFile(memory, size, fileName, deleteMemoryWhenDropped);
	return file;
}


} // end namespace io
} // end namespace irr

