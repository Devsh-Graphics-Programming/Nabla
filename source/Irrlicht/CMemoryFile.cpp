// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CMemoryFile.h"

namespace irr
{
namespace io
{

CMemoryFile::CMemoryFile(const size_t& len, const io::path& fileName)
                    : Buffer(len), Pos(0), Filename(fileName)
{
}

CMemoryFile::~CMemoryFile()
{
}



//! Constructor
CMemoryReadFile::CMemoryReadFile(const void* contents, const size_t& len, const io::path& fileName)
                : CMemoryFile(len,fileName)
{
	#ifdef _DEBUG
	setDebugName("CMemoryReadFile");
	#endif
    memcpy(Buffer.data(), contents, len);
}

//! returns how much was read
int32_t CMemoryReadFile::read(void* buffer, uint32_t sizeToRead)
{
	int64_t amount = static_cast<int64_t>(sizeToRead);
	if (Pos + amount > getSize())
		amount -= Pos + amount - Buffer.size();

	if (amount <= 0ll)
		return 0;

	memcpy(buffer, Buffer.data() + Pos, amount);

	Pos += amount;

	return static_cast<int32_t>(amount);
}


//! Constructor
CMemoryWriteFile::CMemoryWriteFile(const size_t& len, const io::path& fileName)
                : CMemoryFile(len,fileName)
{
	#ifdef _DEBUG
	setDebugName("CMemoryWriteFile");
	#endif
}

bool CMemoryWriteFile::seek(const size_t& finalPos, bool relativeMovement)
{
    Pos = size_t(relativeMovement)*Pos + finalPos;
    return true;
}

//! returns how much was written
int32_t CMemoryWriteFile::write(const void* buffer, uint32_t sizeToWrite)
{
    if (Pos + sizeToWrite > Buffer.size())
        Buffer.resize(Pos + sizeToWrite);

	memcpy(Buffer.data() + Pos, buffer, sizeToWrite);

	Pos += sizeToWrite;

	return sizeToWrite;
}


} // end namespace io
} // end namespace irr

