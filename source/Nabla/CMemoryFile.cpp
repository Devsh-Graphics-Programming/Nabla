// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "CMemoryFile.h"

namespace nbl
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

#if 0
//! Constructor
CMemoryWriteFile::CMemoryWriteFile(const size_t& len, const io::path& fileName)
                : CMemoryFile(len,fileName)
{
	#ifdef _NBL_DEBUG
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
#endif

} // end namespace io
} // end namespace nbl

