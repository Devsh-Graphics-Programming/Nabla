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
CMemoryWriteFile::CMemoryWriteFile(const size_t& len, const io::path& fileName)
                : CMemoryFile(len,fileName)
{
	#ifdef _IRR_DEBUG
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

