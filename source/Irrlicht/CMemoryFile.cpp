// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CMemoryFile.h"

namespace irr
{
namespace io
{

template<class T>
CMemoryFile<T>::CMemoryFile(T memory, const size_t& len, const io::path& fileName, bool d)
                    : Buffer(memory), Len(len), Pos(0), Filename(fileName), deleteMemoryWhenDropped(d)
{
}

template<class T>
CMemoryFile<T>::~CMemoryFile()
{
	if (deleteMemoryWhenDropped)
		delete [] (int8_t*)Buffer;
}



//! Constructor
CMemoryReadFile::CMemoryReadFile(const void* memory, const size_t& len, const io::path& fileName, bool d)
                : CMemoryFile<const void*>(memory,len,fileName,d)
{
	#ifdef _DEBUG
	setDebugName("CMemoryReadFile");
	#endif
}

//! returns how much was read
int32_t CMemoryReadFile::read(void* buffer, uint32_t sizeToRead)
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


//! Constructor
CMemoryWriteFile::CMemoryWriteFile(void* memory, const size_t& len, const io::path& fileName, bool d)
                : CMemoryFile<void*>(memory,len,fileName,d)
{
	#ifdef _DEBUG
	setDebugName("CMemoryWriteFile");
	#endif
}

//! returns how much was written
int32_t CMemoryWriteFile::write(const void* buffer, uint32_t sizeToWrite)
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


IReadFile* createMemoryReadFile(const void* memory, const size_t& size, const io::path& fileName, bool deleteMemoryWhenDropped)
{
	CMemoryReadFile* file = new CMemoryReadFile(memory, size, fileName, deleteMemoryWhenDropped);
	return file;
}


} // end namespace io
} // end namespace irr

