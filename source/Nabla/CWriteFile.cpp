// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "CWriteFile.h"
#include <stdio.h>

namespace nbl
{
namespace io
{
CWriteFile::CWriteFile(const io::path& fileName, bool append)
    : FileSize(0)
{
#ifdef _NBL_DEBUG
    setDebugName("CWriteFile");
#endif

    Filename = fileName;
    openFile(append);
}

CWriteFile::~CWriteFile()
{
    if(File)
        fclose(File);
}

//! returns if file is open
inline bool CWriteFile::isOpen() const
{
    return File != 0;
}

//! returns how much was read
int32_t CWriteFile::write(const void* buffer, uint32_t sizeToWrite)
{
    if(!isOpen())
        return 0;

    return (int32_t)fwrite(buffer, 1, sizeToWrite, File);
}

//! changes position in file, returns true if successful
//! if relativeMovement==true, the pos is changed relative to current pos,
//! otherwise from begin of file
bool CWriteFile::seek(const size_t& finalPos, bool relativeMovement)
{
    if(!isOpen())
        return false;

    return fseek(File, finalPos, relativeMovement ? SEEK_CUR : SEEK_SET) == 0;
}

//! returns where in the file we are.
size_t CWriteFile::getPos() const
{
    return ftell(File);
}

//! opens the file
void CWriteFile::openFile(bool append)
{
    if(Filename.size() == 0)
    {
        File = 0;
        return;
    }

#if defined(_NBL_WCHAR_FILESYSTEM)
    File = _wfopen(Filename.c_str(), append ? L"ab" : L"wb");
#else
    File = fopen(Filename.c_str(), append ? "ab" : "wb");
#endif

    if(File)
    {
        // get FileSize

        fseek(File, 0, SEEK_END);
        FileSize = ftell(File);
        fseek(File, 0, SEEK_SET);
    }
}

//! returns name of file
const io::path& CWriteFile::getFileName() const
{
    return Filename;
}

IWriteFile* createWriteFile(const io::path& fileName, bool append)
{
    CWriteFile* file = new CWriteFile(fileName, append);
    if(file->isOpen())
        return file;

    file->drop();
    return 0;
}

}  // end namespace io
}  // end namespace nbl
