// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "nbl/core/math/glslFunctions.tcc"

#include "CLimitReadFile.h"

namespace nbl
{
namespace io
{
CLimitReadFile::CLimitReadFile(IReadFile* alreadyOpenedFile, const size_t& pos,
    const size_t& areaSize, const io::path& name)
    : Filename(name), AreaStart(0), AreaEnd(0), Pos(0),
      File(alreadyOpenedFile)
{
#ifdef _NBL_DEBUG
    setDebugName("CLimitReadFile");
#endif

    if(File)
    {
        File->grab();
        AreaStart = pos;
        AreaEnd = AreaStart + areaSize;
    }
}

CLimitReadFile::~CLimitReadFile()
{
    if(File)
        File->drop();
}

//! returns how much was read
int32_t CLimitReadFile::read(void* buffer, uint32_t sizeToRead)
{
#if 1
    if(0 == File)
        return 0;

    int32_t r = AreaStart + Pos;
    int32_t toRead = core::min<int32_t>(AreaEnd, r + sizeToRead) - core::max<int32_t>(AreaStart, r);
    if(toRead < 0)
        return 0;
    File->seek(r);
    r = File->read(buffer, toRead);
    Pos += r;
    return r;
#else
    const size_t pos = File->getPos();

    if(pos >= AreaEnd)
        return 0;

    if(pos + (size_t)sizeToRead >= AreaEnd)
        sizeToRead = AreaEnd - pos;

    return File->read(buffer, sizeToRead);
#endif
}

//! changes position in file, returns true if successful
bool CLimitReadFile::seek(const size_t& finalPos, bool relativeMovement)
{
#if 1
    Pos = core::clamp<int32_t, int32_t>(finalPos + (relativeMovement ? Pos : 0), 0, AreaEnd - AreaStart);
    return true;
#else
    const size_t pos = File->getPos();

    if(relativeMovement)
    {
        if(pos + finalPos > AreaEnd)
            finalPos = AreaEnd - pos;
    }
    else
    {
        finalPos += AreaStart;
        if(finalPos > AreaEnd)
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

}  // end namespace io
}  // end namespace nbl
