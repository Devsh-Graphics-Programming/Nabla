// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_I_WRITE_FILE_H_INCLUDED__
#define __NBL_I_WRITE_FILE_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"
#include "path.h"

namespace nbl
{
namespace io
{
//! Interface providing write access to a file.
class IWriteFile : public virtual core::IReferenceCounted
{
public:
    //! Writes an amount of bytes to the file.
    /** \param buffer Pointer to buffer of bytes to write.
		\param sizeToWrite Amount of bytes to write to the file.
		\return How much bytes were written. */
    virtual int32_t write(const void* buffer, uint32_t sizeToWrite) = 0;

    //! Changes position in file
    /** \param finalPos Destination position in the file.
		\param relativeMovement If set to true, the position in the file is
		changed relative to current position. Otherwise the position is changed
		from begin of file.
		\return True if successful, otherwise false. */
    virtual bool seek(const size_t& finalPos, bool relativeMovement = false) = 0;

    //! Get the current position in the file.
    /** \return Current position in the file in bytes. */
    virtual size_t getPos() const = 0;

    //! Get name of file.
    /** \return File name as zero terminated character string. */
    virtual const path& getFileName() const = 0;
};

//! Internal function, please do not use.
IWriteFile* createWriteFile(const io::path& fileName, bool append);

}  // end namespace io
}  // end namespace nbl

#endif
