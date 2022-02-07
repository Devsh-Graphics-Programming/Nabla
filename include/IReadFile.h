// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_I_READ_FILE_H_INCLUDED__
#define __NBL_I_READ_FILE_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"
#include "nbl/core/string/stringutil.h"
#include "path.h"

namespace nbl
{
namespace io
{
//! Interface providing read acess to a file.
class IReadFile : public virtual core::IReferenceCounted
{
public:
    //! Reads an amount of bytes from the file.
    /** \param buffer Pointer to buffer where read bytes are written to.
		\param sizeToRead Amount of bytes to read from the file.
		\return How many bytes were read. */
    virtual int32_t read(void* buffer, uint32_t sizeToRead) = 0;

    //! Changes position in file
    /** \param finalPos Destination position in the file.
		\param relativeMovement If set to true, the position in the file is
		changed relative to current position. Otherwise the position is changed
		from beginning of file.
		\return True if successful, otherwise false. */
    virtual bool seek(const size_t& finalPos, bool relativeMovement = false) = 0;

    //! Get size of file.
    /** \return Size of the file in bytes. */
    virtual size_t getSize() const = 0;

    //! Get the current position in the file.
    /** \return Current position in the file in bytes. */
    virtual size_t getPos() const = 0;

    //! Get name of file.
    /** \return File name as zero terminated character string. */
    virtual const io::path& getFileName() const = 0;
};

}  // end namespace io
}  // end namespace nbl

#endif
