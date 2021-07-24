// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_I_READ_FILE_H_INCLUDED__
#define __NBL_I_READ_FILE_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"
#include <filesystem>

namespace nbl
{
namespace system
{

	//! Interface providing read acess to a file.
	// @sadiuk inspect if virtual inheritance is needed here
	class IReadFile : public virtual core::IReferenceCounted
	{
	public:
		//! Reads an amount of bytes from the file.
		/** \param buffer Pointer to buffer where read bytes are written to.
		\param sizeToRead Amount of bytes to read from the file.
		\return How many bytes were read. */
		// @sadiuk impl of this for representation of real disk files should call ISystem::readFile
		virtual int32_t read(void* buffer, uint32_t offset, uint32_t sizeToRead) = 0;

		//! Get size of file.
		/** \return Size of the file in bytes. */
		virtual size_t getSize() const = 0;

		//! Get name of file.
		/** \return File name as zero terminated character string. */
		virtual const std::filesystem::path& getFileName() const = 0;
	};

} // end namespace io
} // end namespace nbl

#endif

