// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_I_WRITE_FILE_H_INCLUDED__
#define __NBL_I_WRITE_FILE_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"
#include <filesystem>

namespace nbl
{
namespace system
{

	//! Interface providing write access to a file.
	// @sadiuk inspect if virtual inheritance is needed here
	class IWriteFile : public virtual core::IReferenceCounted
	{
	public:
		//! Writes an amount of bytes to the file.
		/** \param buffer Pointer to buffer of bytes to write.
		\param sizeToWrite Amount of bytes to write to the file.
		\return How much bytes were written. */
		// @sadiuk impl of this for representation of real disk files should call ISystem::writeFile
		virtual int32_t write(const void* buffer, uint32_t offset, uint32_t sizeToWrite) = 0;

		//! Get name of file.
		/** \return File name as zero terminated character string. */
		virtual const std::filesystem::path& getFileName() const = 0;
	};

} // end namespace io
} // end namespace nbl

#endif

