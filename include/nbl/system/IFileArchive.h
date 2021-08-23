// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_I_FILE_ARCHIVE_H_INCLUDED__
#define __NBL_I_FILE_ARCHIVE_H_INCLUDED__

#include "nbl/system/IFile.h"

namespace nbl
{

namespace system
{

//! The FileArchive manages archives and provides access to files inside them.
class IFileArchive : public core::IReferenceCounted
{
public:
	struct SOpenFileParams
	{
		std::filesystem::path filename;
		std::string_view password;
	};

	//! Opens a file based on its name
	virtual core::smart_refctd_ptr<IFile> readFile(const SOpenFileParams& params) = 0;
};


class IArchiveLoader : public core::IReferenceCounted
{
public:
	//! Check if the file might be loaded by this class
	/** This check may look into the file.
	\param file File handle to check.
	\return True if file seems to be loadable. */
	virtual bool isALoadableFileFormat(IFile* file) const = 0;

	//! Returns an array of string literals terminated by nullptr
	virtual const char** getAssociatedFileExtensions() const = 0;

	virtual bool isMountable() const = 0;

	virtual bool mount(const IFile* file, const std::string_view& passphrase) = 0;
	
	virtual bool unmount(const IFile* file) = 0;

	//! Creates an archive from the file
	/** \param file File handle to use.
	\return Pointer to newly created archive, or 0 upon error. */
	core::smart_refctd_ptr<IFileArchive> createArchive(IFile* file) const
	{
		if (!(file->getFlags() & IFile::ECF_READ))
			return nullptr;

		return createArchive_impl(file);
	}

protected:
	virtual core::smart_refctd_ptr<IFileArchive> createArchive_impl(IFile* file) const = 0;
};


} // end namespace system
} // end namespace nbl

#endif

