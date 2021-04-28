// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_I_FILE_ARCHIVE_H_INCLUDED__
#define __NBL_I_FILE_ARCHIVE_H_INCLUDED__

#include "nbl/system/IReadFile.h"
#include "nbl/system/IFileList.h"

namespace nbl
{

namespace system
{

// @sadiuk I think we can completely get rid of it,
// check if virtual filesystem is used anywhere in the engine somehow
//
//! FileSystemType: which Filesystem should be used for e.g. browsing
enum EFileSystemType
{
	FILESYSTEM_NATIVE = 0,	// Native OS FileSystem
	FILESYSTEM_VIRTUAL	// Virtual FileSystem
};

// @sadiuk Move this enum into IFileArchive
//! Contains the different types of archives
enum E_FILE_ARCHIVE_TYPE
{
	//! A PKZIP archive
	EFAT_ZIP,

	//! A gzip archive
	EFAT_GZIP,

	//! A virtual directory
	EFAT_FOLDER,

	//! An ID Software PAK archive
	EFAT_PAK,

	//! A Tape ARchive
	EFAT_TAR,

	//! The type of this archive is unknown
	EFAT_UNKNOWN
};

//! The FileArchive manages archives and provides access to files inside them.
class IFileArchive : public virtual core::IReferenceCounted
{
public:
	struct SOpenFileParams
	{
		std::string password;
	};

	//! Opens a file based on its name
	/** Creates and returns a new IReadFile for a file in the archive.
	\param filename The file to open
	\return Returns A pointer to the created file on success,
	or 0 on failure. */
	virtual core::smart_refctd_ptr<IReadFile> createAndOpenFile(const std::filesystem::path& filename, const SOpenFileParams& params) = 0;

	//! Returns the complete file tree
	/** \return Returns the complete directory tree for the archive,
	including all files and folders */
	virtual const IFileList* getFileList() const = 0;

	//! get the archive type
	virtual E_FILE_ARCHIVE_TYPE getType() const { return EFAT_UNKNOWN; }
};

//! Class which is able to create an archive from a file.
/** If you want the Irrlicht Engine be able to load archives of
currently unsupported file formats (e.g .wad), then implement
this and add your new Archive loader with
IFileSystem::addArchiveLoader() to the engine. */
class IArchiveLoader : public virtual core::IReferenceCounted
{
public:
	//! Check if the file might be loaded by this class
	/** Check based on the file extension (e.g. ".zip")
	\param filename Name of file to check.
	\return True if file seems to be loadable. */
	virtual bool isALoadableFileFormat(const std::filesystem::path& filename) const =0;

	//! Check if the file might be loaded by this class
	/** This check may look into the file.
	\param file File handle to check.
	\return True if file seems to be loadable. */
	virtual bool isALoadableFileFormat(IReadFile* file) const =0;

	//! Check to see if the loader can create archives of this type.
	/** Check based on the archive type.
	\param fileType The archive type to check.
	\return True if the archile loader supports this type, false if not */
	virtual bool isALoadableFileFormat(E_FILE_ARCHIVE_TYPE fileType) const =0;

	//! Creates an archive from the file
	/** \param file File handle to use.
	\param ignoreCase Searching is performed without regarding the case
	\param ignorePaths Files are searched for without checking for the directories
	\return Pointer to newly created archive, or 0 upon error. */
	virtual core::smart_refctd_ptr<IFileArchive> createArchive(IReadFile* file) const =0;
};


} // end namespace io
} // end namespace nbl

#endif

