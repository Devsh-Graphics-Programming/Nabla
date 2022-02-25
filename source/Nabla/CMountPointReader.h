// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_C_MOUNT_READER_H_INCLUDED__
#define __NBL_C_MOUNT_READER_H_INCLUDED__

#include "nbl/asset/compile_config.h"

#ifdef __NBL_COMPILE_WITH_MOUNT_ARCHIVE_LOADER_

#include "nbl/system/ISystem.h"
#include "nbl/system/IFile.h"

namespace nbl
{
namespace io
{

	//! Archiveloader capable of loading MountPoint Archives
	class CArchiveLoaderMount : public system::IArchiveLoader
	{
	public:

		//! Constructor
		CArchiveLoaderMount(io::IFileSystem* fs);

		//! returns true if the file maybe is able to be loaded by this class
		//! based on the file extension (e.g. ".zip")
		virtual bool isALoadableFileFormat(const std::filesystem::path& filename) const;

		//! Check if the file might be loaded by this class
		/** Check might look into the file.
		\param file File handle to check.
		\return True if file seems to be loadable. */
		virtual bool isALoadableFileFormat(io::IReadFile* file) const;

		//! Check to see if the loader can create archives of this type.
		/** Check based on the archive type.
		\param fileType The archive type to check.
		\return True if the archile loader supports this type, false if not */
		//virtual bool isALoadableFileFormat(E_FILE_ARCHIVE_TYPE fileType) const;

		//! Creates an archive from the filename
		/** \param file File handle to check.
		\return Pointer to newly created archive, or 0 upon error. */
		//! creates/loads an archive from the file.
		//! \return Pointer to the created archive. Returns 0 if loading failed.
		//virtual IFileArchive* createArchive(io::IReadFile* file) const;

	private:
		io::IFileSystem* FileSystem;
	};

	//! A File Archive which uses a mountpoint
	class CMountPointReader : public virtual system::IFileArchive//, virtual system::CFileList
	{
	public:

		//! Constructor
		CMountPointReader(IFileSystem *parent, const std::filesystem::path& basename);

		//! opens a file by file name
		virtual IReadFile* createAndOpenFile(const std::filesystem::path& filename);

		//! returns the list of files
		virtual const IFileList* getFileList() const;

		//! get the class Type
		//virtual E_FILE_ARCHIVE_TYPE getType() const { return EFAT_FOLDER; }

	private:

		core::vector<std::filesystem::path> RealFileNames;

		IFileSystem *Parent;
		void buildDirectory();
	};
} // io
} // nbl

#endif // __NBL_COMPILE_WITH_MOUNT_ARCHIVE_LOADER_
#endif
