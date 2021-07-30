// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_C_PAK_READER_H_INCLUDED__
#define __NBL_C_PAK_READER_H_INCLUDED__

#include "nbl/asset/compile_config.h"

#ifdef __NBL_COMPILE_WITH_PAK_ARCHIVE_LOADER_

#include "nbl/core/IReferenceCounted.h"
#include "nbl/system/IFile.h"
#include "IFileSystem.h"

namespace nbl
{
namespace io
{
	//! File header containing location and size of the table of contents
	struct SPAKFileHeader
	{
		// Don't change the order of these fields!  They must match the order stored on disk.
		char tag[4];
		uint32_t offset;
		uint32_t length;
	};

	//! An entry in the PAK file's table of contents.
	struct SPAKFileEntry
	{
		// Don't change the order of these fields!  They must match the order stored on disk.
		char name[56];
		uint32_t offset;
		uint32_t length;
	};

	//! Archiveloader capable of loading PAK Archives
	class CArchiveLoaderPAK : public IArchiveLoader
	{
	public:

		//! Constructor
		CArchiveLoaderPAK(io::IFileSystem* fs);

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
		virtual bool isALoadableFileFormat(E_FILE_ARCHIVE_TYPE fileType) const;

		//! Creates an archive from the filename
		/** \param file File handle to check.
		\return Pointer to newly created archive, or 0 upon error. */
		virtual IFileArchive* createArchive(const std::filesystem::path& filename) const;

		//! creates/loads an archive from the file.
		//! \return Pointer to the created archive. Returns 0 if loading failed.
		virtual io::IFileArchive* createArchive(io::IReadFile* file) const;

		//! Returns the type of archive created by this loader
		virtual E_FILE_ARCHIVE_TYPE getType() const { return EFAT_PAK; }

	private:
		io::IFileSystem* FileSystem;
	};


	//! reads from pak
	class CPakReader : public virtual IFileArchive, virtual CFileList
	{
    protected:
		virtual ~CPakReader();

	public:
		CPakReader(IReadFile* file);

		// file archive methods

		//! return the id of the file Archive
		virtual const std::filesystem::path& getArchiveName() const
		{
			return File->getFileName();
		}

		//! opens a file by file name
		virtual IReadFile* createAndOpenFile(const std::filesystem::path& filename);

		//! returns the list of files
		virtual const IFileList* getFileList() const;

		//! get the class Type
		virtual E_FILE_ARCHIVE_TYPE getType() const { return EFAT_PAK; }

	private:

		//! scans for a local header, returns false if the header is invalid
		bool scanLocalHeader();

		IReadFile* File;

	};

} // end namespace io
} // end namespace nbl

#endif // __NBL_COMPILE_WITH_PAK_ARCHIVE_LOADER_

#endif

