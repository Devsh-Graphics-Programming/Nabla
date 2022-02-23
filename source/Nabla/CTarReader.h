// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors
#ifdef NEW_FILESYSTEM
#ifndef __NBL_C_TAR_READER_H_INCLUDED__
#define __NBL_C_TAR_READER_H_INCLUDED__

#include "nbl/asset/compile_config.h"

#ifdef __NBL_COMPILE_WITH_TAR_ARCHIVE_LOADER_

#include "nbl/core/IReferenceCounted.h"
#include "nbl/system/IFile.h"
#include "IFileSystem.h"

namespace nbl
{
namespace io
{

	enum E_TAR_LINK_INDICATOR
	{
		ETLI_REGULAR_FILE_OLD      =  0 ,
		ETLI_REGULAR_FILE          = '0',
		ETLI_LINK_TO_ARCHIVED_FILE = '1', // Hard link
		ETLI_SYMBOLIC_LINK         = '2',
		ETLI_CHAR_SPECIAL_DEVICE   = '3',
		ETLI_BLOCK_SPECIAL_DEVICE  = '4',
		ETLI_DIRECTORY             = '5',
		ETLI_FIFO_SPECIAL_FILE     = '6',
		ETLI_CONTIGUOUS_FILE       = '7'
	};

// byte-align structures
#include "nbl/nblpack.h"

	struct STarHeader
	{
		char FileName[100];
		char FileMode[8];
		char UserID[8];
		char GroupID[8];
		char Size[12];
		char ModifiedTime[12];
		char Checksum[8];
		char Link;
		char LinkName[100];
		char Magic[6];
		char USTARVersion[2];
		char UserName[32];
		char GroupName[32];
		char DeviceMajor[8];
		char DeviceMinor[8];
		char FileNamePrefix[155];
	} PACK_STRUCT;

// Default alignment
#include "nbl/nblunpack.h"

	//! Archiveloader capable of loading ZIP Archives
	class CArchiveLoaderTAR : public IArchiveLoader
	{
	public:

		//! Constructor
		CArchiveLoaderTAR(system::ISystem* fs);

		//! returns true if the file maybe is able to be loaded by this class
		//! based on the file extension (e.g. ".tar")
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

	private:
		io::IFileSystem* FileSystem;
	};



	class CTarReader : public virtual IFileArchive, virtual CFileList
	{
    protected:
		virtual ~CTarReader();

	public:
		CTarReader(IReadFile* file);

		//! opens a file by file name
		virtual IReadFile* createAndOpenFile(const std::filesystem::path& filename);

		//! returns the list of files
		virtual const IFileList* getFileList() const;

		//! get the class Type
		virtual E_FILE_ARCHIVE_TYPE getType() const { return EFAT_TAR; }

	private:

		uint32_t populateFileList();

		IReadFile* File;
	};

} // end namespace io
} // end namespace nbl

#endif // __NBL_COMPILE_WITH_TAR_ARCHIVE_LOADER_
#endif
#endif