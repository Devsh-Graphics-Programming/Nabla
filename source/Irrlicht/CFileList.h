// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_FILE_LIST_H_INCLUDED__
#define __C_FILE_LIST_H_INCLUDED__

#include "IFileList.h"
#include "irrString.h"
#include "irrArray.h"


namespace irr
{
namespace io
{

//! An entry in a list of files, can be a folder or a file.
struct SFileListEntry
{
	//! The name of the file
	/** If this is a file or folder in the virtual filesystem and the archive
	was created with the ignoreCase flag then the file name will be lower case. */
	io::path Name;

	//! The name of the file including the path
	/** If this is a file or folder in the virtual filesystem and the archive was
	created with the ignoreDirs flag then it will be the same as Name. */
	io::path FullName;

	//! The size of the file in bytes
	uint32_t Size;

	//! The ID of the file in an archive
	/** This is used to link the FileList entry to extra info held about this
	file in an archive, which can hold things like data offset and CRC. */
	uint32_t ID;

	//! FileOffset inside an archive
	uint32_t Offset;

	//! True if this is a folder, false if not.
	bool IsDirectory;

	//! The == operator is provided so that CFileList can slowly search the list!
	bool operator ==(const struct SFileListEntry& other) const
	{
		if (IsDirectory != other.IsDirectory)
			return false;

		return FullName.equals_ignore_case(other.FullName);
	}

	//! The < operator is provided so that CFileList can sort and quickly search the list.
	bool operator <(const struct SFileListEntry& other) const
	{
		if (IsDirectory != other.IsDirectory)
			return IsDirectory;

		return FullName.lower_ignore_case(other.FullName);
	}
};


//! Implementation of a file list
class CFileList : public IFileList
{
    protected:
        //! Destructor
        virtual ~CFileList();
    public:
        //! Constructor
        /** \param path The path of this file archive */
        CFileList(const io::path& path, bool ignoreCase, bool ignorePaths);

        //! Add as a file or folder to the list
        /** \param fullPath The file name including path, up to the root of the file list.
        \param isDirectory True if this is a directory rather than a file.
        \param offset The offset where the file is stored in an archive
        \param size The size of the file in bytes.
        \param id The ID of the file in the archive which owns it */
        virtual uint32_t addItem(const io::path& fullPath, uint32_t offset, uint32_t size, bool isDirectory, uint32_t id=0);

        //! Sorts the file list. You should call this after adding any items to the file list
        virtual void sort();

        //! Returns the amount of files in the filelist.
        virtual uint32_t getFileCount() const;

        //! Gets the name of a file in the list, based on an index.
        virtual const io::path& getFileName(uint32_t index) const;

        //! Gets the full name of a file in the list, path included, based on an index.
        virtual const io::path& getFullFileName(uint32_t index) const;

        //! Returns the ID of a file in the file list, based on an index.
        virtual uint32_t getID(uint32_t index) const;

        //! Returns true if the file is a directory
        virtual bool isDirectory(uint32_t index) const;

        //! Returns the size of a file
        virtual uint32_t getFileSize(uint32_t index) const;

        //! Returns the offest of a file
        virtual uint32_t getFileOffset(uint32_t index) const;

        //! Searches for a file or folder within the list, returns the index
        virtual int32_t findFile(const io::path& filename, bool isFolder) const;

        //! Returns the base path of the file list
        virtual const io::path& getPath() const;

    protected:

        //! Ignore paths when adding or searching for files
        bool IgnorePaths;

        //! Ignore case when adding or searching for files
        bool IgnoreCase;

        //! Path to the file list
        io::path Path;

        //! List of files
        core::array<SFileListEntry> Files;
};


} // end namespace irr
} // end namespace io


#endif

