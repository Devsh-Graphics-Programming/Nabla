// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_I_FILE_LIST_H_INCLUDED__
#define __NBL_I_FILE_LIST_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"
#include <filesystem>

namespace nbl
{
namespace system
{

//! An entry in a list of files, can be a folder or a file.
struct SFileListEntry
{
	//! The name of the file including the path
	/** 
	Extract just filename with FullName.filename().
	*/
	std::filesystem::path FullName;

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
	inline bool operator ==(const struct SFileListEntry& other) const
	{
		if (IsDirectory != other.IsDirectory)
			return false;

		const auto& a = FullName.string();
		const auto& b = other.FullName.string();
		return std::equal(std::begin(a), std::end(a), std::begin(b), [](auto a, auto b) {return std::tolower(a) == std::tolower(b); });
	}

	//! The < operator is provided so that CFileList can sort and quickly search the list.
	inline bool operator <(const struct SFileListEntry& other) const
	{
		if (IsDirectory != other.IsDirectory)
			return IsDirectory;

		const auto& a = FullName.string();
		const auto& b = other.FullName.string();
		return std::equal(std::begin(a), std::end(a), std::begin(b), [](auto a, auto b) {return std::tolower(a) < std::tolower(b); });
	}
};

//! Provides a list of files and folders.
/** File lists usually contain a list of all files in a given folder,
but can also contain a complete directory structure. */
class IFileList : public core::IReferenceCounted
{
public:
    typedef core::vector<SFileListEntry>::const_iterator ListCIterator;

	//! Get the number of files in the filelist.
	/** \return Amount of files and directories in the file list. */
	virtual uint32_t getFileCount() const = 0;

	//!
	virtual const core::vector<SFileListEntry>& getFiles() const = 0;

    //! If @retval not equal to @param _end then file was found and return value is a valid pointer in the range given
	virtual ListCIterator findFile(ListCIterator _begin, ListCIterator _end, const std::filesystem::path& filename, bool isDirectory = false) const = 0;

	//! Returns the base path of the file list
	virtual const std::filesystem::path& getPath() const = 0;
};

} // end namespace nbl
} // end namespace io


#endif

