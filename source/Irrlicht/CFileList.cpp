// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CFileList.h"
#include "IrrCompileConfig.h"
#include "coreutil.h"

#include <algorithm>

#include "os.h"

namespace irr
{
namespace io
{

static const io::path emptyFileListEntry;

CFileList::CFileList(const io::path& path, bool ignoreCase, bool ignorePaths)
 : IgnorePaths(ignorePaths), IgnoreCase(ignoreCase), Path(path)
{
	#ifdef _DEBUG
	setDebugName("CFileList");
	#endif

	handleBackslashes(&Path);
}

CFileList::~CFileList()
{
	Files.clear();
}

uint32_t CFileList::getFileCount() const
{
	return Files.size();
}

void CFileList::sort()
{
	std::sort(Files.begin(),Files.end());
}

const io::path& CFileList::getFileName(uint32_t index) const
{
	if (index >= Files.size())
		return emptyFileListEntry;

	return Files[index].Name;
}


//! Gets the full name of a file in the list, path included, based on an index.
const io::path& CFileList::getFullFileName(uint32_t index) const
{
	if (index >= Files.size())
		return emptyFileListEntry;

	return Files[index].FullName;
}

//! adds a file or folder
uint32_t CFileList::addItem(const io::path& fullPath, uint32_t offset, uint32_t size, bool isDirectory, uint32_t id)
{
	SFileListEntry entry;
	entry.ID   = id ? id : Files.size();
	entry.Offset = offset;
	entry.Size = size;
	entry.Name = fullPath;
	handleBackslashes(&entry.Name);
	entry.IsDirectory = isDirectory;

	// remove trailing slash
	if (entry.Name.lastChar() == '/')
	{
		entry.IsDirectory = true;
		entry.Name[entry.Name.size()-1] = 0;
		entry.Name.validate();
	}

	if (IgnoreCase)
		entry.Name.make_lower();

	entry.FullName = entry.Name;

	core::deletePathFromFilename(entry.Name);

	if (IgnorePaths)
		entry.FullName = entry.Name;

	//os::Printer::log(Path.c_str(), entry.FullName);

	Files.push_back(entry);

	return Files.size() - 1;
}

//! Returns the ID of a file in the file list, based on an index.
uint32_t CFileList::getID(uint32_t index) const
{
	return index < Files.size() ? Files[index].ID : 0;
}

bool CFileList::isDirectory(uint32_t index) const
{
	bool ret = false;
	if (index < Files.size())
		ret = Files[index].IsDirectory;

	return ret;
}

//! Returns the size of a file
uint32_t CFileList::getFileSize(uint32_t index) const
{
	return index < Files.size() ? Files[index].Size : 0;
}

//! Returns the size of a file
uint32_t CFileList::getFileOffset(uint32_t index) const
{
	return index < Files.size() ? Files[index].Offset : 0;
}


//! Searches for a file or folder within the list, returns the index
int32_t CFileList::findFile(const io::path& filename, bool isDirectory = false) const
{
	SFileListEntry entry;
	// we only need FullName to be set for the search
	entry.FullName = filename;
	entry.IsDirectory = isDirectory;

	// exchange
	handleBackslashes(&entry.FullName);

	// remove trailing slash
	if (entry.FullName.lastChar() == '/')
	{
		entry.IsDirectory = true;
		entry.FullName[entry.FullName.size()-1] = 0;
		entry.FullName.validate();
	}

	if (IgnoreCase)
		entry.FullName.make_lower();

	if (IgnorePaths)
		core::deletePathFromFilename(entry.FullName);

    auto retval = std::lower_bound(Files.begin(),Files.end(),entry);
    if (retval!=Files.end() && !(entry<*retval))
        return std::distance(Files.begin(),retval);
    else
        return -1;
}


//! Returns the base path of the file list
const io::path& CFileList::getPath() const
{
	return Path;
}


} // end namespace irr
} // end namespace io

