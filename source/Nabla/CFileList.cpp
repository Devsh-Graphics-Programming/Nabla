// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "CFileList.h"
#include "nbl/core/core.h"

#include <algorithm>

#include "nbl_os.h"

namespace nbl
{
namespace io
{

static const io::path emptyFileListEntry;

CFileList::CFileList(const io::path& path) : Path(path)
{
	#ifdef _NBL_DEBUG
	setDebugName("CFileList");
	#endif

	handleBackslashes(&Path);
}

CFileList::~CFileList()
{
	Files.clear();
}

//! adds a file or folder
void CFileList::addItem(const io::path& fullPath, uint32_t offset, uint32_t size, bool isDirectory, uint32_t id)
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

	entry.FullName = entry.Name;

	core::deletePathFromFilename(entry.Name);

	//os::Printer::log(Path.c_str(), entry.FullName);

	Files.insert(std::lower_bound(Files.begin(),Files.end(),entry),entry);
}



//! Searches for a file or folder within the list, returns the index
IFileList::ListCIterator CFileList::findFile(IFileList::ListCIterator _begin, IFileList::ListCIterator _end, const io::path& filename, bool isDirectory) const
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
	entry.Name = entry.FullName;

    auto retval = std::lower_bound(_begin,_end,entry);
    if (retval!=_end && entry<*retval)
        return _end;
    return retval;
}


} // end namespace nbl
} // end namespace io

