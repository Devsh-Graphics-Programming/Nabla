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
	#ifdef _IRR_DEBUG
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

	if (IgnoreCase)
		entry.Name.make_lower();

	entry.FullName = entry.Name;

	core::deletePathFromFilename(entry.Name);

	if (IgnorePaths)
		entry.FullName = entry.Name;

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

    if (IgnoreCase)
        entry.FullName.make_lower();

    if (IgnorePaths)
        core::deletePathFromFilename(entry.FullName);

    auto retval = std::lower_bound(_begin,_end,entry);
    if (retval!=_end && entry<*retval)
        return _end;
    return retval;
}


} // end namespace irr
} // end namespace io

