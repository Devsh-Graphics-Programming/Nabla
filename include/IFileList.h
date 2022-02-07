// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_I_FILE_LIST_H_INCLUDED__
#define __NBL_I_FILE_LIST_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"
#include "path.h"

namespace nbl
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
    inline bool operator==(const struct SFileListEntry& other) const
    {
        if(IsDirectory != other.IsDirectory)
            return false;

        return FullName.equals_ignore_case(other.FullName);
    }

    //! The < operator is provided so that CFileList can sort and quickly search the list.
    inline bool operator<(const struct SFileListEntry& other) const
    {
        if(IsDirectory != other.IsDirectory)
            return IsDirectory;

        return FullName.lower_ignore_case(other.FullName);
    }
};

//! Provides a list of files and folders.
/** File lists usually contain a list of all files in a given folder,
but can also contain a complete directory structure. */
class IFileList : public virtual core::IReferenceCounted
{
public:
    typedef core::vector<SFileListEntry>::const_iterator ListCIterator;

    //! Get the number of files in the filelist.
    /** \return Amount of files and directories in the file list. */
    virtual uint32_t getFileCount() const = 0;

    //!
    virtual core::vector<SFileListEntry> getFiles() const = 0;

    //! If @retval not equal to @param _end then file was found and return value is a valid pointer in the range given
    virtual ListCIterator findFile(ListCIterator _begin, ListCIterator _end, const io::path& filename, bool isDirectory = false) const = 0;

    //! Returns the base path of the file list
    virtual const io::path& getPath() const = 0;
};

}  // end namespace nbl
}  // end namespace io

#endif
