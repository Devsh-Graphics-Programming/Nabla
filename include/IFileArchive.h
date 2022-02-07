// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_I_FILE_ARCHIVE_H_INCLUDED__
#define __NBL_I_FILE_ARCHIVE_H_INCLUDED__

#include "IReadFile.h"
#include "IFileList.h"

namespace nbl
{
namespace io
{
//! FileSystemType: which Filesystem should be used for e.g. browsing
enum EFileSystemType
{
    FILESYSTEM_NATIVE = 0,  // Native OS FileSystem
    FILESYSTEM_VIRTUAL  // Virtual FileSystem
};

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
    //! Opens a file based on its name
    /** Creates and returns a new IReadFile for a file in the archive.
	\param filename The file to open
	\return Returns A pointer to the created file on success,
	or 0 on failure. */
    virtual IReadFile* createAndOpenFile(const path& filename) = 0;

    //! Returns the complete file tree
    /** \return Returns the complete directory tree for the archive,
	including all files and folders */
    virtual const IFileList* getFileList() const = 0;

    //! get the archive type
    virtual E_FILE_ARCHIVE_TYPE getType() const { return EFAT_UNKNOWN; }

    //! An optionally used password string
    /** This variable is publicly accessible from the interface in order to
	avoid single access patterns to this place, and hence allow some more
	obscurity.
	*/
    core::stringc Password;
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
    virtual bool isALoadableFileFormat(const path& filename) const = 0;

    //! Check if the file might be loaded by this class
    /** This check may look into the file.
	\param file File handle to check.
	\return True if file seems to be loadable. */
    virtual bool isALoadableFileFormat(io::IReadFile* file) const = 0;

    //! Check to see if the loader can create archives of this type.
    /** Check based on the archive type.
	\param fileType The archive type to check.
	\return True if the archile loader supports this type, false if not */
    virtual bool isALoadableFileFormat(E_FILE_ARCHIVE_TYPE fileType) const = 0;

    //! Creates an archive from the filename
    /** \param filename File to use.
	\param ignoreCase Searching is performed without regarding the case
	\param ignorePaths Files are searched for without checking for the directories
	\return Pointer to newly created archive, or 0 upon error. */
    virtual IFileArchive* createArchive(const path& filename) const = 0;

    //! Creates an archive from the file
    /** \param file File handle to use.
	\param ignoreCase Searching is performed without regarding the case
	\param ignorePaths Files are searched for without checking for the directories
	\return Pointer to newly created archive, or 0 upon error. */
    virtual IFileArchive* createArchive(io::IReadFile* file) const = 0;
};

}  // end namespace io
}  // end namespace nbl

#endif
