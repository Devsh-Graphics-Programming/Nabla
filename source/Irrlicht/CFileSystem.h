// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_FILE_SYSTEM_H_INCLUDED__
#define __C_FILE_SYSTEM_H_INCLUDED__

#include "IFileSystem.h"

namespace irr
{
namespace io
{

	class CZipReader;
	class CPakReader;
	class CMountPointReader;

/*!
	FileSystem which uses normal files and one zipfile
*/
class CFileSystem : public IFileSystem
{
    protected:
        //! destructor
        virtual ~CFileSystem();
    public:

        //! constructor
        CFileSystem();

        //! opens a file for read access
        virtual IReadFile* createAndOpenFile(const io::path& filename);

        //! Creates an IReadFile interface for accessing memory like a file.
        virtual IReadFile* createMemoryReadFile(const void* contents, size_t len, const io::path& fileName) override;

        //! Creates an IReadFile interface for accessing files inside files
        virtual IReadFile* createLimitReadFile(const io::path& fileName, IReadFile* alreadyOpenedFile, const size_t& pos, const size_t& areaSize);

        //! Creates an IWriteFile interface for accessing memory like a file.
        virtual IWriteFile* createMemoryWriteFile(size_t len, const io::path& fileName) override;

        //! Opens a file for write access.
        virtual IWriteFile* createAndWriteFile(const io::path& filename, bool append=false);

        //! Adds an archive to the file system.
        virtual bool addFileArchive(const io::path& filename,
                bool ignoreCase = true, bool ignorePaths = true,
                E_FILE_ARCHIVE_TYPE archiveType = EFAT_UNKNOWN,
                const core::stringc& password="",
                IFileArchive** retArchive = 0);

        //! Adds an archive to the file system.
        virtual bool addFileArchive(IReadFile* file, bool ignoreCase=true,
                bool ignorePaths=true,
                E_FILE_ARCHIVE_TYPE archiveType=EFAT_UNKNOWN,
                const core::stringc& password="",
                IFileArchive** retArchive = 0);

        //! Adds an archive to the file system.
        virtual bool addFileArchive(IFileArchive* archive);

        //! move the hirarchy of the filesystem. moves sourceIndex relative up or down
        virtual bool moveFileArchive(uint32_t sourceIndex, int32_t relative);

        //! Adds an external archive loader to the engine.
        virtual void addArchiveLoader(IArchiveLoader* loader);

        //! Returns the total number of archive loaders added.
        virtual uint32_t getArchiveLoaderCount() const;

        //! Gets the archive loader by index.
        virtual IArchiveLoader* getArchiveLoader(uint32_t index) const;

        //! gets the file archive count
        virtual uint32_t getFileArchiveCount() const;

        //! gets an archive
        virtual IFileArchive* getFileArchive(uint32_t index);

        //! removes an archive from the file system.
        virtual bool removeFileArchive(uint32_t index);

        //! removes an archive from the file system.
        virtual bool removeFileArchive(const io::path& filename);

        //! Removes an archive from the file system.
        virtual bool removeFileArchive(const IFileArchive* archive);

        //! Returns the string of the current working directory
        virtual const io::path& getWorkingDirectory();

        //! Changes the current Working Directory to the string given.
        //! The string is operating system dependent. Under Windows it will look
        //! like this: "drive:\directory\sudirectory\"
        virtual bool changeWorkingDirectoryTo(const io::path& newDirectory);

        //! Converts a relative path to an absolute (unique) path, resolving symbolic links
        virtual io::path getAbsolutePath(const io::path& filename) const;

        //! Get the relative filename, relative to the given directory
        virtual path getRelativeFilename(const path& filename, const path& directory) const;

        virtual EFileSystemType setFileListSystem(EFileSystemType listType);

        //! Creates a list of files and directories in the current working directory
        //! and returns it.
        virtual IFileList* createFileList();

        //! Creates an empty filelist
        virtual IFileList* createEmptyFileList(const io::path& path, bool ignoreCase, bool ignorePaths);

        //! determines if a file exists and would be able to be opened.
        virtual bool existFile(const io::path& filename) const;

    private:

        // don't expose, needs refactoring
        bool changeArchivePassword(const path& filename,
                const core::stringc& password,
                IFileArchive** archive = 0);

        //! Currently used FileSystemType
        EFileSystemType FileSystemType;
        //! WorkingDirectory for Native and Virtual filesystems
        io::path WorkingDirectory [2];
        //! currently attached ArchiveLoaders
        core::vector<IArchiveLoader*> ArchiveLoader;
        //! currently attached Archives
        core::vector<IFileArchive*> FileArchives;
};


} // end namespace irr
} // end namespace io

#endif

