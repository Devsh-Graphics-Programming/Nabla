// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_FILE_LIST_H_INCLUDED__
#define __C_FILE_LIST_H_INCLUDED__

#include "irr/core/core.h"
#include "IFileList.h"


namespace irr
{
namespace io
{

//! Implementation of a file list
class CFileList : public IFileList
{
    protected:
        //! Destructor
        virtual ~CFileList();
    public:
        //! Constructor
        /** \param path The path of this file archive */
        CFileList(const io::path& path);

        //! Add as a file or folder to the list
        /** \param fullPath The file name including path, up to the root of the file list.
        \param isDirectory True if this is a directory rather than a file.
        \param offset The offset where the file is stored in an archive
        \param size The size of the file in bytes.
        \param id The ID of the file in the archive which owns it */
        virtual void addItem(const io::path& fullPath, uint32_t offset, uint32_t size, bool isDirectory, uint32_t id=0);

        //! Returns the amount of files in the filelist.
        virtual uint32_t getFileCount() const override {return Files.size();}

        //!
        inline const core::vector<SFileListEntry>& getFilesReference() const {return Files;}

        //!
        virtual core::vector<SFileListEntry> getFiles() const override {return Files;}

        //!
        virtual ListCIterator findFile(ListCIterator _begin, ListCIterator _end, const io::path& filename, bool isDirectory = false) const override;

        //! Returns the base path of the file list
        virtual const io::path& getPath() const {return Path;}

    protected:
        //! Path to the file list
        io::path Path;

        //! List of files
        core::vector<SFileListEntry> Files;
};


} // end namespace irr
} // end namespace io


#endif

