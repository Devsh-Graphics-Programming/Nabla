// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_I_FILE_ARCHIVE_H_INCLUDED__
#define __NBL_I_FILE_ARCHIVE_H_INCLUDED__

#include "nbl/system/IFile.h"
#include "nbl/system/CFileView.h"
#include "nbl/system/IFileViewAllocator.h"
#include "nbl/system/CFileViewVirtualAllocatorWin32.h"
namespace nbl
{
namespace system
{
template<typename T>
class CInnerArchiveFile : public CFileView<T>
{
    std::atomic_flag* alive;

public:
    CInnerArchiveFile(CFileView<T>* arch, std::atomic_flag* _flag)
        : CFileView<T>(std::move(*arch)), alive(_flag)
    {
    }
    ~CInnerArchiveFile() = default;

    static void* operator new(size_t size) noexcept
    {
        assert(false);
        return nullptr;
    }
    static void* operator new[](size_t size) noexcept
    {
        assert(false);
        return nullptr;
    }
    static void* operator new(size_t size, void* ptr, std::atomic_flag* alive)
    {
        alive->test_and_set();
        return ::operator new(size, ptr);
    }
    static void operator delete(void* ptr) noexcept
    {
        static_cast<CInnerArchiveFile*>(ptr)->alive->clear();
        static_cast<CInnerArchiveFile*>(ptr)->alive->notify_one();
    }
    static void operator delete[](void* ptr) noexcept
    {
        assert(false);
    }
};

//! The FileArchive manages archives and provides access to files inside them.
class IFileArchive : public core::IReferenceCounted
{
    static inline constexpr size_t SIZEOF_INNER_ARCHIVE_FILE = std::max(sizeof(CInnerArchiveFile<CPlainHeapAllocator>), sizeof(CInnerArchiveFile<VirtualAllocator>));
    static inline constexpr size_t ALIGNOF_INNER_ARCHIVE_FILE = std::max(alignof(CInnerArchiveFile<CPlainHeapAllocator>), alignof(CInnerArchiveFile<VirtualAllocator>));

protected:
    enum E_ALLOCATOR_TYPE
    {
        EAT_NONE = 0,
        EAT_NULL,
        EAT_VIRTUAL_ALLOC,
        EAT_MALLOC
    };
    core::smart_refctd_ptr<IFile> m_file;
    system::logger_opt_smart_ptr m_logger;

    // TODO: This one is only required here because CFileView needs ISystem, must be removed when that's fixed
    core::smart_refctd_ptr<ISystem> m_system;

public:
    IFileArchive(core::smart_refctd_ptr<IFile>&& file, core::smart_refctd_ptr<ISystem>&& system, system::logger_opt_smart_ptr&& logger)
        : m_file(std::move(file)), m_system(std::move(system)), m_logger(std::move(logger)) {}
    ~IFileArchive()
    {
        _NBL_ALIGNED_FREE(m_filesBuffer);
        _NBL_ALIGNED_FREE(m_fileFlags);
    }

    // List all files and directories in a specific dir of the archive
    virtual core::vector<system::path> listAssets(const char* asset_path)
    {
        constexpr auto isSubDir = [](path p, path root) -> bool {
            while(p != path())
            {
                if(p == root)
                {
                    return true;
                }
                p = p.parent_path();
            }
            return false;
        };
        core::vector<path> res;
        for(auto& entry : m_files)
        {
            if(isSubDir(entry.fullName, asset_path))
            {
                res.push_back(entry.fullName);
            }
        }
        return res;
    }

    //! An entry in a list of files, can be a folder or a file.
    struct SFileListEntry
    {
        //! The name of the file
        /** If this is a file or folder in the virtual filesystem and the archive
		was created with the ignoreCase flag then the file name will be lower case. */
        system::path name;

        //! The name of the file including the path
        /** If this is a file or folder in the virtual filesystem and the archive was
		created with the ignoreDirs flag then it will be the same as Name. */
        system::path fullName;

        //! The size of the file in bytes
        uint32_t size;

        //! The ID of the file in an archive
        /** This is used to link the FileList entry to extra info held about this
		file in an archive, which can hold things like data offset and CRC. */
        uint32_t ID;

        //! FileOffset inside an archive
        uint32_t offset;

        E_ALLOCATOR_TYPE allocatorType;

        //! The == operator is provided so that CFileList can slowly search the list!
        inline bool operator==(const struct SFileListEntry& other) const
        {
            return core::strcmpi(fullName.string(), other.fullName.string()) == 0;
        }

        //! The < operator is provided so that CFileList can sort and quickly search the list.
        inline bool operator<(const struct SFileListEntry& other) const
        {
            return fullName < other.fullName;
        }
    };
    struct SOpenFileParams
    {
        std::filesystem::path filename;
        std::filesystem::path absolutePath;
        std::string_view password;
    };

    std::mutex fileMutex;
    std::atomic_flag* m_fileFlags = nullptr;
    std::byte* m_filesBuffer = nullptr;

    core::smart_refctd_ptr<IFile> readFile(const SOpenFileParams& params)
    {
        auto index = getIndexByPath(params.filename);
        if(index == -1)
            return nullptr;
        switch(this->getFileType(index))
        {
            case EAT_NULL:
                return getFile_impl<CNullAllocator>(params, index);
                break;
            case EAT_MALLOC:
                return getFile_impl<CPlainHeapAllocator>(params, index);
                break;
            case EAT_VIRTUAL_ALLOC:
                return getFile_impl<VirtualAllocator>(params, index);
                break;
        }
        assert(false);
        return nullptr;
    }
    virtual core::smart_refctd_ptr<IFile> readFile_impl(const SOpenFileParams& params) = 0;
    int32_t getIndexByPath(const system::path& p)
    {
        for(int i = 0; i < m_files.size(); ++i)
        {
            if(p == m_files[i].fullName)
                return i;
        }
        return -1;
    }
    E_ALLOCATOR_TYPE getFileType(uint32_t index)
    {
        return m_files[index].allocatorType;
    }
    template<class Allocator>
    core::smart_refctd_ptr<CInnerArchiveFile<Allocator>> getFile_impl(const SOpenFileParams& params, const uint32_t index)
    {
        std::unique_lock lock(fileMutex);

        auto* file = reinterpret_cast<CInnerArchiveFile<Allocator>*>(m_filesBuffer + index * SIZEOF_INNER_ARCHIVE_FILE);
        //  intentionally calling grab() on maybe-not-existing object
        const auto oldRefcount = file->grab();

        if(oldRefcount == 0)  // need to construct
        {
            m_fileFlags[index].wait(true);  //what should the param of wait be?
            new(file, &m_fileFlags[index]) CInnerArchiveFile<Allocator>(static_cast<CFileView<Allocator>*>(readFile_impl(params).get()), &m_fileFlags[index]);
        }
        return core::smart_refctd_ptr<CInnerArchiveFile<Allocator>>(file, core::dont_grab);
    }

    const core::vector<SFileListEntry>& getArchivedFiles() const { return m_files; }
    IFile* asFile() { return m_file.get(); }

protected:
    void setFlagsVectorSize(size_t fileCount);

    virtual void addItem(const system::path& fullPath, uint32_t offset, uint32_t size, E_ALLOCATOR_TYPE allocatorType, uint32_t id = 0)
    {
        SFileListEntry entry;
        entry.ID = id ? id : m_files.size();
        entry.offset = offset;
        entry.size = size;
        entry.name = fullPath;
        entry.allocatorType = allocatorType;
        entry.fullName = entry.name;

        core::deletePathFromFilename(entry.name);

        m_files.insert(std::lower_bound(m_files.begin(), m_files.end(), entry), entry);
    }
    size_t read_blocking(system::IFile* file, void* buffer, size_t offset, size_t size)
    {
        return file->read_impl(buffer, offset, size);
    }
    size_t write_blocking(system::IFile* file, void* buffer, size_t offset, size_t size)
    {
        return file->write_impl(buffer, offset, size);
    }
    core::vector<SFileListEntry> m_files;
};

class IArchiveLoader : public core::IReferenceCounted
{
protected:
    system::logger_opt_smart_ptr m_logger;

    // TODO: This one is only required here because CFileView needs ISystem, must be removed when that's fixed
    core::smart_refctd_ptr<ISystem> m_system;

public:
    IArchiveLoader(core::smart_refctd_ptr<ISystem>&& system, system::logger_opt_smart_ptr&& logger)
        : m_logger(std::move(logger)), m_system(std::move(system)) {}
    //! Check if the file might be loaded by this class
    /** This check may look into the file.
	\param file File handle to check.
	\return True if file seems to be loadable. */
    virtual bool isALoadableFileFormat(IFile* file) const = 0;

    //! Returns an array of string literals terminated by nullptr
    virtual const char** getAssociatedFileExtensions() const = 0;

    //! Creates an archive from the file
    /** \param file File handle to use.
	\return Pointer to newly created archive, or 0 upon error. */
    core::smart_refctd_ptr<IFileArchive> createArchive(core::smart_refctd_ptr<IFile>&& file, const std::string_view& password = "") const
    {
        if(!(file->getFlags() & IFile::ECF_READ))
            return nullptr;

        return createArchive_impl(std::move(file), password);
    }

protected:
    virtual core::smart_refctd_ptr<IFileArchive> createArchive_impl(core::smart_refctd_ptr<IFile>&& file, const std::string_view& password) const = 0;
};

}  // end namespace system
}  // end namespace nbl

#endif
