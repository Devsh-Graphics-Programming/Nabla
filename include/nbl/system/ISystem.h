#ifndef __NBL_I_SYSTEM_H_INCLUDED__
#define __NBL_I_SYSTEM_H_INCLUDED__

#include "nbl/core/declarations.h"
#include "nbl/builtin/common.h"

#include <variant>

#include "nbl/system/ICancellableAsyncQueueDispatcher.h"
#include "nbl/system/IFileArchive.h"
#include "nbl/system/IFile.h"
#include "nbl/system/CFileView.h"
#include "nbl/core/util/bitflag.h"

#if defined(_NBL_PLATFORM_LINUX_)
#include <sys/sysinfo.h>
#endif

namespace nbl::system
{

class ISystemCaller : public core::IReferenceCounted // why does `ISystemCaller` need to be public?
{
protected:
    virtual ~ISystemCaller() = default;

public:  

    core::smart_refctd_ptr<IFile> createFile(core::smart_refctd_ptr<ISystem>&& sys, const std::filesystem::path& filename, core::bitflag<IFile::E_CREATE_FLAGS> flags);

    size_t read(IFile* file, void* buffer, size_t offset, size_t size)
    {
        return file->read_impl(buffer, offset, size);
    }
    size_t write(IFile* file, const void* buffer, size_t offset, size_t size)
    {
        return file->write_impl(buffer, offset, size);
    }
    bool invalidateMapping(IFile* file, size_t offset, size_t size)
    {
        return false;
    }
    bool flushMapping(IFile* file, size_t offset, size_t size)
    {
        return false;
    }
protected:
    virtual core::smart_refctd_ptr<IFile> createFile_impl(core::smart_refctd_ptr<ISystem>&& sys, const std::filesystem::path& filename, core::bitflag<IFile::E_CREATE_FLAGS> flags) = 0;
};

class ISystem : public core::IReferenceCounted
{
    private:
        friend class IFile;
        friend class ISystemCaller;
        static inline constexpr uint32_t CircularBufferSize = 256u;

        // @sadiuk add more request types if needed
        enum E_REQUEST_TYPE
        {
            ERT_CREATE_FILE,
            ERT_READ,
            ERT_WRITE
        };
        template <E_REQUEST_TYPE RT>
        struct SRequestParamsBase
        {
            static inline constexpr E_REQUEST_TYPE type = RT;
        };
        struct SRequestParams_CREATE_FILE : SRequestParamsBase<ERT_CREATE_FILE>
        {
            inline static constexpr uint32_t MAX_FILENAME_LENGTH = 4096;

            char filename[MAX_FILENAME_LENGTH] {};
            core::bitflag<IFile::E_CREATE_FLAGS> flags;
        };
        struct SRequestParams_READ : SRequestParamsBase<ERT_READ>
        {
            IFile* file;
            void* buffer;
            size_t offset;
            size_t size;
        };
        struct SRequestParams_WRITE : SRequestParamsBase<ERT_WRITE>
        {
            IFile* file;
            const void* buffer;
            size_t offset;
            size_t size;
        };
        struct SRequestType : impl::ICancellableAsyncQueueDispatcherBase::request_base_t
        {
            E_REQUEST_TYPE type;
            std::variant<
                SRequestParams_CREATE_FILE,
                SRequestParams_READ,
                SRequestParams_WRITE
            > params;
        };

        class CAsyncQueue : public ICancellableAsyncQueueDispatcher<CAsyncQueue,SRequestType,CircularBufferSize>
        {
                using base_t = ICancellableAsyncQueueDispatcher<CAsyncQueue,SRequestType,CircularBufferSize>;
                friend base_t;

            public:
                CAsyncQueue(ISystem* owner, core::smart_refctd_ptr<ISystemCaller>&& caller) : base_t(base_t::start_on_construction), m_owner(owner), m_caller(std::move(caller)) {}

                template <typename FutureType, typename RequestParams>
                void request_impl(SRequestType& req, FutureType& future, RequestParams&& params)
                {
                    req.type = params.type;
                    req.params = std::move(params);
                    base_t::associate_request_with_future(req, future);
                }

                void process_request(SRequestType& req)
                {
                    switch (req.type)
                    {
                    case ERT_CREATE_FILE:
                    {
                        auto& p = std::get<SRequestParams_CREATE_FILE>(req.params);
                        base_t::notify_future<core::smart_refctd_ptr<IFile>>(req, m_caller->createFile(core::smart_refctd_ptr<ISystem>(m_owner), p.filename, p.flags));
                    }
                        break;
                    case ERT_READ:
                    {
                        auto& p = std::get<SRequestParams_READ>(req.params);
                        base_t::notify_future<size_t>(req, m_caller->read(p.file, p.buffer, p.offset, p.size));
                    }
                        break;
                    case ERT_WRITE:
                    {
                        auto& p = std::get<SRequestParams_WRITE>(req.params);
                        base_t::notify_future<size_t>(req, m_caller->write(p.file, p.buffer, p.offset, p.size));
                    }
                        break;
                    }
                }

                void init() {}

            private:
                core::smart_refctd_ptr<ISystem> m_owner;
                core::smart_refctd_ptr<ISystemCaller> m_caller;
        };
        CAsyncQueue m_dispatcher;

        // TODO: files shall have public read/write methods, and these should be protected, then the `IFile` implementations should call these behind the scenes via a friendship
        bool readFile(future<size_t>& future, IFile* file, void* buffer, size_t offset, size_t size)
        {
            SRequestParams_READ params;
            params.buffer = buffer;
            params.file = file;
            params.offset = offset;
            params.size = size;
            m_dispatcher.request(future, params);
            return true;
        }

        bool writeFile(future<size_t>& future, IFile* file, const void* buffer, size_t offset, size_t size)
        {
            SRequestParams_WRITE params;
            params.buffer = buffer;
            params.file = file;
            params.offset = offset;
            params.size = size;
            m_dispatcher.request(future, params);
            return true;
        }

        std::pair<IFileArchive*,IFileArchive::SOpenFileParams> findFileInArchive(const system::path& path) const;
        core::smart_refctd_ptr<IFile> getFileFromArchive(const system::path& path)
        {
            auto found = findFileInArchive(path);
            auto archive = std::get<IFileArchive*>(found);
            if (!archive)
                return nullptr;
            
            auto& params = std::get<IFileArchive::SOpenFileParams>(found);
            // TODO: support passwords
            auto f =  archive->readFile(params);
            //assert(f.get()); // cannot fail
            return f;
        }

    protected:
        virtual ~ISystem() {}

        struct Loaders
        {
            core::vector<core::smart_refctd_ptr<IArchiveLoader> > vector;
            //! The key is file extension
            core::CMultiObjectCache<std::string,core::smart_refctd_ptr<IArchiveLoader>,std::vector> perFileExt;

            void pushToVector(core::smart_refctd_ptr<IArchiveLoader>&& _loader)
            {
                vector.push_back(std::move(_loader));
            }
            void eraseFromVector(decltype(vector)::const_iterator _loaderItr)
            {
                vector.erase(_loaderItr);
            }
        } m_loaders;

        core::CMultiObjectCache<system::path,core::smart_refctd_ptr<IFileArchive>> m_cachedArchiveFiles;
        //core::CMultiObjectCache<system::path, system::path> m_cachedPathAliases;

    public:
        template <typename T>
        using future_t = CAsyncQueue::future_t<T>;

        explicit ISystem(core::smart_refctd_ptr<ISystemCaller>&& caller);

        uint32_t addArchiveLoader(core::smart_refctd_ptr<IArchiveLoader>&& loader)
        {
            const char** exts = loader->getAssociatedFileExtensions();
            uint32_t i = 0u;
            while (const char* e = exts[i++])
                m_loaders.perFileExt.insert(e, core::smart_refctd_ptr(loader));
            m_loaders.pushToVector(std::move(loader));

            return m_loaders.vector.size() - 1u;
        }

        //
        bool exists(const system::path& filename, core::bitflag<IFile::E_CREATE_FLAGS> flags) const
        {
        #ifdef _NBL_EMBED_BUILTIN_RESOURCES_
            std::pair<const uint8_t*, size_t> found = nbl::builtin::get_resource_runtime(filename.string());
            if (found.first && found.second)
                return true;
        #endif
            // filename too long
            if (filename.string().size() >= sizeof(SRequestParams_CREATE_FILE::filename))
                return false;
            // archive file
            if (flags.value&IFile::ECF_READ)
            if (std::get<IFileArchive*>(findFileInArchive(filename)))
                return true;
            // regular file
            return std::filesystem::exists(filename);
        }

        //
        bool createFile(future_t<core::smart_refctd_ptr<IFile>>& future, const std::filesystem::path& filename, core::bitflag<IFile::E_CREATE_FLAGS> flags)
        {
            SRequestParams_CREATE_FILE params;
            if (filename.string().size() >= sizeof(params.filename))
                return false;

            strcpy(params.filename, filename.string().c_str());
            params.flags = flags;
        
            m_dispatcher.request(future, params);

            return true;
        }

        inline core::smart_refctd_ptr<IFile> loadBuiltinData(const std::string& builtinPath)
        {
         #ifdef _NBL_EMBED_BUILTIN_RESOURCES_
            std::pair<const uint8_t*, size_t> found = nbl::builtin::get_resource_runtime(builtinPath);
            if (found.first && found.second)
            {
                auto fileView = core::make_smart_refctd_ptr<CFileView<VirtualAllocator>>(core::smart_refctd_ptr<ISystem>(this), builtinPath, core::bitflag<IFile::E_CREATE_FLAGS>(IFile::ECF_READ) | IFile::ECF_WRITE, found.second);
                fileView->write_impl(found.first, 0, found.second);
                return fileView;
            }
            return nullptr;
        #else
            constexpr auto pathPrefix = "nbl/builtin/";
            auto pos = builtinPath.find(pathPrefix);
            std::string path;
            if (pos != std::string::npos)
                path = builtinResourceDirectory + builtinPath.substr(pos + strlen(pathPrefix));
            else
                path = builtinResourceDirectory + builtinPath;

            future_t<core::smart_refctd_ptr<IFile>> fut;
            createFile(future, path.c_str(), core::bitflag<IFile::E_CREATE_FLAGS>(IFile::ECF_READ) :: IFile::ECF_MAPPABLE);
            auto file = fut.get();
            if (file.get())
            {
                return file;
            }
            return nullptr;
    #endif
        }
        //! Compile time resource ID
        template<typename StringUniqueType>
        inline core::smart_refctd_ptr<IFile> loadBuiltinData()
        {
    #ifdef _NBL_EMBED_BUILTIN_RESOURCES_
            std::pair<const uint8_t*, size_t> found = nbl::builtin::get_resource<StringUniqueType>();
            if (found.first && found.second)
            {
                auto fileView = core::make_smart_refctd_ptr<CFileView<VirtualAllocator>>(core::smart_refctd_ptr<ISystem>(this), StringUniqueType::value, core::bitflag<IFile::E_CREATE_FLAGS>(IFile::ECF_READ) | IFile::ECF_WRITE, found.second);
                fileView->write_impl(found.first, 0, found.second);
                return fileView;
            }
            return nullptr;
    #else
            return loadBuiltinData(StringUniqueType::value);
    #endif
        }

        //! Warning: blocking call
        core::smart_refctd_ptr<IFileArchive> openFileArchive(const std::filesystem::path& filename, const std::string_view& password = "")
        {
            future_t<core::smart_refctd_ptr<IFile>> future;
            if (!createFile(future, filename, core::bitflag<IFile::E_CREATE_FLAGS>(IFile::ECF_READ) | IFile::ECF_MAPPABLE))
                return nullptr;

            auto file = future.get();
            if (file.get() == nullptr) return nullptr;
            return openFileArchive(std::move(file), password);
        }
        core::smart_refctd_ptr<IFileArchive> openFileArchive(core::smart_refctd_ptr<IFile>&& file, const std::string_view& password = "")
        {
            if (file->getFlags() & IFile::ECF_READ == 0)
                return nullptr;

            auto ext = system::extension_wo_dot(file->getFileName());
            auto loaders = m_loaders.perFileExt.findRange(ext);
            for (auto& loader : loaders)
            {
                auto arch = loader.second->createArchive(std::move(file), password);
                if (arch.get() == nullptr) continue;
                return arch;
            }
            return nullptr;
        }

        void mount(core::smart_refctd_ptr<IFileArchive>&& archive, const system::path& pathAlias = "")
        {
            auto path = std::filesystem::absolute(archive->asFile()->getFileName()).generic_string();
            m_cachedArchiveFiles.insert(path, std::move(archive));
            if (!pathAlias.empty())
            {
                m_cachedArchiveFiles.insert(pathAlias, std::move(archive));
            }
        }
        void unmount(const IFileArchive* archive, const system::path& pathAlias)
        {
            assert(false); // TODO(dan): i swear i'll get to it at some point
        }

        /*
            Returns true if the path is writable (e.g. if p is a path inside an archive the function will return true).
            The path existence is not checked.
        */
        bool isPathReadOnly(const system::path& p)
        {
            auto curPath = p;
            while (!curPath.empty() && curPath.parent_path() != curPath)
            {
                auto archives = m_cachedArchiveFiles.findRange(curPath);
                if (!archives.empty()) return true;

                curPath = curPath.parent_path().generic_string();
            }
            return false;
        }

        bool createDirectory(const system::path& p)
        {
            return std::filesystem::create_directories(p);
        }

        bool deleteDirectory(const system::path& p)
        {
            return std::filesystem::remove(p);
        }

        bool moveFileOrDirectory(const system::path oldPath, const system::path newPath)
        {
            std::error_code ec;
            std::filesystem::rename(oldPath, newPath, ec);
            return static_cast<bool>(ec);
        }
        /*
            Recursively lists all files and directories in the directory.
        */
        core::vector<system::path> listFilesInDirectory(const system::path& p)
        {
            if (isPathReadOnly(p))
            {
                auto curPath = p;
                while (!curPath.empty() && curPath.parent_path() != curPath)
                {
                    auto archives = m_cachedArchiveFiles.findRange(curPath);
                    for (auto& arch : archives)
                    {
                        auto rel = std::filesystem::relative(p, arch.first);
                        auto res =  arch.second->listAssets(rel.generic_string().c_str());
                        std::for_each(res.begin(), res.end(), [&arch](system::path& p) {p = arch.first / p; });
                        return res;
                    }

                    curPath = curPath.parent_path().generic_string();
                }
                return {};
            }
            else
            {
                uint32_t fileCount = std::distance(std::filesystem::recursive_directory_iterator(p), std::filesystem::recursive_directory_iterator{});
                core::vector<system::path> res;
                res.reserve(fileCount);
                for (auto entry : std::filesystem::recursive_directory_iterator(p))
                {
                    res.push_back(entry.path());
                }
                return res;
            }
        }


        bool isDirectory(const system::path& p)
        {
            if (isPathReadOnly(p))
            {
                return p.extension() == ""; // TODO: this is a temporary decision until i figure out how to check if a file is directory in androi APK
            }
            else
            {
                return std::filesystem::is_directory(p);
            }
        }

        /*
            Recursively copy a directory or a file from one place to another.
            from - a path to the source file or directory. Must exist. Can be both readonly and mutable path.
            to - a path to the destination file or directory. Must be mutable path (isPathReadonly(to) must be false).
        */
        bool copy(const system::path& from, const system::path& to)
        {
            auto copyFile = [this](const system::path& from, const system::path& to)
            {
                future_t<core::smart_refctd_ptr<IFile>> readFileFut, writeFileFut;

                createFile(readFileFut, from, IFile::ECF_READ);
                createFile(writeFileFut, to, IFile::ECF_WRITE);
                auto readF = readFileFut.get();
                auto writeF = writeFileFut.get();

                char* fileData = (char*)malloc(readF->getSize());
                readF->read_impl(fileData, 0, readF->getSize());

                writeF->write_impl(fileData, 0, readF->getSize());

                free(fileData);
            };
            if (isPathReadOnly(from))
            {
                if (isPathReadOnly(to)) return false;
                if (isDirectory(from))
                {
                    auto allFiles = listFilesInDirectory(from);
                    for (const auto& file : allFiles)
                    {
                        auto relative = std::filesystem::relative(file, from);
                        system::path targetName = (to / relative).generic_string();
                        std::filesystem::create_directories(targetName.parent_path());
                        if (!isDirectory(targetName))
                        {
                            copyFile(file, targetName);
                        }
                    }
                }
                else
                {
                    copyFile(from, to);
                }
                return true;
            }
            else
            {
                std::error_code error;
                std::filesystem::copy(from, to, error);
                return static_cast<bool>(error);
            }
        }

        struct SystemMemory
        {
            uint32_t totalMemory = {};
            uint32_t availableMemory = {};
        };

        static inline SystemMemory getSystemMemory()
        {
            SystemMemory systemMemory;

            #if defined(_NBL_PLATFORM_WINDOWS_)
            MEMORYSTATUS memoryStatus;
            memoryStatus.dwLength = sizeof(MEMORYSTATUS);

            GlobalMemoryStatus(&memoryStatus);

            systemMemory.totalMemory = (uint32_t)(memoryStatus.dwTotalPhys >> 10);
            systemMemory.availableMemory = (uint32_t)(memoryStatus.dwAvailPhys >> 10);
            #elif defined(_NBL_PLATFORM_LINUX_)
            #if defined(_SC_PHYS_PAGES) && defined(_SC_AVPHYS_PAGES)
            sysinfo linuxSystemInfo;
            assert(sysinfo(&linuxSystemInfo));

            systemMemory.totalMemory = linuxSystemInfo.totalram;
            systemMemory.availableMemory = linuxSystemInfo.freeram;
            #endif
            #elif defined(_NBL_PLATFORM_ANDROID_)
            // @sadiuk TODO
            #elif defined(_NBL_PLATFORM_OSX_) 
            // TODO: implement for OSX
            #endif

            return systemMemory;
        }
};

template<typename T>
class future : public ISystem::future_t<T> {};
}

#endif
