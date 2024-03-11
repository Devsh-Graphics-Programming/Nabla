#ifndef _NBL_SYSTEM_I_SYSTEM_H_INCLUDED_
#define _NBL_SYSTEM_I_SYSTEM_H_INCLUDED_


#include "nbl/core/declarations.h"
#include "nbl/core/util/bitflag.h"

#include <variant>

#include "nbl/system/IFileArchive.h"
#include "nbl/system/IAsyncQueueDispatcher.h"

namespace nbl::system
{

class IFile;
class ISystemFile;

class NBL_API2 ISystem : public core::IReferenceCounted
{
    public:
        inline static constexpr uint32_t MAX_FILENAME_LENGTH = 4096;

        //! We overrride the future a little bit, to allow to put a result in it right away without asynchronocity
        template <typename T>
        struct future_t final : public impl::IAsyncQueueDispatcherBase::cancellable_future_t<T>
        {
            private:
                friend class ISystem;
                friend class IFutureManipulator;

                using base_t = impl::IAsyncQueueDispatcherBase::cancellable_future_t<T>;

                template <typename... Args>
                inline void set_result(Args&&... args)
                {
                    base_t::state.waitTransition(base_t::STATE::EXECUTING,base_t::STATE::INITIAL);
                    base_t::construct(std::forward<Args>(args)...);
                    base_t::notify();
                }
        };
        class IFutureManipulator
        {
            protected:
                inline void set_result(future_t<size_t>& future, const size_t value) const
                {
                    future.set_result(value);
                }
        };
		
		#ifndef NBL_EMBED_BUILTIN_RESOURCES
        constexpr std::string_view getBuiltinResourcesDirectoryPath()
        {
            std::string_view retval = NBL_BUILTIN_RESOURCES_DIRECTORY_PATH;
            return retval;
        }
		#endif

        //
        inline void addArchiveLoader(core::smart_refctd_ptr<IArchiveLoader>&& loader)
        {
            const char** exts = loader->getAssociatedFileExtensions();
            uint32_t i = 0u;
            while (const char* e = exts[i++])
                m_loaders.perFileExt.insert(e, core::smart_refctd_ptr(loader));
            m_loaders.vector.push_back(std::move(loader));
        }

        // `flags` is the intended usage of the file
        bool exists(const system::path& filename, const core::bitflag<IFileBase::E_CREATE_FLAGS> flags) const;

        /*
            Returns true if the path is writable (e.g. if p is a path inside an archive the function will return true).
            The path existence is not checked.
        */
        bool isPathReadOnly(const system::path& p) const;

        //
        virtual inline bool isDirectory(const system::path& p) const
        {
            if (isPathReadOnly(p))
                return p.extension()==""; // TODO: this is a temporary decision until we figure out how to check if a file is directory in android APK
            else
                return std::filesystem::is_directory(p);
        }

        /*
            Recursively lists all files and directories in the directory.
        */
        core::vector<system::path> listItemsInDirectory(const system::path& p) const;

        // can only perform operations on non-virtual filesystem paths
        bool createDirectory(const system::path& p);

        // can only perform operations on non-virtual filesystem paths
        bool deleteDirectory(const system::path& p);

        // can only perform operations on non-virtual filesystem paths
        std::error_code moveFileOrDirectory(const system::path& oldPath, const system::path& newPath);

        /*
            Recursively copy a directory or a file from one place to another.
            from - a path to the source file or directory. Must exist. Can be both readonly and mutable path.
            to - a path to the destination file or directory. Must be mutable path (isPathReadonly(to) must be false).
        */
        bool copy(const system::path& from, const system::path& to); // TODO: as a future

        //
        void createFile(
            future_t<core::smart_refctd_ptr<IFile>>& future, // creation may happen on a dedicated thread, so its async
            path filename, // absolute path within our virtual filesystem
            const core::bitflag<IFileBase::E_CREATE_FLAGS> flags, // access flags (IMPORTANT: files from most archives wont open with ECF_WRITE bit)
            const std::string_view& accessToken="" // usually password for archives, but should be SSH key for URL downloads
        );
        
        // Create a IFileArchive from a IFile
        core::smart_refctd_ptr<IFileArchive> openFileArchive(core::smart_refctd_ptr<IFile>&& file, const std::string_view& password="");
        //! A utility method. Warning: blocking call
        core::smart_refctd_ptr<IFileArchive> openFileArchive(const std::filesystem::path& filename, const std::string_view& password = "")
        {
            future_t<core::smart_refctd_ptr<IFile>> future;
            createFile(future, filename, core::bitflag<IFileBase::E_CREATE_FLAGS>(IFileBase::ECF_READ)|IFileBase::ECF_MAPPABLE);

            if (future.wait())
                return openFileArchive(future.copy(),password);
            return nullptr;
        }

        // After opening and archive, you must mount it if you want the global path lookup to work seamlessly.
        inline void mount(core::smart_refctd_ptr<IFileArchive>&& archive, const system::path& pathAlias="")
        {
            if (pathAlias.empty())
                m_cachedArchiveFiles.insert(archive->getDefaultAbsolutePath(),std::move(archive));
            else
                m_cachedArchiveFiles.insert(pathAlias,std::move(archive));
        }

        //
        inline void unmount(const IFileArchive* archive, const system::path& pathAlias = "")
        {
            auto dummy = reinterpret_cast<const core::smart_refctd_ptr<IFileArchive>&>(archive);
            if (pathAlias.empty())
                m_cachedArchiveFiles.removeObject(dummy,archive->getDefaultAbsolutePath());
            else
                m_cachedArchiveFiles.removeObject(dummy,pathAlias);
        }

        void unmountBuiltins();

        //
        struct SystemInfo
        {
            uint64_t cpuFrequencyHz = 0u;

            // in bytes
            uint64_t totalMemory = 0u;
            uint64_t availableMemory = 0u;

            uint32_t desktopResX = 0u;
            uint32_t desktopResY = 0u;

            std::string OSFullName = "Unknown";
        };
        virtual SystemInfo getSystemInfo() const = 0;
        

    protected:
        // all file operations take place serially on a dedicated thread (to make fibers possible in the future)
        class ICaller : public core::IReferenceCounted
        {
            public:
                // each per-platform backend must override this function
                virtual core::smart_refctd_ptr<ISystemFile> createFile(const std::filesystem::path& filename, const core::bitflag<IFileBase::E_CREATE_FLAGS> flags) = 0;

                // these contain some hoisted common sense checks
                bool invalidateMapping(IFile* file, size_t offset, size_t size);
                bool flushMapping(IFile* file, size_t offset, size_t size);

            protected:
                ICaller(ISystem* _system) : m_system(_system) {}
                virtual ~ICaller() = default;

                // TODO: maybe change the file type to `ISystemFile` ?
                virtual bool invalidateMapping_impl(IFile* file, size_t offset, size_t size) { assert(false); return false; } // TODO
                virtual bool flushMapping_impl(IFile* file, size_t offset, size_t size) { assert(false); return false; } // TODO

                ISystem* m_system;
        };

        //
        explicit ISystem(core::smart_refctd_ptr<ICaller>&& caller);
        virtual ~ISystem() {}

        // given an `absolutePath` find the archive it belongs to
        struct FoundArchiveFile
        {
            IFileArchive* archive;
            path pathRelativeToArchive;
        };
        FoundArchiveFile findFileInArchive(const system::path& absolutePath) const;


        //
        struct Loaders
        {
            core::vector<core::smart_refctd_ptr<IArchiveLoader> > vector;
            //! The key is file extension
            core::CMultiObjectCache<std::string,core::smart_refctd_ptr<IArchiveLoader>,std::vector> perFileExt;
        } m_loaders;
        //
        core::CMultiObjectCache<system::path,core::smart_refctd_ptr<IFileArchive>> m_cachedArchiveFiles;

    private:
        struct SRequestParams_NOOP
        {
            using retval_t = void;
            inline void operator()(core::StorageTrivializer<retval_t>* retval, ICaller* _caller) {assert(false);}
        };
        struct SRequestParams_CREATE_FILE
        {
            using retval_t = core::smart_refctd_ptr<IFile>;
            void operator()(core::StorageTrivializer<retval_t>* retval, ICaller* _caller);

            char filename[MAX_FILENAME_LENGTH] {};
            IFileBase::E_CREATE_FLAGS flags;
        };
        struct SRequestParams_READ
        {
            using retval_t = size_t;
            void operator()(core::StorageTrivializer<retval_t>* retval, ICaller* _caller);

            ISystemFile* file;
            void* buffer;
            size_t offset;
            size_t size;
        };
        struct SRequestParams_WRITE
        {
            using retval_t = size_t;
            void operator()(core::StorageTrivializer<retval_t>* retval, ICaller* _caller);

            ISystemFile* file;
            const void* buffer;
            size_t offset;
            size_t size;
        };
        struct SRequestType
        {
            std::variant<
                SRequestParams_NOOP,
                SRequestParams_CREATE_FILE,
                SRequestParams_READ,
                SRequestParams_WRITE
            > params = SRequestParams_NOOP();
        };
        static inline constexpr uint32_t CircularBufferSize = 256u;
        class NBL_API2 CAsyncQueue final : public IAsyncQueueDispatcher<CAsyncQueue,SRequestType,CircularBufferSize>
        {
                using base_t = IAsyncQueueDispatcher<CAsyncQueue,SRequestType,CircularBufferSize>;

                core::smart_refctd_ptr<ICaller> m_caller;

            public:
                inline CAsyncQueue(core::smart_refctd_ptr<ICaller>&& caller) : base_t(base_t::start_on_construction), m_caller(std::move(caller))
                {
                    //waitForInitComplete(); init is a NOOP
                }

                void process_request(base_t::future_base_t* _future_base, SRequestType& req);

                void init() {}
        };
        // friendship needed to be able to know about the request types
        friend class ISystemFile;

        CAsyncQueue m_dispatcher;
};

}

#endif
