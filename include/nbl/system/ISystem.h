#ifndef _NBL_SYSTEM_I_SYSTEM_H_INCLUDED_
#define _NBL_SYSTEM_I_SYSTEM_H_INCLUDED_


#include "nbl/builtin/common.h"

#include "nbl/core/declarations.h"
#include "nbl/core/util/bitflag.h"

#include <variant>

#include "nbl/system/ICancellableAsyncQueueDispatcher.h"
#include "nbl/system/IFileArchive.h"


namespace nbl::system
{

class IFile;
class ISystemFile;

class ISystem : public core::IReferenceCounted
{
    protected:
        class ICaller;
    private:
        // add more request types if needed
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
            IFileBase::E_CREATE_FLAGS flags;
        };
        struct SRequestParams_READ : SRequestParamsBase<ERT_READ>
        {
            ISystemFile* file;
            void* buffer;
            size_t offset;
            size_t size;
        };
        struct SRequestParams_WRITE : SRequestParamsBase<ERT_WRITE>
        {
            ISystemFile* file;
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
        static inline constexpr uint32_t CircularBufferSize = 256u;
        class CAsyncQueue : public ICancellableAsyncQueueDispatcher<CAsyncQueue,SRequestType,CircularBufferSize>
        {
                using base_t = ICancellableAsyncQueueDispatcher<CAsyncQueue,SRequestType,CircularBufferSize>;
                //friend base_t;

            public:
                CAsyncQueue(core::smart_refctd_ptr<ICaller>&& caller) : base_t(base_t::start_on_construction), m_caller(std::move(caller)) {}

                template <typename FutureType, typename RequestParams>
                void request_impl(SRequestType& req, FutureType& future, RequestParams&& params)
                {
                    req.type = params.type;
                    req.params = std::move(params);
                    base_t::associate_request_with_future(req, future);
                }

                void process_request(SRequestType& req);

                void init() {}

            private:
                core::smart_refctd_ptr<ICaller> m_caller;
        };
        friend class ISystemFile;
        CAsyncQueue m_dispatcher;

    public:
        template <typename T>
        struct future_t : public CAsyncQueue::future_t<T>
        {
                friend class ISystem;
                friend class IFutureManipulator;
        };
        class IFutureManipulator
        {
            protected:
                //template <typename T>
                //inline void fake_notify(future_t<T>& future, T&& value) const
                inline void fake_notify(future_t<size_t>& future, const size_t value) const
                {
                    future.notify(value);
                }
        };

        // [[deprecated]] use `createFile` instead
        core::smart_refctd_ptr<IFile> loadBuiltinData(const std::string& builtinPath) const;

        //! Compile time resource ID
        template<typename StringUniqueType>
        inline core::smart_refctd_ptr<IFile> loadBuiltinData() const
        {
        #ifdef _NBL_EMBED_BUILTIN_RESOURCES_
            return impl_loadEmbeddedBuiltinData(StringUniqueType::value,nbl::builtin::get_resource<StringUniqueType>());
        #else
            return loadBuiltinData(StringUniqueType::value);
        #endif
        }

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
        virtual bool isDirectory(const system::path& p) const
        {
            if (isPathReadOnly(p))
                return p.extension()==""; // TODO: this is a temporary decision until we figure out how to check if a file is directory in android APK
            else
                return std::filesystem::is_directory(p);
        }

        /*
            Recursively lists all files and directories in the directory.
        */
        core::vector<system::path> listFilesInDirectory(const system::path& p) const;

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
        bool copy(const system::path& from, const system::path& to);

        //
        void createFile(future_t<core::smart_refctd_ptr<IFile>>& future, std::filesystem::path filename, const core::bitflag<IFileBase::E_CREATE_FLAGS> flags);
        
        // Create a IFileArchive from a IFile
        core::smart_refctd_ptr<IFileArchive> openFileArchive(core::smart_refctd_ptr<IFile>&& file, const std::string_view& password="");
        //! A utility method. Warning: blocking call
        core::smart_refctd_ptr<IFileArchive> openFileArchive(const std::filesystem::path& filename, const std::string_view& password = "")
        {
            future_t<core::smart_refctd_ptr<IFile>> future;
            createFile(future, filename, core::bitflag<IFileBase::E_CREATE_FLAGS>(IFileBase::ECF_READ)|IFileBase::ECF_MAPPABLE);

            auto file = future.get();
            if (!file)
                return nullptr;

            return openFileArchive(std::move(file),password);
        }

        // After opening and archive, you must mount it if you want the global path lookup to work seamlessly.
        void mount(core::smart_refctd_ptr<IFileArchive>&& archive, const system::path& pathAlias="");
        void unmount(const IFileArchive* archive, const system::path& pathAlias="");

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

        //
        core::smart_refctd_ptr<IFile> impl_loadEmbeddedBuiltinData(const std::string& builtinPath, const std::pair<const uint8_t*,size_t>& found) const;

        // given an absolute `path` find the archive it belongs to
        std::pair<IFileArchive*,IFileArchive::SOpenFileParams> findFileInArchive(const system::path& _path) const;

        //
        core::smart_refctd_ptr<IFile> getFileFromArchive(const system::path& path);


        //
        struct Loaders
        {
            core::vector<core::smart_refctd_ptr<IArchiveLoader> > vector;
            //! The key is file extension
            core::CMultiObjectCache<std::string,core::smart_refctd_ptr<IArchiveLoader>,std::vector> perFileExt;
        } m_loaders;
        //
        core::CMultiObjectCache<system::path,core::smart_refctd_ptr<IFileArchive>> m_cachedArchiveFiles;
};

}

#endif
