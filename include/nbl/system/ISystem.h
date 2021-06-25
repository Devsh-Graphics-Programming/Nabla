#ifndef __NBL_I_SYSTEM_H_INCLUDED__
#define __NBL_I_SYSTEM_H_INCLUDED__

#include <variant>
#include "nbl/core/IReferenceCounted.h"
#include "nbl/system/ICancellableAsyncQueueDispatcher.h"
#include "nbl/system/IFileArchive.h"
#include "nbl/system/IFile.h"
#include "CObjectCache.h"

namespace nbl {
namespace system
{

class ISystem final : public core::IReferenceCounted
{
public:
    class ISystemCaller : public core::IReferenceCounted
    {
    protected:
        virtual ~ISystemCaller() = default;

    public:
        virtual core::smart_refctd_ptr<IFile> createFile(ISystem* sys, const std::filesystem::path& filename, IFile::E_CREATE_FLAGS flags) = 0;
        virtual size_t read(IFile* file, void* buffer, size_t offset, size_t size) = 0;
        virtual size_t write(IFile* file, const void* buffer, size_t offset, size_t size) = 0;
        virtual bool invalidateMapping(IFile* file, size_t offset, size_t size) = 0;
        virtual bool flushMapping(IFile* file, size_t offset, size_t size) = 0;
    };

private:
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
        IFile::E_CREATE_FLAGS flags;
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

    class CAsyncQueue : public ICancellableAsyncQueueDispatcher<CAsyncQueue, SRequestType, CircularBufferSize>
    {
        using base_t = ICancellableAsyncQueueDispatcher<CAsyncQueue, SRequestType, CircularBufferSize>;
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
                base_t::notify_future<core::smart_refctd_ptr<IFile>>(req, m_caller->createFile(m_owner, p.filename, p.flags));
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
    public:
        void init() {}
    private:
        ISystem* m_owner;
        core::smart_refctd_ptr<ISystemCaller> m_caller;
    };

    struct Loaders {
        core::vector<core::smart_refctd_ptr<IArchiveLoader> > vector;
        //! The key is file extension
        core::CMultiObjectCache<std::string, core::smart_refctd_ptr<IArchiveLoader>, std::vector> perFileExt;

        void pushToVector(core::smart_refctd_ptr<IArchiveLoader>&& _loader)
        {
            vector.push_back(std::move(_loader));
        }
        void eraseFromVector(decltype(vector)::const_iterator _loaderItr)
        {
            vector.erase(_loaderItr);
        }
    } m_loaders;

    CAsyncQueue m_dispatcher;

public:
    template <typename T>
    using future_t = CAsyncQueue::future_t<T>;

    explicit ISystem(core::smart_refctd_ptr<ISystemCaller>&& caller) : m_dispatcher(this, std::move(caller))
    {
        // add all possible archive loaders to m_loaders containers here
        // @sadiuk see IAssetManager for reference
    }

    uint32_t addArchiveLoader(core::smart_refctd_ptr<IArchiveLoader>&& loader)
    {
        const char** exts = loader->getAssociatedFileExtensions();
        uint32_t i = 0u;
        while (const char* e = exts[i++])
            m_loaders.perFileExt.insert(e, core::smart_refctd_ptr(loader));
        m_loaders.pushToVector(std::move(loader));

        return m_loaders.vector.size() - 1u;
    }

    bool createFile(future_t<core::smart_refctd_ptr<IFile>>& future, const std::filesystem::path& filename, IFile::E_CREATE_FLAGS flags)
    {
        SRequestParams_CREATE_FILE params;
        if (filename.string().size() >= sizeof(params.filename))
            return false;

        strcpy(params.filename, filename.string().c_str());
        params.flags = flags;
        
        m_dispatcher.request(future, params);

        return true;
    }

    bool readFile(future_t<uint32_t>& future, IFile* file, void* buffer, size_t offset, size_t size)
    {
        SRequestParams_READ params;
        params.buffer = buffer;
        params.file = file;
        params.offset = offset;
        params.size = size;
        m_dispatcher.request(future, params);
        return true;
    }

    bool writeFile(future_t<uint32_t>& future, IFile* file, const void* buffer, size_t offset, size_t size)
    {
        SRequestParams_WRITE params;
        params.buffer = buffer;
        params.file = file;
        params.offset = offset;
        params.size = size;
        m_dispatcher.request(future, params);
        return true;
    }

    // @sadiuk add more methods taken from IFileSystem and IOSOperator
    // and implement via m_dispatcher and ISystemCaller if needed
    // (any system calls should take place in ISystemCaller which is called by CAsyncQueue and nothing else)

    virtual void seek(IFile* file, uint32_t pos) = 0;
    virtual void read(IFile* file, char* outData, size_t count) const;
    virtual size_t getPos(IFile* file) const = 0;


    inline core::smart_refctd_ptr<asset::ICPUBuffer> loadBuiltinData(const std::string& builtinPath)
    {
#ifdef _NBL_EMBED_BUILTIN_RESOURCES_
        std::pair<const uint8_t*, size_t> found = nbl::builtin::get_resource_runtime(builtinPath);
        if (found.first && found.second)
        {
            auto returnValue = core::make_smart_refctd_ptr<asset::ICPUBuffer>(found.second);
            memcpy(returnValue->getPointer(), found.first, returnValue->getSize());
            return returnValue;
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

        auto file = this->createAndOpenFile(path.c_str());
        if (file)
        {
            auto retval = core::make_smart_refctd_ptr<asset::ICPUBuffer>(file->getSize());
            file->read(retval->getPointer(), file->getSize());
            file->drop();
            return retval;
        }
        return nullptr;
#endif
    }
    //! Compile time resource ID
    template<typename StringUniqueType>
    inline core::smart_refctd_ptr<asset::ICPUBuffer> loadBuiltinData()
    {
#ifdef _NBL_EMBED_BUILTIN_RESOURCES_
        std::pair<const uint8_t*, size_t> found = nbl::builtin::get_resource<StringUniqueType>();
        if (found.first && found.second)
        {
            auto returnValue = core::make_smart_refctd_ptr<asset::ICPUBuffer>(found.second);
            memcpy(returnValue->getPointer(), found.first, returnValue->getSize());
            return returnValue;
        }
        return nullptr;
#else
        return loadBuiltinData(StringUniqueType::value);
#endif
    }

    //! Warning: blocking call
    core::smart_refctd_ptr<IFileArchive> createFileArchive(const std::filesystem::path& filename)
    {
        future_t<core::smart_refctd_ptr<IFile>> future;
        if (!createFile(future, filename, IFile::ECF_READ))
            return nullptr;

        auto file = std::move(future.get());

        return createFileArchive(file.get());
    }
    core::smart_refctd_ptr<IFileArchive> createFileArchive(IFile* file)
    {
        if (file->getFlags() & IFile::ECF_READ == 0)
            return nullptr;

        // @sadiuk implement in manner similar to IAssetManager::getAsset

        return nullptr;
    }
};

}
}

#endif
