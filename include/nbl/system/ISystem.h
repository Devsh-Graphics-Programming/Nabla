#ifndef __NBL_I_SYSTEM_H_INCLUDED__
#define __NBL_I_SYSTEM_H_INCLUDED__

#include <variant>
#include "nbl/core/IReferenceCounted.h"
#include "nbl/system/ISkippableAsyncQueueDispatcher.h"
#include "nbl/system/IFileArchive.h"
#include "nbl/system/IWriteFile.h"

namespace nbl {
namespace system
{

class ISystem final : public core::IReferenceCounted
{
public:
    class ISystemCaller : core::IReferenceCounted
    {
    protected:
        virtual ~ISystemCaller() = default;

    public:
        virtual core::smart_refctd_ptr<IReadFile> createReadFile(ISystem* sys, const std::filesystem::path& filename) = 0;
        virtual core::smart_refctd_ptr<IWriteFile> createWriteFile(ISystem* sys, const std::filesystem::path& filename) = 0;
        virtual uint32_t read(IReadFile* file, void* buffer, size_t offset, size_t size) = 0;
        virtual uint32_t write(IWriteFile* file, const void* buffer, size_t offset, size_t size) = 0;
    };

protected:
    struct SRequestType;
    class CAsyncQueue;

    class SFutureBase
    {
        friend ISystem;

    protected:
        bool valid_flag = false;
        SRequestType* request;
        CAsyncQueue* asyncQ;

    public:
        // SFuture is non-copyable and non-movable
        SFutureBase(const SFutureBase&) = delete;

        ~SFutureBase()
        {
            request->set_skip();
        }

        bool ready() const { return !request || request->ready; }
        bool valid() const { return valid_flag; }

        void wait()
        {
            asyncQ->waitForRequestCompletion(request[0]);
        }
    };

    template <typename T>
    class SFuture : public SFutureBase
    {
        alignas(T) uint8_t storage[sizeof(T)];

    public:
        ~SFuture()
        {
            if (valid_flag)
                getStorage()->~T();
        }

        T* getStorage() { return reinterpret_cast<T*>(storage); }
        T get()
        {
            wait();
            assert(valid_flag);
            valid_flag = false;
            T* ptr = getStorage();
            return std::move(ptr[0]);
        }
    };

    // @sadiuk add more request types if needed
    enum E_REQUEST_TYPE
    {
        ERT_CREATE_READ,
        ERT_CREATE_WRITE,
        ERT_READ,
        ERT_WRITE
    };
    template <E_REQUEST_TYPE RT>
    struct SRequestParamsBase
    {
        static inline constexpr E_REQUEST_TYPE type = RT;
        SFutureBase* future;
        void* retval_storage;
    };
    struct SRequestParams_CREATE_READ : SRequestParamsBase<ERT_CREATE_READ>
    {
        std::filesystem::path filename;
    };
    struct SRequestParams_CREATE_WRITE : SRequestParamsBase<ERT_CREATE_WRITE>
    {
        std::filesystem::path filename;
    };
    struct SRequestParams_READ : SRequestParamsBase<ERT_READ>
    {
        IReadFile* file;
        void* buffer;
        size_t offset;
        size_t size;
    };
    struct SRequestParams_WRITE : SRequestParamsBase<ERT_WRITE>
    {
        IWriteFile* file;
        const void* buffer;
        size_t offset;
        size_t size;
    };
    struct SRequestType : SSkippableRequestBase
    {
        SFutureBase* future = nullptr;
        void* retval_storage = nullptr;
        E_REQUEST_TYPE type;
        std::variant<
            SRequestParams_CREATE_READ,
            SRequestParams_CREATE_WRITE,
            SRequestParams_READ,
            SRequestParams_WRITE
        > params;

        void set_skip()
        {
            write_lock_guard<> lk(rwlock);
            set_skip_no_lock();
            future->request = nullptr;
        }
    };

    class CAsyncQueue : public ISkippableAsyncQueueDispatcher<CAsyncQueue, SRequestType, 256u>
    {
        using base_t = ISkippableAsyncQueueDispatcher<CAsyncQueue, SRequestType, 256u>;

    public:
        explicit CAsyncQueue(ISystem* owner, core::smart_refctd_ptr<ISystemCaller>&& caller) : m_owner(owner), m_caller(std::move(caller)) {}

        template <typename RequestParams>
        void request_impl(SRequestType& req, RequestParams&& params)
        {
            req.type = params.type;
            req.params = std::move(params);
            req.future = params.future;
            req.retval_storage = params.retval_storage;
        }

        void process_request(SRequestType& req)
        {
            switch (req.type)
            {
            case ERT_CREATE_READ:
            {
                auto& p = std::get<SRequestParams_CREATE_READ>(req.params);
                reinterpret_cast<core::smart_refctd_ptr<IReadFile>*>(req.retval_storage)[0] = m_caller->createReadFile(m_owner, p.filename);
            }
                break;
            case ERT_CREATE_WRITE:
            {
                auto& p = std::get<SRequestParams_CREATE_WRITE>(req.params);
                reinterpret_cast<core::smart_refctd_ptr<IWriteFile>*>(req.retval_storage)[0] = m_caller->createWriteFile(m_owner, p.filename);
            }
                break;
            case ERT_READ:
            {
                auto& p = std::get<SRequestParams_READ>(req.params);
                reinterpret_cast<uint32_t*>(req.retval_storage)[0] = m_caller->read(p.file, p.buffer, p.offset, p.size);
            }
                break;
            case ERT_WRITE:
            {
                auto& p = std::get<SRequestParams_WRITE>(req.params);
                reinterpret_cast<uint32_t*>(req.retval_storage)[0] = m_caller->write(p.file, p.buffer, p.offset, p.size);
            }
                break;
            }
        }

    private:
        ISystem* m_owner;
        core::smart_refctd_ptr<ISystemCaller> m_caller;
    };

    CAsyncQueue m_dispatcher;


    template <typename FutureType, typename RequestParamsType>
    void requestAndPrepareFuture(SFuture<core::smart_refctd_ptr<IReadFile>>& future, RequestParamsType&& params)
    {
        auto& req = m_dispatcher.request(std::move(params));
        future.asyncQ = &m_dispatcher;
        future.request = &req;
    }

public:
    explicit ISystem(core::smart_refctd_ptr<ISystemCaller>&& caller) : m_dispatcher(this, std::move(caller))
    {

    }

    bool createReadFile(SFuture<core::smart_refctd_ptr<IReadFile>>& future, const std::filesystem::path& filename)
    {
        SRequestParams_CREATE_READ params;
        params.filename = filename;
        params.future = static_cast<SFutureBase*>(&future);
        params.retval_storage = future.getStorage();
        
        requestAndPrepareFuture<core::smart_refctd_ptr<IReadFile>, SRequestParams_CREATE_READ>(future, std::move(params));
    }

    // @sadiuk Implement the rest of functions in manner analogous to createReadFile

    bool createWriteFile(SFuture<core::smart_refctd_ptr<IWriteFile>>& future, const std::filesystem::path& filename)
    {

    }

    bool readFile(SFuture<uint32_t>& future, IReadFile* file, void* buffer, size_t offset, size_t size)
    {

    }

    bool writeFile(SFuture<uint32_t>& future, IWriteFile* file, const void* buffer, size_t offset, size_t size)
    {

    }

    // @sadiuk add more methods taken from IFileSystem and IOSOperator
    // and implement via m_dispatcher and ISystemCaller if needed
    // (any system calls should take place in ISystemCaller which is called by CAsyncQueue and nothing else)
};

}
}

#endif
