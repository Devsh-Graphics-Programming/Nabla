#ifndef _NBL_VIDEO_I_DEFERRED_OPERATION_H_INCLUDED_
#define _NBL_VIDEO_I_DEFERRED_OPERATION_H_INCLUDED_


#include "nbl/system/declarations.h"

#include "nbl/video/decl/IBackendObject.h"


namespace nbl::video
{

// Note how we don't have a `getResult()` because the return value depends on the actual operation being deferred
// Therefore every class with deferred operations has a `getResult(const IDeferredOperation* const)` or similar
class IDeferredOperation : public IBackendObject
{
    public:
        // How many calls to `execute()` in parallel you can do before you stop seeing a benefit
        virtual uint32_t getMaxConcurrency() const = 0;

        // You call this multiple times on a thread until its COMPLETED or DONE
        enum class STATUS : uint8_t
        {
            COMPLETED,
            THREAD_DONE,
            THREAD_IDLE,
            _ERROR
        };
        inline STATUS execute()
        {
            const auto retval = execute_impl();
            if (retval==STATUS::COMPLETED || retval==STATUS::_ERROR)
                m_resourceTracking.clear();
            return retval;
        }

        // Returns false if nothing was deferred or execution has fully completed
        virtual bool isPending() const = 0;

        // Returns true when deferred operation is no longer pending, false upon some error
        inline bool executeToCompletion()
        {
            // In a multithreaded execution this is the part you want to thread
            auto status = execute();
            while (status==STATUS::THREAD_IDLE)
            {
                std::this_thread::yield();
                status = execute();
            }
            if (status==STATUS::_ERROR)
                return false;
            // and this is where you'd want to join all threads
            if (status!=STATUS::COMPLETED) // this could just be done with one thread
            do
            {
                std::this_thread::yield();
            } while (isPending());
            return true;
        }

    protected:
        explicit IDeferredOperation(core::smart_refctd_ptr<const ILogicalDevice>&& dev) : IBackendObject(std::move(dev)), m_resourceTracking() {}

        virtual STATUS execute_impl() = 0;

    private:
        friend class ILogicalDevice;
        // when we improve allocators, etc. we'll stop using STL containers here
        core::vector<core::smart_refctd_ptr<const IReferenceCounted>> m_resourceTracking;
};

}

#endif