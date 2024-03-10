#ifndef _NBL_VIDEO_THREADSAFE_QUEUE_ADAPTER_H_INCLUDED_
#define _NBL_VIDEO_THREADSAFE_QUEUE_ADAPTER_H_INCLUDED_

#include "nbl/video/IQueue.h"

namespace nbl::video
{

/*
    A thread-safe implementation of IQueue.
    Note, that using the same queue as both a threadsafe queue and a normal queue invalidates the safety.
*/
class CThreadSafeQueueAdapter final : public IQueue
{
        friend class ILogicalDevice; // to access the destructor
        friend class ISwapchain;

    public:
        inline CThreadSafeQueueAdapter(ILogicalDevice* originDevice, IQueue* const original)
            : IQueue(originDevice,original->getFamilyIndex(),original->getFlags(),original->getPriority()), originalQueue(original) {}        
        inline CThreadSafeQueueAdapter() : IQueue(nullptr, 0, CREATE_FLAGS::PROTECTED_BIT, 0.f) {}

        inline bool startCapture() override
        { 
            std::lock_guard g(m);
            return originalQueue->startCapture();
        }
        inline bool endCapture() override
        { 
            std::lock_guard g(m);
            return originalQueue->endCapture(); 
        }

        virtual bool insertDebugMarker(const char* name, const core::vector4df_SIMD& color=core::vector4df_SIMD(1.0, 1.0, 1.0, 1.0)) override
        {
            std::lock_guard g(m);
            return originalQueue->insertDebugMarker(name,color);
        }
        virtual bool beginDebugMarker(const char* name, const core::vector4df_SIMD& color=core::vector4df_SIMD(1.0, 1.0, 1.0, 1.0)) override
        {
            std::lock_guard g(m);
            return originalQueue->beginDebugMarker(name,color);
        }
        virtual bool endDebugMarker() override
        {
            std::lock_guard g(m);
            return originalQueue->endDebugMarker();
        }

        inline RESULT submit(const std::span<const SSubmitInfo> _submits) override
        {
            std::lock_guard g(m);
            return originalQueue->submit(_submits);
        }

        inline RESULT waitIdle() override
        {
            std::lock_guard g(m);
            return originalQueue->waitIdle();
        }

        inline uint32_t cullResources(const ISemaphore* sema=nullptr) override
        {
            std::lock_guard g(m);
            return originalQueue->cullResources(sema);
        }

        inline IQueue* getUnderlyingQueue() const
        {
            return originalQueue;
        }

        inline const void* getNativeHandle() const
        {
            return originalQueue->getNativeHandle();
        }

    protected:
        inline ~CThreadSafeQueueAdapter()
        {
            delete originalQueue;
        }

        // These shall never be called, they're here just to stop the class being pure virtual
        inline RESULT submit_impl(const std::span<const SSubmitInfo> _submits) override
        {
            assert(false);
            return originalQueue->submit_impl(_submits);
        }
        inline RESULT waitIdle_impl() const override
        {
            assert(false);
            return originalQueue->waitIdle_impl();
        }


        // used to use unique_ptr here, but it needed `~IQueue` to be public, which requires a custom deleter, etc.
        IQueue* const originalQueue = nullptr;
        mutable std::mutex m;
};

}
#endif