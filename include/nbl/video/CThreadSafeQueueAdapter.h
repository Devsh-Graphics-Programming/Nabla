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
        friend class ISwapchain;
    protected:
        std::unique_ptr<IQueue> originalQueue = nullptr;
        std::mutex m;

        inline RESULT submit_impl(const uint32_t _count, const SSubmitInfo* const _submits) override
        {
            IQueue* msvcIsDumb = originalQueue.get();
            return this->submit_impl(_count,_submits);
        }

    public:
        inline CThreadSafeQueueAdapter(ILogicalDevice* originDevice, std::unique_ptr<IQueue>&& original)
            : IQueue(originDevice, original->getFamilyIndex(),original->getFlags(),original->getPriority()), originalQueue(std::move(original)) {}        
        inline CThreadSafeQueueAdapter() : IQueue(nullptr, 0, CREATE_FLAGS::PROTECTED_BIT, 0.f) {}

        inline RESULT waitIdle() const override
        {
            std::lock_guard g(m);
            return originalQueue->waitIdle();
        }

        inline RESULT submit(const uint32_t _count, const SSubmitInfo* const _submits) override
        {
            std::lock_guard g(m);
            return originalQueue->submit(_count, _submits);
        }

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

        inline IQueue* getUnderlyingQueue() const
        {
            return originalQueue.get();
        }

        inline const void* getNativeHandle() const
        {
            return originalQueue->getNativeHandle();
        }
};

}
#endif