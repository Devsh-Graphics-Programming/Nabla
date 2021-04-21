#ifndef __IRR_GPU_QUEUE_THREADSAFE_ADAPTER_H_INCLUDED__
#define __IRR_GPU_QUEUE_THREADSAFE_ADAPTER_H_INCLUDED__
#include "IGPUQueue.h"

namespace nbl::video
{
    class CThreadSafeGPUQueueAdapter : public IGPUQueue
    {
    protected:
        nbl::core::smart_refctd_ptr<IGPUQueue> originalQueue = nullptr;
        std::mutex m;
    public:
        CThreadSafeGPUQueueAdapter(nbl::core::smart_refctd_ptr<IGPUQueue>&& originalQueue) : IGPUQueue(nullptr, originalQueue->getFamilyIndex(), originalQueue->getFlags(), originalQueue->getPriority()), originalQueue(originalQueue) {}
        CThreadSafeGPUQueueAdapter(CThreadSafeGPUQueueAdapter&& other) : IGPUQueue(nullptr, other.originalQueue->getFamilyIndex(), other.originalQueue->getFlags(), other.originalQueue->getPriority()), originalQueue(other.originalQueue) {}
        
        CThreadSafeGPUQueueAdapter(const CThreadSafeGPUQueueAdapter& other) : IGPUQueue(nullptr, other.originalQueue->getFamilyIndex(), other.originalQueue->getFlags(), other.originalQueue->getPriority()), originalQueue(other.originalQueue) {}
        CThreadSafeGPUQueueAdapter(const nbl::core::smart_refctd_ptr<IGPUQueue>& originalQueue) : IGPUQueue(nullptr, originalQueue->getFamilyIndex(), originalQueue->getFlags(), originalQueue->getPriority()), originalQueue(originalQueue) {}
        
        CThreadSafeGPUQueueAdapter() : IGPUQueue(nullptr, 0, E_CREATE_FLAGS::ECF_PROTECTED_BIT, 0.f) {};

        CThreadSafeGPUQueueAdapter& operator=(CThreadSafeGPUQueueAdapter&& other)
        {
            originalQueue = other.originalQueue;
            return *this;
        }
        CThreadSafeGPUQueueAdapter& operator=(nbl::core::smart_refctd_ptr<IGPUQueue>&& other)
        {
            originalQueue = other;
            return *this;
        }
        virtual bool submit(uint32_t _count, const SSubmitInfo* _submits, IGPUFence* _fence) override
        {
            std::lock_guard g(m);
            return originalQueue->submit(_count, _submits, _fence);
        }

        virtual bool present(const SPresentInfo& info) override
        {
            std::lock_guard g(m);
            return originalQueue->present(info);
        }

        nbl::core::smart_refctd_ptr<IGPUQueue> getUnderlyingQueue() const
        {
            return originalQueue;
        }
    };
}
#endif