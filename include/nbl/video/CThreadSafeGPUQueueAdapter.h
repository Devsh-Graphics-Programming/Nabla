#ifndef __IRR_GPU_QUEUE_THREADSAFE_ADAPTER_H_INCLUDED__
#define __IRR_GPU_QUEUE_THREADSAFE_ADAPTER_H_INCLUDED__
#include "IGPUQueue.h"

namespace nbl::video
{

/*
    A thread-safe implementation of IGPUQueue.
    Note, that using the same queue as both a threadsafe queue and a normal queue invalidates the safety.
*/
class CThreadSafeGPUQueueAdapter : public IGPUQueue
{
    protected:
        core::smart_refctd_ptr<IGPUQueue> originalQueue = nullptr;
        std::mutex m;
    public:
        // CThreadSafeGPUQueueAdapter(nbl::core::smart_refctd_ptr<IGPUQueue>&& original, core::smart_refctd_ptr<const ILogicalDevice>&& device)
        //     : IGPUQueue(std::move(device),original->getFamilyIndex(),original->getFlags(),original->getPriority()), originalQueue(std::move(original)) {}        
        CThreadSafeGPUQueueAdapter(nbl::core::smart_refctd_ptr<IGPUQueue>&& original, video::ILogicalDevice* device)
            : IGPUQueue(device,original->getFamilyIndex(),original->getFlags(),original->getPriority()), originalQueue(std::move(original)) {}        

        CThreadSafeGPUQueueAdapter() : IGPUQueue(nullptr, 0, E_CREATE_FLAGS::ECF_PROTECTED_BIT, 0.f) {};

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

        IGPUQueue* getUnderlyingQueue() const
        {
            return originalQueue.get();
        }
};

}
#endif