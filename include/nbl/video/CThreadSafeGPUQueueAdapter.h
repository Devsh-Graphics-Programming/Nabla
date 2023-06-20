#ifndef _NBL_VIDEO_THREADSAFE_GPU_QUEUE_ADAPTER_H_INCLUDED_
#define _NBL_VIDEO_THREADSAFE_GPU_QUEUE_ADAPTER_H_INCLUDED_

#include "nbl/video/IGPUQueue.h"
#include "nbl/video/CVulkanSwapchain.h"

namespace nbl::video
{

/*
    A thread-safe implementation of IGPUQueue.
    Note, that using the same queue as both a threadsafe queue and a normal queue invalidates the safety.
*/
class CThreadSafeGPUQueueAdapter : public IGPUQueue
{
    friend class CVulkanSwapchain;
    protected:
        IGPUQueue* originalQueue = nullptr;
        std::mutex m;
    public:
        CThreadSafeGPUQueueAdapter(ILogicalDevice* originDevice, IGPUQueue* original)
            : IGPUQueue(originDevice, original->getFamilyIndex(),original->getFlags(),original->getPriority()), originalQueue(original) {}        

        CThreadSafeGPUQueueAdapter() : IGPUQueue(nullptr, 0, CREATE_FLAGS::PROTECTED_BIT, 0.f) {};

        ~CThreadSafeGPUQueueAdapter()
        {
            if (originalQueue)
                delete originalQueue;
        }

        virtual bool submit(const uint32_t _count, const SSubmitInfo* const _submits, IGPUFence* const _fence) override
        {
            std::lock_guard g(m);
            return originalQueue->submit(_count, _submits, _fence);
        }

        virtual bool startCapture() override 
        { 
            std::lock_guard g(m);
            return originalQueue->startCapture();
        }
        virtual bool endCapture() override 
        { 
            std::lock_guard g(m);
            return originalQueue->endCapture(); 
        }

        IGPUQueue* getUnderlyingQueue() const
        {
            return originalQueue;
        }

        virtual const void* getNativeHandle() const
        {
            return originalQueue->getNativeHandle();
        }
};

}
#endif