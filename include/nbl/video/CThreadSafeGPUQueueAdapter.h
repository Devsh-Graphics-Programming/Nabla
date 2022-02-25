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
        IGPUQueue* originalQueue = nullptr;
        std::mutex m;
    public:
        CThreadSafeGPUQueueAdapter(ILogicalDevice* originDevice, IGPUQueue* original)
            : IGPUQueue(originDevice, original->getFamilyIndex(),original->getFlags(),original->getPriority()), originalQueue(original) {}        

        CThreadSafeGPUQueueAdapter() : IGPUQueue(nullptr, 0, E_CREATE_FLAGS::ECF_PROTECTED_BIT, 0.f) {};

        ~CThreadSafeGPUQueueAdapter()
        {
            if (originalQueue)
            {
                delete originalQueue;
            }
        }

        virtual bool submit(uint32_t _count, const SSubmitInfo* _submits, IGPUFence* _fence) override
        {
            std::lock_guard g(m);
            return originalQueue->submit(_count, _submits, _fence);
        }

        virtual ISwapchain::E_PRESENT_RESULT present(const SPresentInfo& info) override
        {
            std::lock_guard g(m);
            return originalQueue->present(info);
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
};

}
#endif