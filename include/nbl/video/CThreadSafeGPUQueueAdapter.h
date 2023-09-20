#ifndef __IRR_GPU_QUEUE_THREADSAFE_ADAPTER_H_INCLUDED__
#define __IRR_GPU_QUEUE_THREADSAFE_ADAPTER_H_INCLUDED__
#include "IGPUQueue.h"
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

        virtual bool insertDebugMarker(const char* name, const core::vector4df_SIMD& color = core::vector4df_SIMD(1.0, 1.0, 1.0, 1.0)) override
        {
            std::lock_guard g(m);
            return originalQueue->insertDebugMarker(name, color);
        }
        virtual bool beginDebugMarker(const char* name, const core::vector4df_SIMD& color = core::vector4df_SIMD(1.0, 1.0, 1.0, 1.0)) override
        {
            std::lock_guard g(m);
            return originalQueue->beginDebugMarker(name, color);
        }
        virtual bool endDebugMarker() override
        {
            std::lock_guard g(m);
            return originalQueue->endDebugMarker();
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