#ifndef _NBL_VIDEO_C_VULKAN_DEFERRED_OPERATION_H_INCLUDED_
#define _NBL_VIDEO_C_VULKAN_DEFERRED_OPERATION_H_INCLUDED_

#include "nbl/video/IDeferredOperation.h"

#include <volk.h>

namespace nbl::video
{
class ILogicalDevice;

class CVulkanDeferredOperation : public IDeferredOperation
{
public:
    CVulkanDeferredOperation(core::smart_refctd_ptr<ILogicalDevice>&& dev, VkDeferredOperationKHR vkDeferredOp)
        : IDeferredOperation(std::move(dev)), m_deferredOp(vkDeferredOp)
    {}

public:
    ~CVulkanDeferredOperation();

    bool join() override;
    uint32_t getMaxConcurrency() override;
    E_STATUS getStatus() override;
    E_STATUS joinAndWait() override;

    static void* operator new(size_t size) noexcept = delete;
    static void* operator new[](size_t size) noexcept = delete;
    static void* operator new(size_t size, std::align_val_t al) noexcept = delete;
    static void* operator new[](size_t size, std::align_val_t al) noexcept = delete;

    static inline void* operator new(size_t size, void* where) noexcept
    {
        return where;  // done
    }

    static inline void operator delete(void* ptr, void* place) noexcept
    {
        assert(false && "don't use");
    }

    static void* operator new[](size_t size, void* where) noexcept = delete;
    static void operator delete(void* ptr) noexcept;
    static void operator delete[](void* ptr) noexcept = delete;
    static void operator delete(void* ptr, size_t size) noexcept
    {
        operator delete(ptr);  //roll back to own operator with no size
    }
    static void operator delete[](void* ptr, size_t size) noexcept = delete;

    inline VkDeferredOperationKHR getInternalObject() const { return m_deferredOp; }

private:
    VkDeferredOperationKHR m_deferredOp;
};

}

#endif
