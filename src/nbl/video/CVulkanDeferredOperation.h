#ifndef _NBL_VIDEO_C_VULKAN_DEFERRED_OPERATION_H_INCLUDED_
#define _NBL_VIDEO_C_VULKAN_DEFERRED_OPERATION_H_INCLUDED_

#include "nbl/video/IDeferredOperation.h"

#include <volk.h>

namespace nbl::video
{

class CVulkanDeferredOperation : public IDeferredOperation
{
    public:
        CVulkanDeferredOperation(const ILogicalDevice* dev, VkDeferredOperationKHR vkDeferredOp)
            : IDeferredOperation(core::smart_refctd_ptr<const ILogicalDevice>(dev)), m_deferredOp(vkDeferredOp) { }

        uint32_t getMaxConcurrency() const override;
        bool isPending() const override;

        static void* operator new(size_t size) noexcept = delete;
        static void* operator new[](size_t size) noexcept = delete;
        static void* operator new(size_t size, std::align_val_t al) noexcept = delete;
        static void* operator new[](size_t size, std::align_val_t al) noexcept = delete;

        static inline void* operator new(size_t size, void* where) noexcept
        {
            return where; // done
        }

        static inline void operator delete (void* ptr, void* place) noexcept
        {
            assert(false && "don't use");
        }

        static void* operator new[](size_t size, void* where) noexcept = delete;
        static void operator delete(void* ptr) noexcept;
        static void operator delete[](void* ptr) noexcept = delete;
        static inline void operator delete(void* ptr, size_t size) noexcept
        {
            operator delete(ptr); //roll back to own operator with no size
        }
        static void operator delete[](void* ptr, size_t size) noexcept = delete;

        inline VkDeferredOperationKHR getInternalObject() const { return m_deferredOp; }

    private:
        ~CVulkanDeferredOperation();

        STATUS execute_impl() override;

        VkDeferredOperationKHR m_deferredOp;
};

}

#endif
