#ifndef __NBL_C_VULKAN_DEFERRED_OPERATION_H_INCLUDED__
#define __NBL_C_VULKAN_DEFERRED_OPERATION_H_INCLUDED__

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
    { }

    ~CVulkanDeferredOperation();
    
    bool join() override;
    uint32_t getMaxConcurrency() override;
    E_STATUS getStatus() override;

    inline VkDeferredOperationKHR getInternalObject() const { return m_deferredOp; }

private:
    VkDeferredOperationKHR m_deferredOp;
};

}

#endif
