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
    { }

public:
    ~CVulkanDeferredOperation();
    
    bool join() override;
    uint32_t getMaxConcurrency() override;
    E_STATUS getStatus() override;
    E_STATUS joinAndWait() override;

    inline VkDeferredOperationKHR getInternalObject() const { return m_deferredOp; }

private:
    VkDeferredOperationKHR m_deferredOp;
};

}

#endif
