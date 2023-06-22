#ifndef _NBL_C_VULKAN_COMMAND_BUFFER_H_INCLUDED_
#define _NBL_C_VULKAN_COMMAND_BUFFER_H_INCLUDED_

#include "nbl/video/IGPUCommandBuffer.h"

#include "nbl/video/CVulkanDeviceFunctionTable.h"

#include "nbl/video/CVulkanEvent.h"
#include "nbl/video/CVulkanBuffer.h"
#include "nbl/video/CVulkanImage.h"
#include "nbl/video/CVulkanDescriptorSet.h"
#include "nbl/video/CVulkanRenderpass.h"
#include "nbl/video/CVulkanFramebuffer.h"
#include "nbl/video/CVulkanPipelineLayout.h"
#include "nbl/video/CVulkanComputePipeline.h"

namespace nbl::video
{

class CVulkanCommandBuffer final : public IGPUCommandBuffer
{
    public:
        CVulkanCommandBuffer(core::smart_refctd_ptr<const ILogicalDevice>&& logicalDevice, const LEVEL level,
            VkCommandBuffer _vkcmdbuf, core::smart_refctd_ptr<IGPUCommandPool>&& commandPool, system::logger_opt_smart_ptr&& logger)
            : IGPUCommandBuffer(std::move(logicalDevice), level, std::move(commandPool), std::move(logger)), m_cmdbuf(_vkcmdbuf)
        {}

        bool begin_impl(const core::bitflag<USAGE> recordingFlags, const SInheritanceInfo* const inheritanceInfo) override final;

        inline bool end_impl() override final
        {
            const VkResult retval = getFunctionTable().vkEndCommandBuffer(m_cmdbuf);
            return retval==VK_SUCCESS;
        }

        inline bool reset_impl(const core::bitflag<RESET_FLAGS> flags) override final
        {
            const VkResult result = getFunctionTable().vkResetCommandBuffer(m_cmdbuf,static_cast<VkCommandBufferResetFlags>(flags.value));
            return result==VK_SUCCESS;
        }

        inline void checkForParentPoolReset_impl() const override {}

        // TODO: rest of methods go to Arek
    
	    inline const void* getNativeHandle() const override {return &m_cmdbuf;}
        VkCommandBuffer getInternalObject() const {return m_cmdbuf;}

    private:
        const VolkDeviceTable& getFunctionTable() const;

        VkCommandBuffer m_cmdbuf;
};  

}

#endif
