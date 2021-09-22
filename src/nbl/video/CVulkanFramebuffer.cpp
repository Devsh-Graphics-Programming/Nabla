#include "nbl/video/CVulkanFramebuffer.h"

#include "nbl/video/CVulkanLogicalDevice.h"
#include "nbl/video/CVulkanRenderpass.h"
#include "nbl/video/CVulkanImageView.h"

namespace nbl::video
{

CVulkanFramebuffer::~CVulkanFramebuffer()
{
    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    auto* vk = vulkanDevice->getFunctionTable();
    vk->vk.vkDestroyFramebuffer(vulkanDevice->getInternalObject(), m_vkfbo, nullptr);
}

}