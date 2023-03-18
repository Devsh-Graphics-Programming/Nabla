#include "nbl/video/CVulkanDescriptorSet.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanDescriptorSet::~CVulkanDescriptorSet()
{
	if (!isZombie() && getPool()->allowsFreeing())
	{
		const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
		auto* vk = vulkanDevice->getFunctionTable();

		const auto* vk_dsPool = IBackendObject::device_compatibility_cast<const CVulkanDescriptorPool*>(getPool(), getOriginDevice());
		assert(vk_dsPool);

		vk->vk.vkFreeDescriptorSets(vulkanDevice->getInternalObject(), vk_dsPool->getInternalObject(), 1u, &m_descriptorSet);
	}
}

}