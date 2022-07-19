#include "nbl/video/IPhysicalDevice.h"
#include "nbl/video/ILogicalDevice.h"
#include "nbl/video/utilities/IDescriptorSetCache.h"

using namespace nbl;
using namespace video;


IDescriptorSetCache::IDescriptorSetCache(ILogicalDevice* device, const uint32_t capacity)
	: m_descPool(), m_canonicalLayout(), m_reserved(malloc(DescSetAllocator::reserved_size(1u,capacity,1u))),
	m_setAllocator(m_reserved,0u,0u,1u,capacity,1u), m_deferredReclaims()
{
	m_cache = new core::smart_refctd_ptr<IGPUDescriptorSet>[capacity];
	std::fill_n(m_cache,capacity,nullptr);
}

IDescriptorSetCache::IDescriptorSetCache(ILogicalDevice* device, core::smart_refctd_ptr<IDescriptorPool>&& _descPool, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _canonicalLayout) : IDescriptorSetCache(device,_descPool->getCapacity())
{
	m_descPool = std::move(_descPool);
	m_canonicalLayout = std::move(_canonicalLayout);
	for (auto i=0u; i<getCapacity(); i++)
	{
		m_cache[i] = device->createDescriptorSet(
			m_descPool.get(),core::smart_refctd_ptr(m_canonicalLayout)
		);
	}
}