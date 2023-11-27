#include "nbl/video/IPhysicalDevice.h"
#include "nbl/video/ILogicalDevice.h"
#include "nbl/video/utilities/ICommandPoolCache.h"

using namespace nbl;
using namespace video;


ICommandPoolCache::ICommandPoolCache(ILogicalDevice* device, const uint32_t queueFamilyIx, const ICommandPool::CREATE_FLAGS _flags, const uint32_t capacity)
	: m_reserved(malloc(CommandPoolAllocator::reserved_size(1u,capacity,1u))), m_cmdPoolAllocator(m_reserved,0u,0u,1u,capacity,1u), m_deferredResets()
{
	m_cache = new core::smart_refctd_ptr<IGPUCommandPool>[capacity];
	for (auto i=0u; i<getCapacity(); i++)
		m_cache[i] = device->createCommandPool(queueFamilyIx,_flags);
}

void ICommandPoolCache::releaseSet(const uint32_t poolIx)
{
	m_cache[poolIx]->reset();
	m_cmdPoolAllocator.free_addr(poolIx,1);
}

void ICommandPoolCache::DeferredCommandPoolResetter::operator()()
{
	#ifdef _NBL_DEBUG
	assert(m_cache && m_poolIx<m_cache->getCapacity());
	#endif // _NBL_DEBUG
	m_cache->releaseSet(m_poolIx);
}