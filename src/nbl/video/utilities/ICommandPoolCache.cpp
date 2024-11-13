#include "nbl/video/IPhysicalDevice.h"
#include "nbl/video/ILogicalDevice.h"
#include "nbl/video/utilities/ICommandPoolCache.h"

using namespace nbl;
using namespace video;


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