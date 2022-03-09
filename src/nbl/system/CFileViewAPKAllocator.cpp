#include "nbl/system/CFileViewAPKAllocator.h"

using namespace nbl::system;

#ifdef _NBL_PLATFORM_ANDROID_
#include <jni.h>
#include <asset_manager.h>

void* CFileViewAPKAllocator::alloc(size_t size)
{
	assert(false);
	exit(-0x45);
	return nullptr;
}

bool CFileViewAPKAllocator::dealloc(void* data, size_t size)
{
	AAsset_close(reinterpret_cast<AAsset*>(m_state));
	return true;
}
#endif