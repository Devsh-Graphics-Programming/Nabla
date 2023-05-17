#include "nbl/system/IFileViewAllocator.h"

using namespace nbl::system;

#if defined(_NBL_PLATFORM_LINUX_) || defined(_NBL_PLATFORM_ANDROID_) || defined(_NBL_PLATFORM_MACOS_)
#include <sys/mman.h>

void* CFileViewVirtualAllocatorPOSIX::alloc(size_t size)
{
	return mmap((caddr_t)0, size, PROT_WRITE|PROT_READ, MAP_PRIVATE|MAP_ANONYMOUS, 0, 0);
}
bool CFileViewVirtualAllocatorPOSIX::dealloc(void* data, size_t size)
{
	const auto ret = munmap(data,size);
	return ret != -1;
}
#endif
