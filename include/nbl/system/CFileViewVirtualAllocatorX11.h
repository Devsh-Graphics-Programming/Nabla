#ifndef __NBL_SYSTEM_C_FILE_VIEW_VIRTUAL_ALLOCATOR_X11_H_INCLUDED__
#define __NBL_SYSTEM_C_FILE_VIEW_VIRTUAL_ALLOCATOR_X11_H_INCLUDED__
#include "IFileViewAllocator.h"

#if defined(_NBL_PLATFORM_LINUX_) || defined(_NBL_PLATFORM_ANDROID_)

namespace nbl::system 
{
class CFileViewVirtualAllocatorX11 : public IFileViewAllocator
{
public:
	void* alloc(size_t size) override
	{
		assert(false);
		return nullptr;
	}
	bool dealloc(void* data) override
	{
		assert(false);
		return false;
	}
};
}

#endif
#endif