#ifndef __NBL_SYSTEM_C_FILE_VIEW_VIRTUAL_ALLOCATOR_WIN32_H_INCLUDED__
#define __NBL_SYSTEM_C_FILE_VIEW_VIRTUAL_ALLOCATOR_WIN32_H_INCLUDED__
#include "IFileViewAllocator.h"

#ifdef _NBL_PLATFORM_WINDOWS_
#include "Windows.h"

namespace nbl::system 
{
class CFileViewVirtualAllocatorWin32 : public IFileViewAllocator
{
public:
	void* alloc(size_t size) override
	{
		return VirtualAlloc(nullptr, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
	}
	bool dealloc(void* data, size_t size) override
	{
		return VirtualFree(data, 0, MEM_RELEASE);
	}
};
}

#endif
#endif