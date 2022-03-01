#include "nbl/system/CFileViewVirtualAllocatorWin32.h"

using namespace nbl::system;

#ifdef _NBL_PLATFORM_WINDOWS_
#include "Windows.h"

void* CFileViewVirtualAllocatorWin32::alloc(size_t size)
{
	return VirtualAlloc(nullptr, size, MEM_COMMIT|MEM_RESERVE, PAGE_READWRITE); // TODO: are these even the right flags? Do we want to commit everything right away?
}
bool CFileViewVirtualAllocatorWin32::dealloc(void* data, size_t size)
{
	return VirtualFree(data, 0, MEM_RELEASE);
}
#endif