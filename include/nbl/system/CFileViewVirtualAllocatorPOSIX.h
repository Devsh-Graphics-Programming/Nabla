#ifndef _NBL_SYSTEM_C_FILE_VIEW_VIRTUAL_ALLOCATOR_POSIX_H_INCLUDED_
#define _NBL_SYSTEM_C_FILE_VIEW_VIRTUAL_ALLOCATOR_POSIX_H_INCLUDED_

namespace nbl::system
{
#if defined(_NBL_PLATFORM_LINUX_) || defined(_NBL_PLATFORM_ANDROID_)
class CFileViewVirtualAllocatorPOSIX : public IFileViewAllocator
{
	public:
		void* alloc(size_t size) override;
		bool dealloc(void* data, size_t size) override;
};
#endif
}

#endif