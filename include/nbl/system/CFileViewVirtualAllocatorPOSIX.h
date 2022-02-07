#ifndef __NBL_SYSTEM_C_FILE_VIEW_VIRTUAL_ALLOCATOR_POSIX_H_INCLUDED__
#define __NBL_SYSTEM_C_FILE_VIEW_VIRTUAL_ALLOCATOR_POSIX_H_INCLUDED__
#include "IFileViewAllocator.h"
#include <sys/mman.h>

#if defined(_NBL_PLATFORM_LINUX_) || defined(_NBL_PLATFORM_ANDROID_)

namespace nbl::system
{
class CFileViewVirtualAllocatorPOSIX : public IFileViewAllocator
{
public:
    void* alloc(size_t size) override
    {
        return mmap((caddr_t)0, size, PROT_WRITE | PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
    }
    bool dealloc(void* data, size_t size) override
    {
        auto ret = munmap(data, size);
        return ret != -1;
    }
};
}

#endif
#endif