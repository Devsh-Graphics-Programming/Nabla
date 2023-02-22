#ifndef _NBL_SYSTEM_C_FILE_VIEW_APK_ALLOCATOR_H_INCLUDED_
#define _NBL_SYSTEM_C_FILE_VIEW_APK_ALLOCATOR_H_INCLUDED_


#include "nbl/system/IFileViewAllocator.h"


namespace nbl::system
{
#ifdef _NBL_PLATFORM_ANDROID_
class CFileViewAPKAllocator : public IFileViewAllocator
{
	public:
		using IFileViewAllocator::IFileViewAllocator;

		// should never be called
		void* alloc(size_t size) override;
		bool dealloc(void* data, size_t size) override;
};
#endif
}

#endif