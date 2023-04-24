#ifndef _NBL_SYSTEM_I_FILE_ALLOCATOR_H_INCLUDED_
#define _NBL_SYSTEM_I_FILE_ALLOCATOR_H_INCLUDED_

#include "nbl/core/declarations.h"

#include <cstdint>
#include <cstdlib>

namespace nbl::system
{

// This interface class provides the callbacks for `CFileView` which creates a mapped file over some memory
class IFileViewAllocator
{
	public:
		IFileViewAllocator(void* _state) : m_state(_state) {}

		virtual void* alloc(size_t size) = 0;
		virtual bool dealloc(void* data, size_t size) = 0;

	protected:
		void* m_state;
};

// Regular old file in RAM
class CPlainHeapAllocator : public IFileViewAllocator
{
	public:
		using IFileViewAllocator::IFileViewAllocator;

		void* alloc(size_t size) override
		{
			return malloc(size);
		}
		bool dealloc(void* data, size_t size) override
		{
			free(data);
			return true;
		}
};

// This allocator is useful to create an `IFile` over memory that already contains something or is owned by some other component
// e.g. memory mapped IGPUBuffer, string_view or a string, or buffers handed out by other APIs
class CNullAllocator : public IFileViewAllocator
{
	public:
		using IFileViewAllocator::IFileViewAllocator;

		void* alloc(size_t size) override
		{
			return nullptr;
		}
		bool dealloc(void* data, size_t size) override
		{
			return true;
		}
};

}

#ifdef _NBL_PLATFORM_WINDOWS_
#include "nbl/system/CFileViewVirtualAllocatorWin32.h"
	using VirtualMemoryAllocator = nbl::system::CFileViewVirtualAllocatorWin32;
#elif defined(_NBL_PLATFORM_LINUX_) || defined(_NBL_PLATFORM_ANDROID_) || defined(_NBL_PLATFORM_MACOS_)
#include "nbl/system/CFileViewVirtualAllocatorPOSIX.h"
	using VirtualMemoryAllocator = nbl::system::CFileViewVirtualAllocatorPOSIX;
#else
#error "Unsupported platform!"
#endif

#endif
