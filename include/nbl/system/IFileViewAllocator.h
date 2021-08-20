#ifndef __NBL_SYSTEM_I_FILE_ALLOCATOR_H_INCLUDED__
#define __NBL_SYSTEM_I_FILE_ALLOCATOR_H_INCLUDED__

#include <nabla.h>

#include <cstdint>
#include <cstdlib>

namespace nbl::system
{
	class IFileViewAllocator
	{
	public:
		virtual void* alloc(size_t size) = 0;
		virtual bool dealloc(void* data) = 0;
	};

	class CPlainHeapAllocator : public IFileViewAllocator
	{
	public:
		void* alloc(size_t size) override
		{
			return malloc(size);
		}
		bool dealloc(void* data) override
		{
			free(data);
			return true;
		}

	};
}
#endif