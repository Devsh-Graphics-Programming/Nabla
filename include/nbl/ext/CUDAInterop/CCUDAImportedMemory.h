#ifndef _NBL_EXT_CUDA_INTEROP_C_CUDA_IMPORTED_MEMORY_H_
#define _NBL_EXT_CUDA_INTEROP_C_CUDA_IMPORTED_MEMORY_H_

#include "nbl/video/declarations.h"

#include <memory>
#include <utility>

namespace nbl::video
{

class CCUDADevice;

namespace cuda_native
{
struct SAccess;
}

class NBL_API2 CCUDAImportedMemory : public core::IReferenceCounted
{
	public:
		~CCUDAImportedMemory() override;

	private:
		friend class CCUDADevice;
		friend struct cuda_native::SAccess;

		struct SNativeState;
		CCUDAImportedMemory(core::smart_refctd_ptr<CCUDADevice> device, core::smart_refctd_ptr<nbl::video::IDeviceMemoryAllocation> src, std::unique_ptr<SNativeState>&& nativeState);

		core::smart_refctd_ptr<CCUDADevice> m_device;
		core::smart_refctd_ptr<IDeviceMemoryAllocation> m_src;
		std::unique_ptr<SNativeState> m_native;
};

}

#endif
