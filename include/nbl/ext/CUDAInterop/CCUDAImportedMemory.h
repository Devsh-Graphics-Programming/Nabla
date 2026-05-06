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

class CCUDAImportedMemory : public core::IReferenceCounted
{
	public:
		struct SNativeState;
		CCUDAImportedMemory(core::smart_refctd_ptr<CCUDADevice> device, core::smart_refctd_ptr<nbl::video::IDeviceMemoryAllocation> src, std::unique_ptr<SNativeState>&& nativeState);

		~CCUDAImportedMemory() override;

	private:
		friend struct cuda_native::SAccess;

		core::smart_refctd_ptr<CCUDADevice> m_device;
		core::smart_refctd_ptr<IDeviceMemoryAllocation> m_src;
		std::unique_ptr<SNativeState> m_native;
};

}

#endif
