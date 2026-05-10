#ifndef _NBL_VIDEO_C_CUDA_IMPORTED_MEMORY_H_
#define _NBL_VIDEO_C_CUDA_IMPORTED_MEMORY_H_

#include "nbl/video/declarations.h"
#include "nbl/video/CUDAInteropHandles.h"

#include <memory>
#include <type_traits>
#include <utility>

namespace nbl::video
{

class CCUDADevice;

class NBL_API2 CCUDAImportedMemory : public core::IReferenceCounted
{
	public:
		~CCUDAImportedMemory() override;
		cuda_interop::SCUexternalMemory getInternalObject() const;
		bool getMappedBuffer(cuda_interop::SCUdeviceptr* mappedBuffer) const;
		bool getMappedBuffer(cuda_interop::SCUdeviceptr& mappedBuffer) const { return getMappedBuffer(&mappedBuffer); }
		template<typename DevicePtr>
		requires (!std::is_same_v<std::remove_cv_t<DevicePtr>,cuda_interop::SCUdeviceptr>)
		bool getMappedBuffer(DevicePtr* mappedBuffer) const
		{
			cuda_interop::SCUdeviceptr opaqueMappedBuffer = {};
			const auto result = getMappedBuffer(&opaqueMappedBuffer);
			if (result && mappedBuffer)
				*mappedBuffer = static_cast<DevicePtr>(opaqueMappedBuffer);
			return result;
		}
		template<typename DevicePtr>
		requires (!std::is_same_v<std::remove_cv_t<DevicePtr>,cuda_interop::SCUdeviceptr>)
		bool getMappedBuffer(DevicePtr& mappedBuffer) const
		{
			return getMappedBuffer(&mappedBuffer);
		}

	private:
		friend class CCUDADevice;

		struct SNativeState;
		CCUDAImportedMemory(core::smart_refctd_ptr<CCUDADevice> device, core::smart_refctd_ptr<nbl::video::IDeviceMemoryAllocation> src, std::unique_ptr<SNativeState>&& nativeState);

		core::smart_refctd_ptr<CCUDADevice> m_device;
		core::smart_refctd_ptr<IDeviceMemoryAllocation> m_src;
		std::unique_ptr<SNativeState> m_native;
};

}

#endif
