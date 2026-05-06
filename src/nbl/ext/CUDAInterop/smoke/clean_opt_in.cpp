#include "nbl/ext/CUDAInterop/CUDAInterop.h"
#include "nbl/system/IApplicationFramework.h"

#include <type_traits>

#ifdef _NBL_COMPILE_WITH_CUDA_
#error "Nabla::Nabla must not propagate the CUDA build define."
#endif

#ifdef CUDA_VERSION
#error "Nabla::Nabla must not require CUDA SDK headers."
#endif

namespace
{

class CUDAInteropCleanOptInSmoke final : public nbl::system::IApplicationFramework
{
	using base_t = nbl::system::IApplicationFramework;

public:
	using base_t::base_t;

	bool onAppInitialized(nbl::core::smart_refctd_ptr<nbl::system::ISystem>&&) override
	{
		static_assert(std::is_same_v<decltype(nbl::video::CCUDAExportableMemory::SCreationParams{}.location), nbl::video::ECUDAMemoryLocation>);

		const nbl::video::CCUDAExportableMemory::SCreationParams params = {
			.size = 4096,
			.alignment = 4096,
			.location = nbl::video::ECUDAMemoryLocation::DEVICE,
		};
		return isAPILoaded() && params.location==nbl::video::ECUDAMemoryLocation::DEVICE;
	}

	void workLoopBody() override {}
	bool keepRunning() override { return false; }
};

}

NBL_MAIN_FUNC(CUDAInteropCleanOptInSmoke)
