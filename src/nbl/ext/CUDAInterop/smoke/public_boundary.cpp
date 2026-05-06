#include "nabla.h"
#include "nbl/system/IApplicationFramework.h"
#include "nbl/ext/CUDAInterop/CUDAInterop.h"

#ifdef _NBL_COMPILE_WITH_CUDA_
#error "Default Nabla consumers must not get the CUDA opt-in define."
#endif

#ifdef CUDA_VERSION
#error "Default Nabla consumers must not include CUDA SDK headers."
#endif

namespace
{

class CUDAInteropPublicBoundarySmoke final : public nbl::system::IApplicationFramework
{
	using base_t = nbl::system::IApplicationFramework;

public:
	using base_t::base_t;

	bool onAppInitialized(nbl::core::smart_refctd_ptr<nbl::system::ISystem>&& system) override
	{
		static_cast<void>(system);
		return isAPILoaded();
	}

	void workLoopBody() override {}
	bool keepRunning() override { return false; }
};

}

NBL_MAIN_FUNC(CUDAInteropPublicBoundarySmoke)
