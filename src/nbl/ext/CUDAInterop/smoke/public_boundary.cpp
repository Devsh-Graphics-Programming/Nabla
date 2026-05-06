#include "nabla.h"

#ifdef _NBL_COMPILE_WITH_CUDA_
#error "Nabla consumers must not get the CUDA opt-in define."
#endif

#ifdef CUDA_VERSION
#error "Nabla consumers must not include CUDA SDK headers."
#endif

#include "nbl/system/IApplicationFramework.h"

#ifdef _NBL_COMPILE_WITH_CUDA_
#error "Nabla consumers must not get the CUDA opt-in define."
#endif

#ifdef CUDA_VERSION
#error "Nabla consumers must not include CUDA SDK headers."
#endif

#include "nbl/ext/CUDAInterop/CUDAInterop.h"

#ifdef _NBL_COMPILE_WITH_CUDA_
#error "Nabla consumers must not get the CUDA opt-in define."
#endif

#ifdef CUDA_VERSION
#error "Nabla consumers must not include CUDA SDK headers."
#endif

namespace
{

class CUDAInteropPublicBoundarySmoke final : public nbl::system::IApplicationFramework
{
	using base_t = nbl::system::IApplicationFramework;

public:
	using base_t::base_t;

	bool onAppInitialized(nbl::core::smart_refctd_ptr<nbl::system::ISystem>&&) override
	{
		return isAPILoaded();
	}

	void workLoopBody() override {}
	bool keepRunning() override { return false; }
};

}

NBL_MAIN_FUNC(CUDAInteropPublicBoundarySmoke)
