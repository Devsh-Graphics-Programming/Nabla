#include "nbl/video/CUDAInterop.h"
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
		static_assert(std::is_class_v<nbl::video::CCUDADevice>);
		static_assert(std::is_class_v<nbl::video::CCUDAExportableMemory>);
		static_assert(std::is_class_v<nbl::video::CCUDAImportedMemory>);
		static_assert(std::is_class_v<nbl::video::CCUDAImportedSemaphore>);
		return isAPILoaded();
	}

	void workLoopBody() override {}
	bool keepRunning() override { return false; }
};

}

NBL_MAIN_FUNC(CUDAInteropCleanOptInSmoke)
