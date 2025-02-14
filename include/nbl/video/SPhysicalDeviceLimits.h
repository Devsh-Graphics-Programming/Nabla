#ifndef _NBL_VIDEO_S_PHYSICAL_DEVICE_LIMITS_H_INCLUDED_
#define _NBL_VIDEO_S_PHYSICAL_DEVICE_LIMITS_H_INCLUDED_


#include "nbl/asset/utils/CGLSLCompiler.h" // asset::CGLSLCompiler::E_SPIRV_VERSION

#include "nbl/asset/IImage.h"
#include "nbl/asset/IRenderpass.h"

#include "nbl/builtin/hlsl/limits.hlsl"
#include "nbl/builtin/hlsl/enums.hlsl"

#include <type_traits>


namespace nbl::video
{

// Struct is populated with Nabla Core Profile Limit Minimums
struct SPhysicalDeviceLimits
{
    enum E_POINT_CLIPPING_BEHAVIOR : uint8_t {
        EPCB_ALL_CLIP_PLANES = nbl::hlsl::PointClippingBehavior::EPCB_ALL_CLIP_PLANES,
        EPCB_USER_CLIP_PLANES_ONLY = nbl::hlsl::PointClippingBehavior::EPCB_USER_CLIP_PLANES_ONLY
    };
    using RESOLVE_MODE_FLAGS = asset::IRenderpass::SCreationParams::SSubpassDescription::SDepthStencilAttachmentsRef::RESOLVE_MODE;

	#include "nbl/video/SPhysicalDeviceLimits_members.h"

    // utility functions
    // In the cases where the workgroups synchronise with each other such as work DAGs (i.e. `CScanner`),
    // `workgroupSpinningProtection` is meant to protect against launching a dispatch so wide that
    // a workgroup of the next cut of the DAG spins for an extended time to wait on a workgroup from a previous one.
    inline uint32_t computeOptimalPersistentWorkgroupDispatchSize(const uint64_t elementCount, const uint32_t workgroupSize, const uint32_t workgroupSpinningProtection = 1u) const
    {
        assert(elementCount != 0ull && "Input element count can't be 0!");
        const uint64_t infinitelyWideDeviceWGCount = (elementCount - 1ull) / (static_cast<uint64_t>(workgroupSize) * static_cast<uint64_t>(workgroupSpinningProtection)) + 1ull;
        const uint32_t maxResidentWorkgroups = maxResidentInvocations / workgroupSize;
        return static_cast<uint32_t>(hlsl::min<uint64_t>(infinitelyWideDeviceWGCount, maxResidentWorkgroups));
    }

	inline bool isSubsetOf(const SPhysicalDeviceLimits& _rhs) const {
        #include "nbl/video/SPhysicalDeviceLimits_subset.h"

        return true;
	}
};

} // nbl::video

#endif
