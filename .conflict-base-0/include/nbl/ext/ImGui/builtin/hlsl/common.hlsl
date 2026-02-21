#ifndef _NBL_IMGUI_EXT_COMMON_HLSL_
#define _NBL_IMGUI_EXT_COMMON_HLSL_

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/bda/struct_declare.hlsl"

namespace nbl
{
namespace ext
{
namespace imgui
{
struct PushConstants
{
	uint64_t elementBDA;
	uint64_t elementCount;
	nbl::hlsl::float32_t2 scale;
	nbl::hlsl::float32_t2 translate;
	nbl::hlsl::float32_t4 viewport;
};

// would like to replace with our own BDA, but can't do bitfields right now (would need a different wrapper for those)
struct PerObjectData 
{
	uint32_t aabbMin, aabbMax; //! snorm16_t2 packed as [uint16_t, uint16_t]
	uint32_t texId : 26;
	uint32_t samplerIx : 6;
};
}
}
}

#endif // _NBL_IMGUI_EXT_COMMON_HLSL_
