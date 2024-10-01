#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"

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

	struct PerObjectData 
	{
		uint32_t aabbMin, aabbMax; //! snorm16_t2 packed as [uint16_t, uint16_t]
		uint32_t texId : 26;
		uint32_t samplerIx : 6;
	};
}
}
}
