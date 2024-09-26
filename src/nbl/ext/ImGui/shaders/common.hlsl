 
namespace nbl::hlsl
{
	struct emulated_snorm16_t2
	{
		uint32_t packed;
	};
}

namespace nbl::ext::imgui
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
		nbl::hlsl::emulated_snorm16_t2 aabbMin, aabbMax;
		uint32_t texId : 26;
		uint32_t samplerIx : 6;
	};
}
