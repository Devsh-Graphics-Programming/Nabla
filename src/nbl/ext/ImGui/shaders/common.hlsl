// temporary, maybe we should ask user for it and the layout BUT only for sampler + texture binding IDs + set IDs they belong to and size of the texture array?
#define NBL_MAX_IMGUI_TEXTURES 69

#ifdef __HLSL_VERSION
	struct VSInput
	{
		[[vk::location(0)]] float2 position : POSITION;
		[[vk::location(1)]] float2 uv : TEXCOORD0;
		[[vk::location(2)]] float4 color : COLOR0;
	};

	struct PSInput
	{
		float4 position : SV_Position;
		float2 uv : TEXCOORD0;
		float4 color    : COLOR0;
		uint drawID : SV_InstanceID;
		float clip[4] : SV_ClipDistance;
	};

	struct PushConstants
	{
		uint64_t elementBDA;
		uint64_t elementCount;
		float2 scale;
		float2 translate;
		float4 viewport;
	};
	
#else
	struct PushConstants
	{
		uint64_t elementBDA;
		uint64_t elementCount;
		float scale[2];
		float translate[2];
		float viewport[4];
	};
#endif // __HLSL_VERSION

struct emulated_snorm16_t2
{
	#ifdef __HLSL_VERSION
	float32_t2 unpack() // returns in NDC [-1, 1] range
	{
		return clamp(float32_t2(x, y) / 32767.0f, -1.f, +1.f);
	}
	#endif

	int16_t x;
	int16_t y;
};

struct PerObjectData 
{
	emulated_snorm16_t2 aabbMin;
	emulated_snorm16_t2 aabbMax;
	uint32_t texId;
	uint32_t padding;
};