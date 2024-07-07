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
		float4 color    : COLOR0;
		float2 uv : TEXCOORD0;
		float clip[2] : SV_ClipDistance;
	};

	struct PushConstants
	{
		uint64_t elementBDA;
		uint64_t elementCount;
		float2 scale;
		float2 translate;
		float4 viewport;
	};
	
	struct PerObjectData 
	{
		float4 scissor;
		uint texId;
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
	
	struct PerObjectData
	{
	  VkRect2D scissor;
	  uint32_t texId = 0;
	};
#endif // __HLSL_VERSION
