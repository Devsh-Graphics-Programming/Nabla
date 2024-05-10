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
	};

	struct PushConstants
	{
		float2 scale;
		float2 translate;
	};
#else
	struct PushConstants
	{
		float scale[2];
		float translate[2];
	};
#endif // __HLSL_VERSION
