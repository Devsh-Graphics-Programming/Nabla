#ifndef _NBL_IMGUI_EXT_PSINPUT_HLSL_
#define _NBL_IMGUI_EXT_PSINPUT_HLSL_

#ifdef __HLSL_VERSION
namespace nbl
{
namespace ext
{
namespace imgui
{
	struct PSInput
	{
		float32_t4 position : SV_Position;
		float32_t2 uv : TEXCOORD0;
		float32_t4 color : COLOR0;
		float32_t4 clip : SV_ClipDistance;
		uint drawID : SV_InstanceID;
	};
}
}
}
#endif // __HLSL_VERSION

#endif // _NBL_IMGUI_EXT_PSINPUT_HLSL_
