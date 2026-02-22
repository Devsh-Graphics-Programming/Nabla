#ifndef _NBL_IMGUI_EXT_PC_HLSL_
#define _NBL_IMGUI_EXT_PC_HLSL_

// TODO: have only unified.hlsl uber shader and common.hlsl then update imgui cpp files, doing a quick workaround for my prebuilds
#include "common.hlsl"
[[vk::push_constant]] struct nbl::ext::imgui::PushConstants pc;

#endif // _NBL_IMGUI_EXT_PC_HLSL_
