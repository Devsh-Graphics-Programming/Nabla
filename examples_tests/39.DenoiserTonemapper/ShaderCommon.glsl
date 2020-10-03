// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define COMPUTE_WG_SIZE 256u
layout(local_size_x=COMPUTE_WG_SIZE) in;

layout(constant_id = 1) const uint EII_COLOR = 0u;
layout(constant_id = 2) const uint EII_ALBEDO = 1u;
layout(constant_id = 3) const uint EII_NORMAL = 2u;

#include "./CommonPushConstants.h"

layout(push_constant, row_major) uniform PushConstants{
	CommonPushConstants data;
} pc;
#define _IRR_GLSL_EXT_LUMA_METER_PUSH_CONSTANTS_DEFINED_


#define SHARED_CHANNELS 3u
// the amount of memory needed for luma metering is bigger than interleaving
#define _IRR_GLSL_SCRATCH_SHARED_SIZE_DEFINED_ ((COMPUTE_WG_SIZE+1u)*8u)
shared uint repackBuffer[_IRR_GLSL_SCRATCH_SHARED_SIZE_DEFINED_];
#define _IRR_GLSL_SCRATCH_SHARED_DEFINED_ repackBuffer


// luma metering stuff
// those don't really influence anything but need to let the header know that we're using the same number of invocations as bins
#define _IRR_GLSL_EXT_LUMA_METER_DISPATCH_SIZE_X_DEFINED_ 256
#define _IRR_GLSL_EXT_LUMA_METER_DISPATCH_SIZE_Y_DEFINED_ 1

#define _IRR_GLSL_EXT_LUMA_METER_MIN_LUMA_DEFINED_ floatBitsToInt(1.0/4096.0)
#define _IRR_GLSL_EXT_LUMA_METER_MAX_LUMA_DEFINED_ floatBitsToInt(32768.0)
// its the mean mode
#define _IRR_GLSL_EXT_LUMA_METER_MODE_DEFINED_ 1

#include "irr/builtin/glsl/colorspace/EOTF.glsl"
#include "irr/builtin/glsl/colorspace/encodeCIEXYZ.glsl"
#include "irr/builtin/glsl/colorspace/decodeCIEXYZ.glsl"
#include "irr/builtin/glsl/colorspace/OETF.glsl"

#define _IRR_GLSL_EXT_LUMA_METER_EOTF_DEFINED_ irr_glsl_eotf_identity
#define _IRR_GLSL_EXT_LUMA_METER_XYZ_CONVERSION_MATRIX_DEFINED_ irr_glsl_sRGBtoXYZ
#define _IRR_GLSL_EXT_LUMA_METER_GET_COLOR_DEFINED_
// won't be using an image as input (we'll provide the colors ourselves)
#define _IRR_GLSL_EXT_LUMA_METER_INPUT_IMAGE_DESCRIPTOR_DEFINED_
#define _IRR_GLSL_EXT_LUMA_METER_OUTPUT_BINDING_DEFINED_ 2
#define _IRR_GLSL_EXT_LUMA_METER_GET_NEXT_LUMA_OUTPUT_OFFSET_FUNC_DEFINED_
#define _IRR_GLSL_EXT_LUMA_METER_GET_CURRENT_LUMA_OUTPUT_OFFSET_FUNC_DEFINED_

#define _IRR_GLSL_EXT_LUMA_METER_INVOCATION_COUNT (_IRR_GLSL_EXT_LUMA_METER_DISPATCH_SIZE_X_DEFINED_*_IRR_GLSL_EXT_LUMA_METER_DISPATCH_SIZE_Y_DEFINED_)
#define _IRR_GLSL_EXT_LUMA_METER_BIN_COUNT _IRR_GLSL_EXT_LUMA_METER_INVOCATION_COUNT
#define _IRR_GLSL_EXT_LUMA_METER_BIN_GLOBAL_REPLICATION 4
#ifdef _IRR_GLSL_EXT_LUMA_METER_FIRST_PASS_DEFINED_
	#include "irr/builtin/glsl/ext/LumaMeter/impl.glsl"

	// need to override the offset and color provision functions
	int irr_glsl_ext_LumaMeter_getNextLumaOutputOffset()
	{
		return pc.data.beforeDenoise!=0u ? 1:0;
	}

	int irr_glsl_ext_LumaMeter_getCurrentLumaOutputOffset()
	{
		return pc.data.beforeDenoise!=0u ? 0:1;
	}

	vec3 globalPixelData;
	vec3 irr_glsl_ext_LumaMeter_getColor(bool wgExecutionMask)
	{
		return globalPixelData;
	}
#else
	#include "irr/builtin/glsl/ext/LumaMeter/common.glsl"
#endif