#ifndef _NBL_BUILTIN_GLSL_SAMPLING_QUANTIZED_SEQUENCE_INCLUDED_
#define _NBL_BUILTIN_GLSL_SAMPLING_QUANTIZED_SEQUENCE_INCLUDED_

// samples are quantized 3 dimensions at a time
#define nbl_glsl_sampling_quantized3D uvec2

vec3 nbl_glsl_sampling_decodeSample3Dimensions(in nbl_glsl_sampling_quantized3D quant3D, in uvec3 scrambleKey)
{
	// We don't even need to mask off the lower bits of X and Y, since they get lost upon the FP32 multiplication
	const uvec3 seqVal = uvec3(
		quant3D[0],
		quant3D[1],
		(quant3D[0]<<21)|((quant3D[1]&0x07FFu)<<10)
	);
	// Due to mantissa truncation, the constant multiplier is not an reciprocal of 2^32-1, because more than one value needs to map to 1.f
	return vec3(seqVal^scrambleKey)*uintBitsToFloat(0x2f800004u);
}

#endif
