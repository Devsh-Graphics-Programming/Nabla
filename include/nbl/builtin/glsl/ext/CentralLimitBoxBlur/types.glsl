#ifndef _NBL_GLSL_EXT_BLUR_TYPES_INCLUDED_
#define _NBL_GLSL_EXT_BLUR_TYPES_INCLUDED_

#if _NBL_GLSL_EXT_BLUR_HALF_STORAGE_!=0

// Todo(achal): For 16 bit float images, making one element of the SSBO correspond to one _pixel_ (not one _channel_) of
// the input image seems much more efficient. For instance, if we have a MxN image with 4 channels, if we store it
// as encoded values using `uvec2` (assuming the input image is 4 channeled always) we would need only M*N*sizeof(uvec2) = 8*M*N
// bytes of temporary SSBO storage, whereas if we keep using a `float` for every channel we would need M*N*channel_count*sizeof(float)
// bytes of temporary SSBO storage --this for 4 channel and 3 channel images have the most wasted memory 2X and (4/3)X respectively.

// #define nbl_glsl_ext_Blur_Storage_t uvec2
// 
// nbl_glsl_ext_Blur_Storage_t nbl_glsl_ext_Blur_Storage_t_get(vec4 val)
// {
// 		return nbl_glsl_ext_Blur_Storage_t(packHalf2x16(val.rg), packHalf2x16(val.ba));
// }
// 
// vec4 nbl_glsl_ext_Blur_Storage_t_set(nbl_glsl_ext_Blur_Storage_t val)
// {
// 		return vec4(unpackHalf2x16(val.x), unpackHalf2x16(val.y));
// }

#define nbl_glsl_ext_Blur_Storage_t float

nbl_glsl_ext_Blur_Storage_t nbl_glsl_ext_Blur_Storage_t_get(float val)
{
	return val;
}

float nbl_glsl_ext_Blur_Storage_t_set(nbl_glsl_ext_Blur_Storage_t val)
{
	return val;
}

#else

#define nbl_glsl_ext_Blur_Storage_t float

nbl_glsl_ext_Blur_Storage_t nbl_glsl_ext_Blur_Storage_t_get(float val)
{
	return val;
}

float nbl_glsl_ext_Blur_Storage_t_set(nbl_glsl_ext_Blur_Storage_t val)
{
	return val;
}

#endif

#endif