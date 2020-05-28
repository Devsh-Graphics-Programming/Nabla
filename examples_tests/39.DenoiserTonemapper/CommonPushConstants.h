#define MAX_MEDIAN_FILTER_RADIUS 2

#ifdef __cplusplus
	#define int int32_t
	#define uint uint32_t
	#define mat3 irr::core::matrix3x4SIMD
#endif
struct CommonPushConstants
{
	uint inImageTexelOffset[3];
	uint inImageTexelPitch[3];
	uint outImageOffset[3];
	uint imageWidth;

	// for the median filter
	int medianFilterRadius;
	
	// luma meter and tonemapping var but also for denoiser
	uint intensityBufferDWORDOffset;
	float denoiserExposureBias;

	// for the tonemapper
	float tonemapperParams[2];

	mat3 normalMatrix;
};
#ifdef __cplusplus
	#undef int
	#undef uint
	#undef mat3
#endif