#if BINNING_METHOD==0

uint getAtomicOffset() {return 0u;}

#elif BINNING_METHOD==1

uint getAtomicOffset()
{ //! MAPS ONLY TO Nvidia
    uvec2 intPix = uvec2(gl_FragCoord.xy)&uvec2(12,8);
    uvec2 tmp = intPix<<uvec2(3,4);
    return tmp.x|tmp.y;
}

#elif BINNING_METHOD==2

#ifdef INTEL
	#define RASTER_SIMD_TILE_SZ_LOG2_X 2u
	#define RASTER_SIMD_TILE_SZ_LOG2_Y 1u
#else // assume AMD
	#define RASTER_SIMD_TILE_SZ_LOG2_X 3u
	#define RASTER_SIMD_TILE_SZ_LOG2_Y 3u
#endif // INTEL

uint getAtomicOffset()
{
    uvec2 intPix = uvec2(gl_FragCoord.xy);
    intPix.x = intPix.x>>RASTER_SIMD_TILE_SZ_LOG2_X;
    intPix &= uvec2(kSqrtCUCountMask,kSqrtCUCountMask<<RASTER_SIMD_TILE_SZ_LOG2_Y);
#if   RASTER_SIMD_TILE_SZ_LOG2_Y>kHalfLog2CUCount
    intPix.y = intPix.y>>(RASTER_SIMD_TILE_SZ_LOG2_Y-kHalfLog2CUCount);
#elif RASTER_SIMD_TILE_SZ_LOG2_Y<kHalfLog2CUCount
    intPix.y = intPix.y<<(kHalfLog2CUCount-RASTER_SIMD_TILE_SZ_LOG2_Y);
#endif

    return intPix.x|intPix.y;
}

#elif BINNING_METHOD==3

#extension GL_ARB_ES3_1_compatibility: require
#extension GL_ARB_shader_ballot: require
#extension GL_ARB_gpu_shader_int64: enable
#extension GL_NV_gpu_shader5: enable

#ifdef INTEL
	#define RASTER_SIMD_TILE_SZ_LOG2_X 2u
	#define RASTER_SIMD_TILE_SZ_LOG2_Y 1u
#else // assume AMD
	#define RASTER_SIMD_TILE_SZ_LOG2_X 3u
	#define RASTER_SIMD_TILE_SZ_LOG2_Y 3u
#endif // INTEL

uint getAtomicOffset()
{
    uvec2 intPix = uvec2(gl_FragCoord.xy);
    intPix.x = intPix.x>>RASTER_SIMD_TILE_SZ_LOG2_X;
    intPix &= uvec2(kSqrtCUCountMask,kSqrtCUCountMask<<RASTER_SIMD_TILE_SZ_LOG2_Y);
#if   RASTER_SIMD_TILE_SZ_LOG2_Y>kHalfLog2CUCount
    intPix.y = intPix.y>>(RASTER_SIMD_TILE_SZ_LOG2_Y-kHalfLog2CUCount);
#elif RASTER_SIMD_TILE_SZ_LOG2_Y<kHalfLog2CUCount
    intPix.y = intPix.y<<(kHalfLog2CUCount-RASTER_SIMD_TILE_SZ_LOG2_Y);
#endif

    return intPix.x|intPix.y;
}

#elif BINNING_METHOD==4

#extension GL_NV_shader_thread_group: require
uint getAtomicOffset()
{
    return gl_SMIDNV;
}

#elif BINNING_METHOD==5

#extension GL_ARB_ES3_1_compatibility: require
#extension GL_ARB_shader_ballot: require
#extension GL_ARB_gpu_shader_int64: enable
#extension GL_NV_gpu_shader5: enable
uint getAtomicOffset()
{
    return 0;
}

#elif BINNING_METHOD==6


#extension GL_NV_shader_thread_group: require
uint getAtomicOffset()
{
    return 0;
}


#endif // BINNING_METHOD


layout(location=0) uniform uint optimizerKillerOffset;


layout(std430, binding = 0) restrict coherent buffer OutputAtomicData {
	uint outArray[];
};


void main()
{
#if BINNING_METHOD==7
    outArray[optimizerKillerOffset] = 1u;
#elif BINNING_METHOD==5||BINNING_METHOD==3
    uint64_t activeThreadMask = ballotARB(!gl_HelperInvocation);
    if (min(findLSB(uint(activeThreadMask)),findLSB(uint(activeThreadMask>>32u))+32u)==gl_SubGroupInvocationARB)
        atomicAdd(outArray[optimizerKillerOffset+getAtomicOffset()],bitCount(uint(activeThreadMask))+bitCount(uint(activeThreadMask>>32u)));
#elif BINNING_METHOD==6||BINNING_METHOD==4
    uint activeThreadMask = ballotThreadNV(!gl_HelperThreadNV);
    if (findLSB(activeThreadMask)==gl_ThreadInWarpNV)
        atomicAdd(outArray[optimizerKillerOffset+getAtomicOffset()],bitCount(activeThreadMask));
#else
    atomicAdd(outArray[optimizerKillerOffset+getAtomicOffset()],1u);
#endif // BINNING_METHOD
}

