#ifndef _IRR_EXT_LUMA_METER_C_GLSL_LUMA_BUILTIN_INCLUDE_LOADER_INCLUDED_
#define _IRR_EXT_LUMA_METER_C_GLSL_LUMA_BUILTIN_INCLUDE_LOADER_INCLUDED_

#include "irrlicht.h"

namespace irr
{
namespace ext
{
namespace LumaMeter
{

class CGLSLLumaBuiltinIncludeLoader : public asset::IBuiltinIncludeLoader
{
    public:
        static core::SRange<const asset::SPushConstantRange> getDefaultPushConstantRanges();

        static core::SRange<const video::IGPUDescriptorSetLayout::SBinding> getDefaultBindings(video::IVideoDriver* driver);

        const char* getVirtualDirectoryName() const override { return "glsl/ext/LumaMeter/"; }

        _IRR_STATIC_INLINE_CONSTEXPR uint32_t DEFAULT_INVOCATION_COUNT = 256u;

        _IRR_STATIC_INLINE_CONSTEXPR uint32_t BIN_GLOBAL_REPLICATION = 4u; // change this simultaneously with the GLSL header's `_IRR_GLSL_EXT_LUMA_METER_BIN_GLOBAL_REPLICATION`

    private:
        static std::string getCommon(const std::string&)
        {
            return
R"(#ifndef _IRR_GLSL_EXT_LUMA_METER_COMMON_INCLUDED_
#define _IRR_GLSL_EXT_LUMA_METER_COMMON_INCLUDED_


#ifndef _IRR_GLSL_EXT_LUMA_METER_INVOCATION_COUNT
#define _IRR_GLSL_EXT_LUMA_METER_INVOCATION_COUNT 256 // change this simultaneously with the constexpr in `CGLSLLumaBuiltinIncludeLoader`
#endif


#ifndef _IRR_GLSL_EXT_LUMA_METER_UNIFORMS_DEFINED_
#define _IRR_GLSL_EXT_LUMA_METER_UNIFORMS_DEFINED_
struct irr_glsl_ext_LumaMeter_Uniforms_t
{
	vec2 meteringWindowScale;
	vec2 meteringWindowOffset;
};
#endif


#define _IRR_GLSL_EXT_LUMA_METER_MODE_GEOM_MEAN 0
#define _IRR_GLSL_EXT_LUMA_METER_MODE_MODE 1

#if _IRR_GLSL_EXT_LUMA_METER_MODE_DEFINED_==_IRR_GLSL_EXT_LUMA_METER_MODE_MODE
    #define _IRR_GLSL_EXT_LUMA_METER_BIN_COUNT _IRR_GLSL_EXT_LUMA_METER_INVOCATION_COUNT

    #ifdef _IRR_GLSL_EXT_LUMA_METER_FIRST_PASS_DEFINED_
    	#define _IRR_GLSL_EXT_LUMA_METER_PADDED_BIN_COUNT (_IRR_GLSL_EXT_LUMA_METER_BIN_COUNT+1)

	    #ifndef _IRR_GLSL_EXT_LUMA_METER_LOCAL_REPLICATION_POW_DEFINED_
	    #define _IRR_GLSL_EXT_LUMA_METER_LOCAL_REPLICATION_POW_DEFINED_ 3
	    #endif

	    #define _IRR_GLSL_EXT_LUMA_METER_LOCAL_REPLICATION (1<<_IRR_GLSL_EXT_LUMA_METER_LOCAL_REPLICATION_POW_DEFINED_)
	    #define _IRR_GLSL_EXT_LUMA_METER_SHARED_SIZE_NEEDED_ (_IRR_GLSL_EXT_LUMA_METER_PADDED_BIN_COUNT*_IRR_GLSL_EXT_LUMA_METER_LOCAL_REPLICATION)
    #else
        #define _IRR_GLSL_EXT_LUMA_METER_SHARED_SIZE_NEEDED_ (_IRR_GLSL_EXT_LUMA_METER_BIN_COUNT*2)
    #endif

/* can't get this to work
    #if ((_IRR_GLSL_EXT_LUMA_METER_MAX_LUMA_DEFINED_-_IRR_GLSL_EXT_LUMA_METER_MIN_LUMA_DEFINED_)&(_IRR_GLSL_EXT_LUMA_METER_BIN_COUNT-1u))!=0
	    #error "The number of bins must evenly divide the histogram range!"
    #endif
*/
    #define _IRR_GLSL_EXT_LUMA_METER_BIN_GLOBAL_REPLICATION 4 // change this simultaneously with the constexpr in `CGLSLLumaBuiltinIncludeLoader`
    #define _IRR_GLSL_EXT_LUMA_METER_BIN_GLOBAL_COUNT (_IRR_GLSL_EXT_LUMA_METER_BIN_COUNT*_IRR_GLSL_EXT_LUMA_METER_BIN_GLOBAL_REPLICATION)
    struct irr_glsl_ext_LumaMeter_output_t
    {
		uint packedHistogram[_IRR_GLSL_EXT_LUMA_METER_BIN_GLOBAL_COUNT];
    };
#elif _IRR_GLSL_EXT_LUMA_METER_MODE_DEFINED_==_IRR_GLSL_EXT_LUMA_METER_MODE_GEOM_MEAN
    #ifdef _IRR_GLSL_EXT_LUMA_METER_FIRST_PASS_DEFINED_
	    #define _IRR_GLSL_EXT_LUMA_METER_SHARED_SIZE_NEEDED_ _IRR_GLSL_EXT_LUMA_METER_INVOCATION_COUNT
    #else
        #define _IRR_GLSL_EXT_LUMA_METER_SHARED_SIZE_NEEDED_ 0
    #endif

    struct irr_glsl_ext_LumaMeter_output_t
    {
        uint unormAverage;
    };
#else
#error "Unsupported Metering Mode!"
#endif



#ifndef _IRR_GLSL_EXT_LUMA_METER_UNIFORMS_SET_DEFINED_
#define _IRR_GLSL_EXT_LUMA_METER_UNIFORMS_SET_DEFINED_ 0
#endif

#ifndef _IRR_GLSL_EXT_LUMA_METER_UNIFORMS_BINDING_DEFINED_
#define _IRR_GLSL_EXT_LUMA_METER_UNIFORMS_BINDING_DEFINED_ 0
#endif


#ifndef _IRR_GLSL_EXT_LUMA_METER_OUTPUT_SET_DEFINED_
#define _IRR_GLSL_EXT_LUMA_METER_OUTPUT_SET_DEFINED_ 0
#endif

#ifndef _IRR_GLSL_EXT_LUMA_METER_OUTPUT_BINDING_DEFINED_
#define _IRR_GLSL_EXT_LUMA_METER_OUTPUT_BINDING_DEFINED_ 1
#endif


#ifndef _IRR_GLSL_EXT_LUMA_METER_INPUT_IMAGE_SET_DEFINED_
#define _IRR_GLSL_EXT_LUMA_METER_INPUT_IMAGE_SET_DEFINED_ 0
#endif

#ifndef _IRR_GLSL_EXT_LUMA_METER_INPUT_IMAGE_BINDING_DEFINED_
#define _IRR_GLSL_EXT_LUMA_METER_INPUT_IMAGE_BINDING_DEFINED_ 2
#endif


#ifdef _IRR_GLSL_EXT_LUMA_METER_FIRST_PASS_DEFINED_

#ifndef _IRR_GLSL_EXT_LUMA_METER_PUSH_CONSTANTS_DEFINED_
#define _IRR_GLSL_EXT_LUMA_METER_PUSH_CONSTANTS_DEFINED_
layout(push_constant) uniform PushConstants
{
	int currentFirstPassOutput;
} pc;
#endif

#ifndef _IRR_GLSL_EXT_LUMA_METER_OUTPUT_DESCRIPTOR_DEFINED_
#define _IRR_GLSL_EXT_LUMA_METER_OUTPUT_DESCRIPTOR_DEFINED_
layout(set=_IRR_GLSL_EXT_LUMA_METER_OUTPUT_SET_DEFINED_, binding=_IRR_GLSL_EXT_LUMA_METER_OUTPUT_BINDING_DEFINED_) restrict coherent buffer OutputBuffer
{
	irr_glsl_ext_LumaMeter_output_t outParams[];
};
#endif

#ifndef _IRR_GLSL_EXT_LUMA_METER_INPUT_IMAGE_DESCRIPTOR_DEFINED_
#define _IRR_GLSL_EXT_LUMA_METER_INPUT_IMAGE_DESCRIPTOR_DEFINED_
layout(set=_IRR_GLSL_EXT_LUMA_METER_INPUT_IMAGE_SET_DEFINED_, binding=_IRR_GLSL_EXT_LUMA_METER_INPUT_IMAGE_BINDING_DEFINED_) uniform sampler2DArray inputImage;
#endif

#endif


#if _IRR_GLSL_EXT_LUMA_METER_SHARED_SIZE_NEEDED_>0 && !defined(_IRR_GLSL_SCRATCH_SHARED_DEFINED_)
#define _IRR_GLSL_SCRATCH_SHARED_DEFINED_ histogram
shared uint _IRR_GLSL_SCRATCH_SHARED_DEFINED_[_IRR_GLSL_EXT_LUMA_METER_SHARED_SIZE_NEEDED_];
#elif defined(_IRR_GLSL_SCRATCH_SHARED_SIZE_DEFINED_) && _IRR_GLSL_SCRATCH_SHARED_SIZE_DEFINED_<_IRR_GLSL_EXT_LUMA_METER_SHARED_SIZE_NEEDED_
    #error "Not enough shared memory declared"
#endif



#if _IRR_GLSL_EXT_LUMA_METER_MODE_DEFINED_==_IRR_GLSL_EXT_LUMA_METER_MODE_MODE
    #ifdef _IRR_GLSL_EXT_LUMA_METER_FIRST_PASS_DEFINED_
        void irr_glsl_ext_LumaMeter_clearFirstPassOutput()
        {
            uint globalIndex = gl_LocalInvocationIndex+gl_WorkGroupID.x*_IRR_GLSL_EXT_LUMA_METER_BIN_COUNT;
            if (globalIndex<_IRR_GLSL_EXT_LUMA_METER_BIN_GLOBAL_COUNT)
            {
    		    outParams[(pc.currentFirstPassOutput!=0 ? 0:textureSize(inputImage,0).z)+int(gl_WorkGroupID.z)].packedHistogram[globalIndex] = 0u;
            }
        }

        void irr_glsl_ext_LumaMeter_setFirstPassOutput(in uint writeOutVal)
        {
            uint globalIndex = gl_LocalInvocationIndex;
            globalIndex += (gl_WorkGroupID.x&uint(_IRR_GLSL_EXT_LUMA_METER_BIN_GLOBAL_REPLICATION-1))*_IRR_GLSL_EXT_LUMA_METER_BIN_COUNT;
		    atomicAdd(outParams[(pc.currentFirstPassOutput!=0 ? textureSize(inputImage,0).z:0)+int(gl_WorkGroupID.z)].packedHistogram[globalIndex],writeOutVal);
        }
    #endif

    // TODO: move to `CGLSLScanBuiltinIncludeLoader` but clean that include up first and fix shaderc macro handling
    uint irr_glsl_workgroupExclusiveAdd(uint val)
    {
        //! Bad INEFFICIENT Kogge-Stone adder, don't implement this way!
        for (int pass=1; pass<_IRR_GLSL_EXT_LUMA_METER_BIN_COUNT; pass<<=1)
        {
            uint index = gl_LocalInvocationIndex+(pass&0x1)*_IRR_GLSL_EXT_LUMA_METER_BIN_COUNT;

            _IRR_GLSL_SCRATCH_SHARED_DEFINED_[index] = val;
            barrier();
            memoryBarrierShared();
            if (gl_LocalInvocationIndex>=pass)
                val += _IRR_GLSL_SCRATCH_SHARED_DEFINED_[index-pass];
        }
        barrier();
        memoryBarrierShared();
        return val;
    }

    // TODO: turn `upper_bound__minus_onePoT` into a macro `irr_glsl_parallel_upper_bound__minus_onePoT` and introduce lower_bound, and non-minus one variants
    #if _IRR_GLSL_EXT_LUMA_METER_BIN_COUNT&(_IRR_GLSL_EXT_LUMA_METER_BIN_COUNT-1)
        #error "Parallel Upper Bound requires the Histogram Bin Count to be PoT"
    #endif
    int upper_bound_minus_onePoT(in uint val, int arrayLenPoT)
    {
        arrayLenPoT >>= 1;
        int ret = (val<_IRR_GLSL_SCRATCH_SHARED_DEFINED_[arrayLenPoT]) ? 0:arrayLenPoT;
        for (; arrayLenPoT>0; arrayLenPoT>>=1)
        {
            int right = ret+arrayLenPoT;
            ret = (val<_IRR_GLSL_SCRATCH_SHARED_DEFINED_[right]) ? 0:right;
        }
        return ret;
    }

    struct irr_glsl_ext_LumaMeter_PassInfo_t
    {
        uvec2 percentileRange; // (lowerPercentile,upperPercentile)
    };
    float irr_glsl_ext_LumaMeter_impl_getMeasuredLumaLog2(in irr_glsl_ext_LumaMeter_output_t firstPassOutput, in irr_glsl_ext_LumaMeter_PassInfo_t info)
    {
        uint histogramVal = firstPassOutput.packedHistogram[gl_LocalInvocationIndex];
        for (int i=0; i<_IRR_GLSL_EXT_LUMA_METER_BIN_GLOBAL_REPLICATION; i++)
            histogramVal += firstPassOutput.packedHistogram[gl_LocalInvocationIndex+i*_IRR_GLSL_EXT_LUMA_METER_BIN_COUNT];

        // do the prefix sum stuff
        _IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex] = irr_glsl_workgroupExclusiveAdd(histogramVal);
        barrier();
        memoryBarrierShared();

        float foundPercentiles[2];
        bool lower2Threads = gl_LocalInvocationIndex<2u;
        if (lower2Threads)
        {
            int found = upper_bound_minus_onePoT(info.percentileRange[gl_LocalInvocationIndex],_IRR_GLSL_EXT_LUMA_METER_BIN_COUNT);

            float foundValue = float(found)/float(_IRR_GLSL_EXT_LUMA_METER_BIN_COUNT);
            _IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex] = floatBitsToUint(foundValue);
        }
        barrier();
        memoryBarrierShared();

        return (uintBitsToFloat(_IRR_GLSL_SCRATCH_SHARED_DEFINED_[0])+uintBitsToFloat(_IRR_GLSL_SCRATCH_SHARED_DEFINED_[1]))*0.5;
    }
    #undef _IRR_GLSL_EXT_LUMA_METER_BIN_GLOBAL_REPLICATION
#elif _IRR_GLSL_EXT_LUMA_METER_MODE_DEFINED_==_IRR_GLSL_EXT_LUMA_METER_MODE_GEOM_MEAN
    #define _IRR_GLSL_EXT_LUMA_METER_GEOM_MEAN_MAX_WG_INCREMENT 0x1000u

    #ifdef _IRR_GLSL_EXT_LUMA_METER_FIRST_PASS_DEFINED_
        void irr_glsl_ext_LumaMeter_clearFirstPassOutput()
        {
		    if (all(equal(uvec2(0,0),gl_GlobalInvocationID.xy)))
		        outParams[(pc.currentFirstPassOutput!=0 ? 0:textureSize(inputImage,0).z)+int(gl_WorkGroupID.z)].unormAverage = 0u;
        }

        void irr_glsl_ext_LumaMeter_setFirstPassOutput(in float writeOutVal)
        {
		    if (gl_LocalInvocationIndex==0u)
		    {
			    float normalizedAvg = writeOutVal/float(_IRR_GLSL_EXT_LUMA_METER_INVOCATION_COUNT);
			    uint incrementor = uint(float(_IRR_GLSL_EXT_LUMA_METER_GEOM_MEAN_MAX_WG_INCREMENT)*normalizedAvg+0.5);
			    atomicAdd(outParams[(pc.currentFirstPassOutput!=0 ? textureSize(inputImage,0).z:0)+int(gl_WorkGroupID.z)].unormAverage,incrementor);
		    }
        }
    #endif

    struct irr_glsl_ext_LumaMeter_PassInfo_t
    {
        float rcpFirstPassWGCount;
    };
    float irr_glsl_ext_LumaMeter_impl_getMeasuredLumaLog2(in irr_glsl_ext_LumaMeter_output_t firstPassOutput, in irr_glsl_ext_LumaMeter_PassInfo_t info)
    {
        return float(firstPassOutput.unormAverage)*info.rcpFirstPassWGCount/float(_IRR_GLSL_EXT_LUMA_METER_GEOM_MEAN_MAX_WG_INCREMENT);
    }
    #undef _IRR_GLSL_EXT_LUMA_METER_GEOM_MEAN_MAX_WG_INCREMENT
#endif


float irr_glsl_ext_LumaMeter_getMeasuredLumaLog2(in irr_glsl_ext_LumaMeter_output_t firstPassOutput, in irr_glsl_ext_LumaMeter_PassInfo_t info)
{
    const float MinLuma = intBitsToFloat(_IRR_GLSL_EXT_LUMA_METER_MIN_LUMA_DEFINED_);
    const float MaxLuma = intBitsToFloat(_IRR_GLSL_EXT_LUMA_METER_MAX_LUMA_DEFINED_);
    return irr_glsl_ext_LumaMeter_impl_getMeasuredLumaLog2(firstPassOutput,info)*log2(MaxLuma/MinLuma)+log2(MinLuma);
}


float irr_glsl_ext_LumaMeter_getOptiXIntensity(in float measuredLumaLog2)
{
    return exp2(log(0.18)-measuredLumaLog2);
}


#endif
)";
        }

    protected:
        inline irr::core::vector<std::pair<std::regex, HandleFunc_t>> getBuiltinNamesToFunctionMapping() const override
        {
            return {
                { std::regex{"common\\.glsl"}, &getCommon }
            };
        }
    };

}
}
}

#endif