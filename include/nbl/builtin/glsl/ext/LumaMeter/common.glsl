// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_GLSL_EXT_LUMA_METER_COMMON_INCLUDED_
#define _NBL_GLSL_EXT_LUMA_METER_COMMON_INCLUDED_

#include <nbl/builtin/glsl/macros.glsl>
#include <nbl/builtin/glsl/math/functions.glsl>

#ifndef _NBL_GLSL_EXT_LUMA_METER_UNIFORMS_DEFINED_
#define _NBL_GLSL_EXT_LUMA_METER_UNIFORMS_DEFINED_
struct nbl_glsl_ext_LumaMeter_Uniforms_t
{
	vec2 meteringWindowScale;
	vec2 meteringWindowOffset;
};
#endif


#define _NBL_GLSL_EXT_LUMA_METER_MODE_GEOM_MEAN 0
#define _NBL_GLSL_EXT_LUMA_METER_MODE_MEDIAN 1

#if NBL_GLSL_EQUAL(_NBL_GLSL_EXT_LUMA_METER_MODE_DEFINED_,_NBL_GLSL_EXT_LUMA_METER_MODE_MEDIAN)
    #ifndef _NBL_GLSL_EXT_LUMA_METER_BIN_COUNT
        #error "You need to define _NBL_GLSL_EXT_LUMA_METER_BIN_COUNT !"
    #endif
    #if NBL_GLSL_NOT_EQUAL(_NBL_GLSL_WORKGROUP_SIZE_,_NBL_GLSL_EXT_LUMA_METER_BIN_COUNT)
	    #error "_NBL_GLSL_WORKGROUP_SIZE_ does not equal _NBL_GLSL_EXT_LUMA_METER_BIN_COUNT"
    #endif 
    #ifndef _NBL_GLSL_EXT_LUMA_METER_BIN_GLOBAL_REPLICATION
        #error "You need to define _NBL_GLSL_EXT_LUMA_METER_BIN_GLOBAL_REPLICATION !"
    #endif

    #ifdef _NBL_GLSL_EXT_LUMA_METER_FIRST_PASS_DEFINED_
    	#define _NBL_GLSL_EXT_LUMA_METER_PADDED_BIN_COUNT (_NBL_GLSL_EXT_LUMA_METER_BIN_COUNT+1)

	    #ifndef _NBL_GLSL_EXT_LUMA_METER_LOCAL_REPLICATION_POW_DEFINED_
	    #define _NBL_GLSL_EXT_LUMA_METER_LOCAL_REPLICATION_POW_DEFINED_ 3
	    #endif

	    #define _NBL_GLSL_EXT_LUMA_METER_LOCAL_REPLICATION (1<<_NBL_GLSL_EXT_LUMA_METER_LOCAL_REPLICATION_POW_DEFINED_)
	    #define _NBL_GLSL_EXT_LUMA_METER_SHARED_SIZE_NEEDED_IMPL_ (_NBL_GLSL_EXT_LUMA_METER_PADDED_BIN_COUNT*_NBL_GLSL_EXT_LUMA_METER_LOCAL_REPLICATION)
    #else
        #define _NBL_GLSL_EXT_LUMA_METER_SHARED_SIZE_NEEDED_IMPL_ (_NBL_GLSL_EXT_LUMA_METER_BIN_COUNT*2)
    #endif

    #include <nbl/builtin/glsl/workgroup/shared_arithmetic.glsl>
    // correct for subgroup emulation stuff
    #if NBL_GLSL_GREATER(_NBL_GLSL_EXT_LUMA_METER_SHARED_SIZE_NEEDED_IMPL_,_NBL_GLSL_WORKGROUP_ARITHMETIC_SHARED_SIZE_NEEDED_)
        #define _NBL_GLSL_EXT_LUMA_METER_SHARED_SIZE_NEEDED_ _NBL_GLSL_EXT_LUMA_METER_SHARED_SIZE_NEEDED_IMPL_
    #else
        #define _NBL_GLSL_EXT_LUMA_METER_SHARED_SIZE_NEEDED_ _NBL_GLSL_WORKGROUP_ARITHMETIC_SHARED_SIZE_NEEDED_
    #endif

    #if NBL_GLSL_NOT_EQUAL(NBL_GLSL_AND(NBL_GLSL_SUB(_NBL_GLSL_EXT_LUMA_METER_MAX_LUMA_DEFINED_,_NBL_GLSL_EXT_LUMA_METER_MIN_LUMA_DEFINED_),_NBL_GLSL_EXT_LUMA_METER_BIN_COUNT-1),0)
	    #error "The number of bins must evenly divide the histogram range!"
    #endif

    #define _NBL_GLSL_EXT_LUMA_METER_BIN_GLOBAL_COUNT (_NBL_GLSL_EXT_LUMA_METER_BIN_COUNT*_NBL_GLSL_EXT_LUMA_METER_BIN_GLOBAL_REPLICATION)
    struct nbl_glsl_ext_LumaMeter_output_t
    {
		uint packedHistogram[_NBL_GLSL_EXT_LUMA_METER_BIN_GLOBAL_COUNT];
    };
#elif _NBL_GLSL_EXT_LUMA_METER_MODE_DEFINED_==_NBL_GLSL_EXT_LUMA_METER_MODE_GEOM_MEAN
    #ifdef _NBL_GLSL_EXT_LUMA_METER_FIRST_PASS_DEFINED_
        #include "nbl/builtin/glsl/workgroup/shared_arithmetic.glsl"
	    #define _NBL_GLSL_EXT_LUMA_METER_SHARED_SIZE_NEEDED_ NBL_GLSL_EVAL(_NBL_GLSL_WORKGROUP_ARITHMETIC_SHARED_SIZE_NEEDED_)
    #else
        #define _NBL_GLSL_EXT_LUMA_METER_SHARED_SIZE_NEEDED_ 0
    #endif

    struct nbl_glsl_ext_LumaMeter_output_t
    {
        uint unormAverage;
    };
#else
#error "Unsupported Metering Mode!"
#endif


#define nbl_glsl_ext_LumaMeter_output_SPIRV_CROSS_is_dumb_t uint


#ifndef _NBL_GLSL_EXT_LUMA_METER_UNIFORMS_SET_DEFINED_
#define _NBL_GLSL_EXT_LUMA_METER_UNIFORMS_SET_DEFINED_ 0
#endif

#ifndef _NBL_GLSL_EXT_LUMA_METER_UNIFORMS_BINDING_DEFINED_
#define _NBL_GLSL_EXT_LUMA_METER_UNIFORMS_BINDING_DEFINED_ 0
#endif


#ifndef _NBL_GLSL_EXT_LUMA_METER_OUTPUT_SET_DEFINED_
#define _NBL_GLSL_EXT_LUMA_METER_OUTPUT_SET_DEFINED_ 0
#endif

#ifndef _NBL_GLSL_EXT_LUMA_METER_OUTPUT_BINDING_DEFINED_
#define _NBL_GLSL_EXT_LUMA_METER_OUTPUT_BINDING_DEFINED_ 1
#endif


#ifndef _NBL_GLSL_EXT_LUMA_METER_INPUT_IMAGE_SET_DEFINED_
#define _NBL_GLSL_EXT_LUMA_METER_INPUT_IMAGE_SET_DEFINED_ 0
#endif

#ifndef _NBL_GLSL_EXT_LUMA_METER_INPUT_IMAGE_BINDING_DEFINED_
#define _NBL_GLSL_EXT_LUMA_METER_INPUT_IMAGE_BINDING_DEFINED_ 2
#endif


#ifdef _NBL_GLSL_EXT_LUMA_METER_FIRST_PASS_DEFINED_

#ifndef _NBL_GLSL_EXT_LUMA_METER_PUSH_CONSTANTS_DEFINED_
#define _NBL_GLSL_EXT_LUMA_METER_PUSH_CONSTANTS_DEFINED_
layout(push_constant) uniform PushConstants
{
	int currentFirstPassOutput;
} pc;
#endif

#ifndef _NBL_GLSL_EXT_LUMA_METER_OUTPUT_DESCRIPTOR_DEFINED_
#define _NBL_GLSL_EXT_LUMA_METER_OUTPUT_DESCRIPTOR_DEFINED_
layout(set=_NBL_GLSL_EXT_LUMA_METER_OUTPUT_SET_DEFINED_, binding=_NBL_GLSL_EXT_LUMA_METER_OUTPUT_BINDING_DEFINED_) restrict coherent buffer OutputBuffer
{
	nbl_glsl_ext_LumaMeter_output_t outParams[];
};
#endif

#ifndef _NBL_GLSL_EXT_LUMA_METER_INPUT_IMAGE_DESCRIPTOR_DEFINED_
#define _NBL_GLSL_EXT_LUMA_METER_INPUT_IMAGE_DESCRIPTOR_DEFINED_
layout(set=_NBL_GLSL_EXT_LUMA_METER_INPUT_IMAGE_SET_DEFINED_, binding=_NBL_GLSL_EXT_LUMA_METER_INPUT_IMAGE_BINDING_DEFINED_) uniform sampler2DArray inputImage;
#endif

#endif // _NBL_GLSL_EXT_LUMA_METER_FIRST_PASS_DEFINED_


#ifdef _NBL_GLSL_SCRATCH_SHARED_DEFINED_
    #if NBL_GLSL_LESS(_NBL_GLSL_SCRATCH_SHARED_SIZE_DEFINED_,_NBL_GLSL_EXT_LUMA_METER_SHARED_SIZE_NEEDED_)
        #error "Not enough shared memory declared for ext::LumaMeter!"
    #endif
#else
    #if NBL_GLSL_GREATER(_NBL_GLSL_EXT_LUMA_METER_SHARED_SIZE_NEEDED_,0)
        #define _NBL_GLSL_SCRATCH_SHARED_DEFINED_ histogram
        #define _NBL_GLSL_SCRATCH_SHARED_SIZE_DEFINED_ _NBL_GLSL_EXT_LUMA_METER_SHARED_SIZE_NEEDED_
        shared uint _NBL_GLSL_SCRATCH_SHARED_DEFINED_[_NBL_GLSL_EXT_LUMA_METER_SHARED_SIZE_NEEDED_];
    #endif
#endif



#if NBL_GLSL_EQUAL(_NBL_GLSL_EXT_LUMA_METER_MODE_DEFINED_,_NBL_GLSL_EXT_LUMA_METER_MODE_MEDIAN)
    #ifdef _NBL_GLSL_EXT_LUMA_METER_FIRST_PASS_DEFINED_
        #ifndef _NBL_GLSL_EXT_LUMA_METER_CLEAR_HISTOGRAM_FUNC_DECLARED_
        #define _NBL_GLSL_EXT_LUMA_METER_CLEAR_HISTOGRAM_FUNC_DECLARED_
        void nbl_glsl_ext_LumaMeter_clearHistogram();
        #endif
    #endif

    struct nbl_glsl_ext_LumaMeter_PassInfo_t
    {
        uvec2 percentileRange; // (lowerPercentile,upperPercentile)
    };
#elif _NBL_GLSL_EXT_LUMA_METER_MODE_DEFINED_==_NBL_GLSL_EXT_LUMA_METER_MODE_GEOM_MEAN
    struct nbl_glsl_ext_LumaMeter_PassInfo_t
    {
        float rcpFirstPassWGCount;
    };
#endif



#define _NBL_GLSL_EXT_LUMA_METER_GEOM_MEAN_MAX_WG_INCREMENT 0x1000u
#ifdef _NBL_GLSL_EXT_LUMA_METER_FIRST_PASS_DEFINED_
    #ifndef _NBL_GLSL_EXT_LUMA_METER_GET_NEXT_LUMA_OUTPUT_OFFSET_FUNC_DECLARED_
    #define _NBL_GLSL_EXT_LUMA_METER_GET_NEXT_LUMA_OUTPUT_OFFSET_FUNC_DECLARED_
    int nbl_glsl_ext_LumaMeter_getNextLumaOutputOffset();
    #endif

    #ifndef _NBL_GLSL_EXT_LUMA_METER_GET_NEXT_LUMA_OUTPUT_OFFSET_FUNC_DEFINED_
    #define _NBL_GLSL_EXT_LUMA_METER_GET_NEXT_LUMA_OUTPUT_OFFSET_FUNC_DEFINED_
    int nbl_glsl_ext_LumaMeter_getNextLumaOutputOffset()
    {
        return (pc.currentFirstPassOutput!=0 ? 0:textureSize(inputImage,0).z)+int(gl_WorkGroupID.z);
    }
    #endif


    #ifndef _NBL_GLSL_EXT_LUMA_METER_GET_CURRENT_LUMA_OUTPUT_OFFSET_FUNC_DECLARED_
    #define _NBL_GLSL_EXT_LUMA_METER_GET_CURRENT_LUMA_OUTPUT_OFFSET_FUNC_DECLARED_
    int nbl_glsl_ext_LumaMeter_getCurrentLumaOutputOffset();
    #endif

    #ifndef _NBL_GLSL_EXT_LUMA_METER_GET_CURRENT_LUMA_OUTPUT_OFFSET_FUNC_DEFINED_
    #define _NBL_GLSL_EXT_LUMA_METER_GET_CURRENT_LUMA_OUTPUT_OFFSET_FUNC_DEFINED_
    int nbl_glsl_ext_LumaMeter_getCurrentLumaOutputOffset()
    {
        return (pc.currentFirstPassOutput!=0 ? textureSize(inputImage,0).z:0)+int(gl_WorkGroupID.z);
    }
    #endif


    #if NBL_GLSL_EQUAL(_NBL_GLSL_EXT_LUMA_METER_MODE_DEFINED_,_NBL_GLSL_EXT_LUMA_METER_MODE_MEDIAN)
        void nbl_glsl_ext_LumaMeter_clearHistogram()
        {
            // TODO: redo how we clear
	        for (int i=0; i<_NBL_GLSL_EXT_LUMA_METER_LOCAL_REPLICATION; i++)
		        _NBL_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex+i*_NBL_GLSL_WORKGROUP_SIZE_] = 0u;
            #if NBL_GLSL_GREATER(_NBL_GLSL_EXT_LUMA_METER_LOCAL_REPLICATION_POW_DEFINED_,0)
	            if (gl_LocalInvocationIndex<_NBL_GLSL_EXT_LUMA_METER_LOCAL_REPLICATION)
		            _NBL_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex+_NBL_GLSL_EXT_LUMA_METER_LOCAL_REPLICATION*_NBL_GLSL_WORKGROUP_SIZE_] = 0u;
            #endif
        }
    #endif


    #ifndef _NBL_GLSL_EXT_LUMA_METER_CLEAR_FIRST_PASS_OUTPUT_FUNC_DECLARED_
    #define _NBL_GLSL_EXT_LUMA_METER_CLEAR_FIRST_PASS_OUTPUT_FUNC_DECLARED_
    void nbl_glsl_ext_LumaMeter_clearFirstPassOutput();
    #endif

    #ifndef _NBL_GLSL_EXT_LUMA_METER_CLEAR_FIRST_PASS_OUTPUT_FUNC_DEFINED_
    #define _NBL_GLSL_EXT_LUMA_METER_CLEAR_FIRST_PASS_OUTPUT_FUNC_DEFINED_
    void nbl_glsl_ext_LumaMeter_clearFirstPassOutput()
    {
        #if NBL_GLSL_EQUAL(_NBL_GLSL_EXT_LUMA_METER_MODE_DEFINED_,_NBL_GLSL_EXT_LUMA_METER_MODE_MEDIAN)
            uint globalIndex = nbl_glsl_dot(uvec3(gl_LocalInvocationIndex,gl_WorkGroupID.xy),uvec3(1u,_NBL_GLSL_EXT_LUMA_METER_BIN_COUNT,gl_NumWorkGroups.x*_NBL_GLSL_EXT_LUMA_METER_BIN_COUNT));
            if (globalIndex<_NBL_GLSL_EXT_LUMA_METER_BIN_GLOBAL_COUNT)
            {
    		    outParams[nbl_glsl_ext_LumaMeter_getNextLumaOutputOffset()].packedHistogram[globalIndex] = 0u;
            }
        #else
		    if (all(equal(uvec2(0,0),gl_GlobalInvocationID.xy)))
		        outParams[nbl_glsl_ext_LumaMeter_getNextLumaOutputOffset()].unormAverage = 0u;
        #endif
    }
    #endif


    #if NBL_GLSL_EQUAL(_NBL_GLSL_EXT_LUMA_METER_MODE_DEFINED_,_NBL_GLSL_EXT_LUMA_METER_MODE_MEDIAN)
        #define nbl_glsl_ext_LumaMeter_WriteOutValue_t uint
    #else
        #define nbl_glsl_ext_LumaMeter_WriteOutValue_t float
    #endif

    #ifndef _NBL_GLSL_EXT_LUMA_METER_SET_FIRST_OUTPUT_FUNC_DECLARED_
    #define _NBL_GLSL_EXT_LUMA_METER_SET_FIRST_OUTPUT_FUNC_DECLARED_
    void nbl_glsl_ext_LumaMeter_setFirstPassOutput(in nbl_glsl_ext_LumaMeter_WriteOutValue_t writeOutVal);
    #endif

    #ifndef _NBL_GLSL_EXT_LUMA_METER_SET_FIRST_OUTPUT_FUNC_DEFINED_
    #define _NBL_GLSL_EXT_LUMA_METER_SET_FIRST_OUTPUT_FUNC_DEFINED_
    void nbl_glsl_ext_LumaMeter_setFirstPassOutput(in nbl_glsl_ext_LumaMeter_WriteOutValue_t writeOutVal)
    {
        int layerIndex = nbl_glsl_ext_LumaMeter_getCurrentLumaOutputOffset();
        #if NBL_GLSL_EQUAL(_NBL_GLSL_EXT_LUMA_METER_MODE_DEFINED_,_NBL_GLSL_EXT_LUMA_METER_MODE_MEDIAN)
            uint globalIndex = gl_LocalInvocationIndex;
            // assert(_NBL_GLSL_EXT_LUMA_METER_BIN_COUNT==_NBL_GLSL_WORKGROUP_SIZE_)
            const uint workgroupHash = gl_WorkGroupID.x+gl_WorkGroupID.y*gl_NumWorkGroups.x;
            globalIndex += (workgroupHash&uint(_NBL_GLSL_EXT_LUMA_METER_BIN_GLOBAL_REPLICATION-1))*_NBL_GLSL_WORKGROUP_SIZE_;
		    atomicAdd(outParams[layerIndex].packedHistogram[globalIndex],writeOutVal);
        #else
		    if (gl_LocalInvocationIndex==0u)
		    {
			    float normalizedAvg = writeOutVal/float(_NBL_GLSL_WORKGROUP_SIZE_);
			    uint incrementor = uint(float(_NBL_GLSL_EXT_LUMA_METER_GEOM_MEAN_MAX_WG_INCREMENT)*normalizedAvg+0.5);
			    atomicAdd(outParams[layerIndex].unormAverage,incrementor);
		    }
        #endif
    }
    #endif
#endif // _NBL_GLSL_EXT_LUMA_METER_FIRST_PASS_DEFINED_



#ifndef _NBL_GLSL_EXT_LUMA_METER_IMPL_GET_MEASURED_LUMA_FUNC_DECLARED_
#define _NBL_GLSL_EXT_LUMA_METER_IMPL_GET_MEASURED_LUMA_FUNC_DECLARED_
float nbl_glsl_ext_LumaMeter_impl_getMeasuredLumaLog2(in nbl_glsl_ext_LumaMeter_output_SPIRV_CROSS_is_dumb_t firstPassOutput, in nbl_glsl_ext_LumaMeter_PassInfo_t info);
#endif


#ifndef _NBL_GLSL_EXT_LUMA_METER_IMPL_GET_MEASURED_LUMA_FUNC_DEFINED_
#define _NBL_GLSL_EXT_LUMA_METER_IMPL_GET_MEASURED_LUMA_FUNC_DEFINED_

#if NBL_GLSL_EQUAL(_NBL_GLSL_EXT_LUMA_METER_MODE_DEFINED_,_NBL_GLSL_EXT_LUMA_METER_MODE_MEDIAN)
    #include <nbl/builtin/glsl/workgroup/arithmetic.glsl>
    #include <nbl/builtin/glsl/workgroup/shuffle.glsl>
#endif

float nbl_glsl_ext_LumaMeter_impl_getMeasuredLumaLog2(in nbl_glsl_ext_LumaMeter_output_SPIRV_CROSS_is_dumb_t firstPassOutput, in nbl_glsl_ext_LumaMeter_PassInfo_t info)
{
    #if NBL_GLSL_EQUAL(_NBL_GLSL_EXT_LUMA_METER_MODE_DEFINED_,_NBL_GLSL_EXT_LUMA_METER_MODE_MEDIAN)
        uint histogramPrefix = nbl_glsl_workgroupExclusiveAdd(firstPassOutput);

        // TODO: We can do it better, and without how right now workgroup size must equal _NBL_GLSL_EXT_LUMA_METER_BIN_COUNT, but it would be good if it didn't (we could carry out many prefix sums in serial).
        // Assign whole subgroup to do a subgroup_uniform_upper_bound on lower percentile, then do the subgroup_uniform_upper_bound again but in the [previousFound,end) range.
        // a subgroup_uniform bound can be carried out by each subgroup invocation doing an upper_bound on 1/gl_SubgroupSize of the range, then find the first subgroup invocation where the found index is not 1-past-the-end
        nbl_glsl_workgroupBallot(histogramPrefix<info.percentileRange[0u]);
        uint foundLow = nbl_glsl_workgroupBallotBitCount();
        nbl_glsl_workgroupBallot(histogramPrefix<info.percentileRange[1u]);
        uint foundHigh = nbl_glsl_workgroupBallotBitCount();
        return (float(foundLow)+float(foundHigh))*0.5/float(_NBL_GLSL_EXT_LUMA_METER_BIN_COUNT-1u);
    #else
        return float(firstPassOutput)*info.rcpFirstPassWGCount/float(_NBL_GLSL_EXT_LUMA_METER_GEOM_MEAN_MAX_WG_INCREMENT);
    #endif
}
#endif
#undef _NBL_GLSL_EXT_LUMA_METER_GEOM_MEAN_MAX_WG_INCREMENT



float nbl_glsl_ext_LumaMeter_getMeasuredLumaLog2(in nbl_glsl_ext_LumaMeter_output_SPIRV_CROSS_is_dumb_t firstPassOutput, in nbl_glsl_ext_LumaMeter_PassInfo_t info)
{
    const float MinLuma = intBitsToFloat(_NBL_GLSL_EXT_LUMA_METER_MIN_LUMA_DEFINED_);
    const float MaxLuma = intBitsToFloat(_NBL_GLSL_EXT_LUMA_METER_MAX_LUMA_DEFINED_);
    return nbl_glsl_ext_LumaMeter_impl_getMeasuredLumaLog2(firstPassOutput,info)*log2(MaxLuma/MinLuma)+log2(MinLuma);
}


float nbl_glsl_ext_LumaMeter_getOptiXIntensity(in float measuredLumaLog2)
{
    return exp2(log2(0.18)-measuredLumaLog2);
}


#endif