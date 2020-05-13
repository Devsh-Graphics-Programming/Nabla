#ifndef _IRR_EXT_LUMA_METER_C_GLSL_LUMA_BUILTIN_INCLUDE_LOADER_INCLUDED_
#define _IRR_EXT_LUMA_METER_C_GLSL_LUMA_BUILTIN_INCLUDE_LOADER_INCLUDED_

#include "irr/asset/IBuiltinIncludeLoader.h"

namespace irr
{
namespace ext
{
namespace LumaMeter
{

class CGLSLLumaBuiltinIncludeLoader : public irr::asset::IBuiltinIncludeLoader
{
    public:
        const char* getVirtualDirectoryName() const override { return "glsl/ext/LumaMeter/"; }

        _IRR_STATIC_INLINE_CONSTEXPR uint32_t DISPATCH_SIZE = 16u; // change this simultaneously with the GLSL header's `_IRR_GLSL_EXT_LUMA_METER_DISPATCH_SIZE_DEFINED_`
        _IRR_STATIC_INLINE_CONSTEXPR uint32_t MODE_BIN_COUNT = DISPATCH_SIZE*DISPATCH_SIZE;

    private:
        static std::string getCommon(const std::string&)
        {
            return
R"(#ifndef _IRR_GLSL_EXT_LUMA_METER_COMMON_INCLUDED_
#define _IRR_GLSL_EXT_LUMA_METER_COMMON_INCLUDED_


#ifndef _IRR_GLSL_EXT_LUMA_METER_LAYERS_TO_PROCESS_DEFINED_
#define _IRR_GLSL_EXT_LUMA_METER_LAYERS_TO_PROCESS_DEFINED_ 1
#endif


#ifndef _IRR_GLSL_EXT_LUMA_METER_DISPATCH_SIZE_DEFINED_
#define _IRR_GLSL_EXT_LUMA_METER_DISPATCH_SIZE_DEFINED_ 16 // change this simultaneously with the constexpr in `CGLSLLumaBuiltinIncludeLoader`
#endif

#define _IRR_GLSL_EXT_LUMA_METER_INVOCATION_COUNT (_IRR_GLSL_EXT_LUMA_METER_DISPATCH_SIZE_DEFINED_*_IRR_GLSL_EXT_LUMA_METER_DISPATCH_SIZE_DEFINED_)


#define _IRR_GLSL_EXT_LUMA_METER_MODE_GEOM_MEAN 0
#define _IRR_GLSL_EXT_LUMA_METER_MODE_MODE 1

#if _IRR_GLSL_EXT_LUMA_METER_MODE_DEFINED_==_IRR_GLSL_EXT_LUMA_METER_MODE_MODE
    #define _IRR_GLSL_EXT_LUMA_METER_BIN_COUNT _IRR_GLSL_EXT_LUMA_METER_INVOCATION_COUNT

    #if (_IRR_GLSL_EXT_LUMA_METER_MAX_LUMA_DEFINED_-_IRR_GLSL_EXT_LUMA_METER_MIN_LUMA_DEFINED_)%%_IRR_GLSL_EXT_LUMA_METER_BIN_COUNT!=0
	    #error "The number of bins must evenly divide the histogram range!"
    #endif

    #define _IRR_GLSL_EXT_LUMA_METER_BIN_GLOBAL_REPLICATION 4
    struct irr_glsl_ext_LumaMeter_output_t
    {
		uint packedHistogram[_IRR_GLSL_EXT_LUMA_METER_BIN_COUNT*_IRR_GLSL_EXT_LUMA_METER_BIN_GLOBAL_REPLICATION];
    };

    void irr_glsl_ext_LumaMeter_setFirstPassOutput(out irr_glsl_ext_LumaMeter_output_t firstPassOutput, in uint writeOutVal)
    {
		uint globalIndex = gl_LocalInvocationIndex;
        globalIndex += (gl_WorkGroupID.x&uint(_IRR_GLSL_EXT_LUMA_METER_BIN_GLOBAL_REPLICATION-1))*_IRR_GLSL_EXT_LUMA_METER_BIN_COUNT;
		atomicAdd(firstPassOutput.packedHistogram[globalIndex],writeOutVal);
    }

    struct irr_glsl_ext_LumaMeter_PassInfo_t
    {
        uint lowerPercentile;
        uint upperPercentile;
    };
// TODO: Binary Search
    float irr_glsl_ext_LumaMeter_getMeasuredLumaLog2(in irr_glsl_ext_LumaMeter_output_t firstPassOutput, in irr_glsl_ext_LumaMeter_PassInfo_t info)
    {
        float foundPercentiles[2];
// TODO
        return (foundPercentiles[0]+foundPercentiles[1])*0.5;
    }
    #undef _IRR_GLSL_EXT_LUMA_METER_BIN_GLOBAL_REPLICATION
#elif _IRR_GLSL_EXT_LUMA_METER_MODE_DEFINED_==_IRR_GLSL_EXT_LUMA_METER_MODE_GEOM_MEAN
    #define _IRR_GLSL_EXT_LUMA_METER_USING_MEAN
    struct irr_glsl_ext_LumaMeter_output_t
    {
        uint unormAverage;
    };

    #define _IRR_GLSL_EXT_LUMA_METER_GEOM_MEAN_MAX_WG_INCREMENT (1u<22u)
    void irr_glsl_ext_LumaMeter_setFirstPassOutput(out irr_glsl_ext_LumaMeter_output_t firstPassOutput, in float writeOutVal)
    {
    	const float MinLuma = intBitsToFloat(_IRR_GLSL_EXT_LUMA_METER_MIN_LUMA_DEFINED_);
    	const float MaxLuma = intBitsToFloat(_IRR_GLSL_EXT_LUMA_METER_MAX_LUMA_DEFINED_);
		if (gl_LocalInvocationIndex==0u)
		{
			float normalizedAvg = writeOutVal/(float(_IRR_GLSL_EXT_LUMA_METER_INVOCATION_COUNT)*log2(MaxLuma/MinLuma));
			uint incrementor = uint(float(_IRR_GLSL_EXT_LUMA_METER_GEOM_MEAN_MAX_WG_INCREMENT)*normalizedAvg+0.5);
			atomicAdd(firstPassOutput.unormAverage,incrementor);
		}
    }

    struct irr_glsl_ext_LumaMeter_PassInfo_t
    {
        float rcpFirstPassWGCount;
    };
    float irr_glsl_ext_LumaMeter_getMeasuredLumaLog2(in irr_glsl_ext_LumaMeter_output_t firstPassOutput, in irr_glsl_ext_LumaMeter_PassInfo_t info)
    {
    	const float MinLuma = intBitsToFloat(_IRR_GLSL_EXT_LUMA_METER_MIN_LUMA_DEFINED_);
    	const float MaxLuma = intBitsToFloat(_IRR_GLSL_EXT_LUMA_METER_MAX_LUMA_DEFINED_);
        float normalizedAvg = float(firstPassOutput.unormAverage)*log2(MaxLuma/MinLuma)/float(_IRR_GLSL_EXT_LUMA_METER_GEOM_MEAN_MAX_WG_INCREMENT);
        return normalizedAvg*info.rcpFirstPassWGCount+log2(MinLuma);
    }
    #undef _IRR_GLSL_EXT_LUMA_METER_GEOM_MEAN_MAX_WG_INCREMENT
#else
#error "Unsupported Metering Mode!"
#endif


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