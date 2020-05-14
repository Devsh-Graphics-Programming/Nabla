#ifndef _IRR_EXT_TONE_MAPPER_C_GLSL_TONE_MAPPING_BUILTIN_INCLUDE_LOADER_INCLUDED_
#define _IRR_EXT_TONE_MAPPER_C_GLSL_TONE_MAPPING_BUILTIN_INCLUDE_LOADER_INCLUDED_

#include "irr/asset/IBuiltinIncludeLoader.h"

namespace irr
{
namespace ext
{
namespace ToneMapper
{

class CGLSLToneMappingBuiltinIncludeLoader : public asset::IBuiltinIncludeLoader
{
    public:
        const char* getVirtualDirectoryName() const override { return "glsl/ext/ToneMapper/"; }

    private:
        static std::string getOperators(const std::string&)
        {
            return
R"(#ifndef _IRR_GLSL_EXT_TONE_MAPPER_OPERATORS_INCLUDED_
#define _IRR_GLSL_EXT_TONE_MAPPER_OPERATORS_INCLUDED_


struct irr_glsl_ext_ToneMapper_ReinhardParams_t
{
	float keyAndManualLinearExposure;
	float rcpWhite2;
};

struct irr_glsl_ext_ToneMapper_ACESParams_t
{
	float gamma; // 1.0
	float exposure; // actualExposure+midGrayLog2
};

#define _IRR_GLSL_EXT_TONE_MAPPER_REINHARD_OPERATOR 0
#define _IRR_GLSL_EXT_TONE_MAPPER_ACES_OPERATOR 1

#if _IRR_GLSL_EXT_TONE_MAPPER_OPERATOR_DEFINED_==_IRR_GLSL_EXT_TONE_MAPPER_REINHARD_OPERATOR
	#ifndef irr_glsl_ext_ToneMapper_Params_t
	#define irr_glsl_ext_ToneMapper_Params_t irr_glsl_ext_ToneMapper_ReinhardParams_t
	#endif

	vec3 irr_glsl_ext_ToneMapper_operator(in irr_glsl_ext_ToneMapper_Params_t params, in vec3 rawCIEXYZcolor, in float extraNegEV)
	{
		vec3 tonemapped = rawCIEXYZcolor;
		tonemapped.y *= params.keyAndManualLinearExposure*exp2(-extraNegEV);
		tonemapped.y *= (1.0+tonemapped.y*params.rcpWhite2)/(1.0+tonemapped.y);
		return tonemapped;
	}
#elif _IRR_GLSL_EXT_TONE_MAPPER_OPERATOR_DEFINED_==_IRR_GLSL_EXT_TONE_MAPPER_ACES_OPERATOR
	#ifndef irr_glsl_ext_ToneMapper_Params_t
	#define irr_glsl_ext_ToneMapper_Params_t irr_glsl_ext_ToneMapper_ACESParams_t
	#endif

	vec3 irr_glsl_ext_ToneMapper_operator(in irr_glsl_ext_ToneMapper_Params_t params, inout vec3 rawCIEXYZcolor, in float extraNegEV)
	{
		vec3 tonemapped = rawCIEXYZcolor;
		tonemapped.y = exp2(log2(tonemapped.y+params.exposure-extraNegEV)*params.gamma);

		// XYZ => RRT_SAT
		// this seems to be a matrix for some hybrid colorspace, coefficients are somewhere inbetween BT2020 and ACEScc(t)
		const mat3 XYZ_RRT_Input = mat3(
			vec3( 1.594168310,-0.6332771780, 0.00892840419),
			vec3(-0.262608051, 1.5840380200, 0.03648501260),
			vec3(-0.231993079, 0.0164147373, 0.87711471300)
		);

		// this is obviously fitted to some particular simulated sensor/film and display
		vec3 v = XYZ_RRT_Input*tonemapped;
		vec3 a = v*(v+vec3(0.0245786))-vec3(0.000090537);
		vec3 b = v*(0.983729*v+vec3(0.4329510))+vec3(0.238081);
		v = a/b;

		// ODT_SAT => XYZ
		// this seems to be a matrix for some hybrid colorspace, coefficients are similar to AdobeRGB,BT2020 and ACEScc(t)
		const mat3 ODT_XYZ_Output = mat3(
			vec3( 0.624798000, 0.268048108, 0.0157514643),
			vec3( 0.164064825, 0.674283803, 0.0526682511),
			vec3( 0.161605373, 0.057667464, 1.0204007600)
		);
		return ODT_XYZ_Output*v;
	}
#else
#error "Unsupported Tonemapping Operator"
#endif

// ideas for more operators https://web.archive.org/web/20191226154550/http://cs.columbia.edu/CAVE/software/softlib/dorf.php
// or get proper ACES RRT and ODTs
// https://partnerhelp.netflixstudios.com/hc/en-us/articles/360000622487-I-m-using-ACES-Which-Output-Transform-should-I-use-
#endif
)";
        }

    protected:
        inline core::vector<std::pair<std::regex,HandleFunc_t> > getBuiltinNamesToFunctionMapping() const override
        {
            return {
                { std::regex{"operators\\.glsl"}, &getOperators }
            };
        }
    };

}
}
}

#endif