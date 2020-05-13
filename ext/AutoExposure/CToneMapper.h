#ifndef _IRR_EXT_TONE_MAPPER_C_TONE_MAPPER_INCLUDED_
#define _IRR_EXT_TONE_MAPPER_C_TONE_MAPPER_INCLUDED_

#include "irrlicht.h"

namespace irr
{
namespace ext
{
namespace ToneMapper
{

// TODO: move this to common header along with tonemapping technique enums
struct alignas(16) ReinhardParams
{
	float keyAndLinearExposure; // 0.18*exp2(exposure)
	float rcpWhite2; // reciprocal(MaxLuminance*keyAndLinearExposure*burn*burn)^2

	static inline ReinhardParams fromKeyAndBurn(float key, float burn, float AvgLuma, float MaxLuma)
	{
		ReinhardParams retval;
		retval.keyAndLinearExposure = key/AvgLuma;
		retval.rcpWhite2 = 1.f/(MaxLuma*retval.keyAndLinearExposure*burn*burn);
		retval.rcpWhite2 *= retval.rcpWhite2;
		return retval;
	}
private:
	uint32_t uselessPadding[2];
};
struct alignas(16) ACESParams
{
	float exposure;
	float preGammaMinus1;

	//! Contrast being a EV multiplier
	static inline ACESParams fromMidAndContrast(float AvgLuma, float Contrast=1.f)
	{
		ACESParams retval;
		assert(false); retval.exposure = NAN; // no idea how to invert the tonemapper yet, `arg(y) s.t. AvgLuma=ACES^-1(0.18)`
		retval.preGammaMinus1 = Contrast-1.f;
		retval.exposure -= log2f(AvgLuma)*retval.preGammaMinus1;
	}
private:
	uint32_t uselessPadding[2];
};

class CToneMapper : public core::IReferenceCounted, public core::InterfaceUnmovable
{
    public:
        static core::smart_refctd_ptr<CToneMapper> create(video::IVideoDriver* _driver, asset::E_FORMAT inputFormat, const asset::IGLSLCompiler* compiler);

		bool tonemap(video::IGPUImageView* inputThatsInTheSet, video::IGPUDescriptorSet* set, uint32_t parameterUBOOffset);

    private:
        CToneMapper(video::IVideoDriver* _driver, asset::E_FORMAT inputFormat,
					core::smart_refctd_ptr<video::IGPUDescriptorSetLayout>&& _dsLayout,
					core::smart_refctd_ptr<video::IGPUPipelineLayout>&& _pipelineLayout,
					core::smart_refctd_ptr<video::IGPUComputePipeline>&& _computePipeline);
        ~CToneMapper() = default;
		
		static inline const char* getInclude()
		{
			return R"===(
#include "irr/builtin/glsl/colorspace/EOTF.glsl"
#include "irr/builtin/glsl/colorspace/encodeCIEXYZ.glsl"
#include "irr/builtin/glsl/colorspace/decodeCIEXYZ.glsl"
#include "irr/builtin/glsl/colorspace/OETF.glsl"


struct irr_glsl_ext_ToneMapper_ReinhardParams_t
{
	float keyAndLinearExposure; // 0.18*exp2(manualExposure)
	float rcpWhite2; // 1.0/(maxWhite*maxWhite)
};

struct irr_glsl_ext_ToneMapper_ACESParams_t
{
	float exposure; // actualExposure-midGrayLog2*(gamma-1)
	float preGammaMinus1; // 0.0
};


// TODO #if
	#define irr_glsl_ext_ToneMapper_Params_t irr_glsl_ext_ToneMapper_ReinhardParams_t
	float irr_glsl_ext_ToneMapper_operator(in irr_glsl_ext_ToneMapper_Params_t params, inout float autoexposedLuma)
	{
		autoexposedLuma *= params.keyAndLinearExposure;
		return (1.0+autoexposedLuma*params.rcpWhite2)/(1.0+autoexposedLuma);
	}
//#else
	#define irr_glsl_ext_ToneMapper_Params_t irr_glsl_ext_ToneMapper_ACESParams_t
	vec3 irr_ext_Autoexposure_ACES(in irr_ext_Autoexposure_ACESParams params, inout float autoexposedLuma)
	{
		float luma = irr_ext_Autoexposure_luminance_R709(color);
		float logLuma = log2(luma);
		vec3 exposed = color*exp2(logLuma*params.preGammaMinus1+params.exposure);


		// XYZ => sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT
		const mat3 ACES_R709_Input = irr_glsl_XYZtosRGB*mat3(
			0.59719, 0.35458, 0.04823,
			0.07600, 0.90834, 0.01566,
			0.02840, 0.13383, 0.83777
		);
		vec3 v = ACES_R709_Input*exposed;

		vec3 a = v*(v+vec3(0.0245786))-vec3(0.000090537);
		vec3 b = v*(0.983729*v+vec3(0.4329510))+vec3(0.238081);

		// ODT_SAT => XYZ => D60_2_D65 => sRGB => XYZ
		const mat3 ACES_R709_Output = mat3(
			 1.60475, -0.53108, -0.07367,
			-0.10208,  1.10813, -0.00605,
			-0.00327, -0.07276,  1.07602
		)*irr_glsl_sRGBtoXYZ;
		return clamp(ACES_R709_Output*(a/b), 0.0, 1.0);
	}
//#endif

// ideas for more operators https://web.archive.org/web/20191226154550/http://cs.columbia.edu/CAVE/software/softlib/dorf.php


vec3 irr_glsl_ext_ToneMapper_tonemap(vec3 cieXYZColor, in irr_glsl_ext_ToneMapper_Params_t params, in float extraNegEV)
{
	cieXYZColor.y *= exp2(-extraNegEV);
	cieXYZColor *= irr_glsl_ext_ToneMapper_operator(params,cieXYZColor.y);
	return cieXYZColor;
}

#endif
			)===";
		}

		_IRR_STATIC_INLINE_CONSTEXPR uint32_t DISPATCH_SIZE = 16u;

        video::IVideoDriver* m_driver;
		asset::E_FORMAT format;
		asset::E_FORMAT viewFormat;
};

}
}
}

#endif
