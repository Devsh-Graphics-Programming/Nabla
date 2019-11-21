#ifndef _IRR_EXT_AUTO_EXPOSURE_C_TONE_MAPPER_INCLUDED_
#define _IRR_EXT_AUTO_EXPOSURE_C_TONE_MAPPER_INCLUDED_

#include "irrlicht.h"

namespace irr
{
namespace ext
{
namespace AutoExposure
{

// TODO: move this to common header along with tonemapping technique enums
struct ReinhardParams
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
};
struct ACESParams
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
};

class CToneMapper : public core::IReferenceCounted, public core::InterfaceUnmovable
{
    public:
        static core::smart_refctd_ptr<CToneMapper> create(video::IVideoDriver* _driver, asset::E_FORMAT inputFormat, const asset::IGLSLCompiler* compiler);

		auto* getDescriptorSetLayout() {return dsLayout.get();}

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
vec3 irr_ext_Autoexposure_linear2SRGB(in vec3 color)
{
	return mix(color*12.92,1.055*pow(color,vec3(1.0/2.4))-0.055,greaterThan(color,vec3(0.0031308)));
}

float irr_ext_Autoexposure_luminance_R709(in vec3 color)
{
	return dot(vec3(0.2126,0.7152,0.0722),color);
}

struct irr_ext_Autoexposure_ReinhardParams
{
	float keyAndLinearExposure; // 0.18*exp2(exposure)
	float rcpWhite2; // 1.0/(maxWhite*maxWhite)
};

vec3 irr_ext_Autoexposure_ToneMapReinhard(in irr_ext_Autoexposure_ReinhardParams params, in vec3 color)
{
	float luma = irr_ext_Autoexposure_luminance_R709(color);
	return color*(1.0+luma*params.rcpWhite2)/(1.0+luma);
}

// https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
struct irr_ext_Autoexposure_ACESParams
{
	float exposure; // actualExposure-midGrayLog2*(gamma-1)
	float preGammaMinus1; // 0.0
};

//! This function has to be wrong, sRGB is apparently included but its a non-linear function!
// its operating on each ACES color space channel separately, then doing a linear inversion to gamma sRGB !?
vec3 irr_ext_Autoexposure_ACES(in irr_ext_Autoexposure_ACESParams params, in vec3 color)
{
	float luma = irr_ext_Autoexposure_luminance_R709(color);
	float logLuma = log2(luma);
	vec3 exposed = color*exp2(logLuma*params.preGammaMinus1+params.exposure);

	mat3 ACES_R709_Input = mat3(
		0.59719, 0.35458, 0.04823,
		0.07600, 0.90834, 0.01566,
		0.02840, 0.13383, 0.83777
	);
	vec3 v = ACES_R709_Input*exposed;

	vec3 a = v*(v+vec3(0.0245786))-vec3(0.000090537);
    vec3 b = v*(0.983729*v+vec3(0.4329510))+vec3(0.238081);

	mat3 ACES_R709_Output = mat3(
		 1.60475, -0.53108, -0.07367,
		-0.10208,  1.10813, -0.00605,
		-0.00327, -0.07276,  1.07602
	);
	return ACES_R709_Output*(a/b);
}
			)===";
		}

        video::IVideoDriver* m_driver;
		asset::E_FORMAT format;
		asset::E_FORMAT viewFormat;
		core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> dsLayout;

		core::smart_refctd_ptr<video::IGPUPipelineLayout> pipelineLayout;
		core::smart_refctd_ptr<video::IGPUComputePipeline> computePipeline;
};

}
}
}

#endif
