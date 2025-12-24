// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/ext/MitsubaLoader/ParserUtil.h"
#include "nbl/ext/MitsubaLoader/CElementBSDF.h"

#include "nbl/ext/MitsubaLoader/ElementMacros.h"

#include "nbl/builtin/hlsl/complex.hlsl"

#include "nbl/type_traits.h" // legacy stuff for `is_any_of`
#include <functional>


namespace nbl::ext::MitsubaLoader
{
namespace impl
{
template<typename T>
struct has_alpha
{
	constexpr static bool value = std::is_base_of_v<CElementBSDF::AllDiffuse,T> || std::is_base_of_v<CElementBSDF::RoughSpecularBase,T>;
};
template<typename T>
struct has_diffuseReflectance
{
	constexpr static bool value = std::is_base_of_v<CElementBSDF::AllDiffuse,T> || std::is_base_of_v<CElementBSDF::AllPlastic,T> ||
		std::is_same_v<CElementBSDF::Phong,T> || std::is_same_v<CElementBSDF::Ward,T>;
};
template<typename T>
struct can_have_isotropicNDF
{
	constexpr static bool value = std::is_base_of_v<CElementBSDF::RoughSpecularBase,T> || std::is_same_v<CElementBSDF::Ward,T>;
};
template<typename T>
struct has_specularReflectance
{
	constexpr static bool value = std::is_base_of_v<CElementBSDF::RoughSpecularBase,T> || std::is_same_v<CElementBSDF::Phong,T> ||
		std::is_same_v<CElementBSDF::Ward,T>;
};
}

auto CElementBSDF::compAddPropertyMap() -> AddPropertyMap<CElementBSDF>
{
	using this_t = CElementBSDF;
	AddPropertyMap<CElementBSDF> retval;

// spectrum setting
#define ADD_VARIANT_SPECTRUM_PROPERTY_CONSTRAINED(MEMBER,CONSTRAINT,...) { \
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_VARIANT_PROPERTY_CONSTRAINED(MEMBER,FLOAT,CONSTRAINT __VA_OPT__(,) __VA_ARGS__) \
		state. ## MEMBER = std::move(_property); \
		success = true; \
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_VARIANT_PROPERTY_CONSTRAINED_END; \
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_VARIANT_PROPERTY_CONSTRAINED(MEMBER,RGB,CONSTRAINT __VA_OPT__(,) __VA_ARGS__) \
		state. ## MEMBER = std::move(_property); \
		success = true; \
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_VARIANT_PROPERTY_CONSTRAINED_END; \
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_VARIANT_PROPERTY_CONSTRAINED(MEMBER,SRGB,CONSTRAINT __VA_OPT__(,) __VA_ARGS__) \
		state. ## MEMBER = std::move(_property); \
		success = true; \
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_VARIANT_PROPERTY_CONSTRAINED_END; \
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_VARIANT_PROPERTY_CONSTRAINED(MEMBER,SPECTRUM,CONSTRAINT __VA_OPT__(,) __VA_ARGS__) \
		state. ## MEMBER = std::move(_property); \
		success = true; \
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_VARIANT_PROPERTY_CONSTRAINED_END; \
}

	// diff trans
	ADD_VARIANT_SPECTRUM_PROPERTY_CONSTRAINED(transmittance,std::is_same,DiffuseTransmitter);

	// diffuse
	ADD_VARIANT_SPECTRUM_PROPERTY_CONSTRAINED(reflectance,derived_from,AllDiffuse);
	ADD_VARIANT_SPECTRUM_PROPERTY_CONSTRAINED(diffuseReflectance,impl::has_diffuseReflectance);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(alpha,FLOAT,impl::has_alpha);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(useFastApprox,BOOLEAN,derived_from,AllDiffuse);

	// specular base
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_VARIANT_PROPERTY_CONSTRAINED(distribution,STRING,derived_from,RoughSpecularBase)
		using ndf_e = RoughSpecularBase::NormalDistributionFunction;
		static const core::unordered_map<std::string,ndf_e,core::CaseInsensitiveHash,core::CaseInsensitiveEquals> StringToType =
		{
			{"beckmann",ndf_e::BECKMANN},
			{"ggx",		ndf_e::GGX},
			{"phong",	ndf_e::PHONG},
			{"as",		ndf_e::ASHIKHMIN_SHIRLEY}
		};
		auto found = StringToType.find(_property.getProperty<SPropertyElementData::Type::STRING>());
		if (found==StringToType.end())
			return;
		state.distribution = found->second;
		success = true;
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_VARIANT_PROPERTY_CONSTRAINED_END;
	// COMMON: alpha
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(alphaU,FLOAT,impl::can_have_isotropicNDF);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(alphaV,FLOAT,impl::can_have_isotropicNDF);
	ADD_VARIANT_SPECTRUM_PROPERTY_CONSTRAINED(specularReflectance,impl::has_specularReflectance);

	// conductor
	ADD_VARIANT_SPECTRUM_PROPERTY_CONSTRAINED(eta,derived_from,AllConductor);
	ADD_VARIANT_SPECTRUM_PROPERTY_CONSTRAINED(k,derived_from,AllConductor);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(extEta,FLOAT,derived_from,AllConductor);
	// adding twice cause two property types are allowed
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_VARIANT_PROPERTY_CONSTRAINED(extEta,STRING,derived_from,AllConductor)
		state.extEta = TransmissiveBase::findIOR(_property.getProperty<SPropertyElementData::Type::STRING>(),logger);
		success = true;
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_VARIANT_PROPERTY_CONSTRAINED_END;
	// special
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_VARIANT_PROPERTY_CONSTRAINED(material,STRING,derived_from,AllConductor)
		state = AllConductor(_property.getProperty<SPropertyElementData::Type::STRING>(),logger);
		success = true;
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_VARIANT_PROPERTY_CONSTRAINED_END;

	// transmissive base
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(intIOR,FLOAT,derived_from,TransmissiveBase);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(extIOR,FLOAT,derived_from,TransmissiveBase);
	// adding twice cause two property types are allowed
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_VARIANT_PROPERTY_CONSTRAINED(intIOR,STRING,derived_from,TransmissiveBase)
		state.intIOR = TransmissiveBase::findIOR(_property.getProperty<SPropertyElementData::Type::STRING>(),logger);
		success = true;
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_VARIANT_PROPERTY_CONSTRAINED_END;
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_VARIANT_PROPERTY_CONSTRAINED(extIOR,STRING,derived_from,TransmissiveBase)
		state.extIOR = TransmissiveBase::findIOR(_property.getProperty<SPropertyElementData::Type::STRING>(),logger);
		success = true;
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_VARIANT_PROPERTY_CONSTRAINED_END;
	ADD_VARIANT_SPECTRUM_PROPERTY_CONSTRAINED(specularTransmittance,derived_from,TransmissiveBase);

	// plastic
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(nonlinear,BOOLEAN,derived_from,AllPlastic);
	// COMMON: diffuseReflectance

	// coating
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(thickness,FLOAT,derived_from,AllCoating);
	ADD_VARIANT_SPECTRUM_PROPERTY_CONSTRAINED(sigmaA,derived_from,AllCoating);

	// bumpmap
	// normalmap

	// mixture
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_VARIANT_PROPERTY_CONSTRAINED(weights,STRING,std::is_same,MixtureBSDF)
		std::istringstream sstr(_property.svalue);
		std::string token;
		while (std::getline(sstr,token,','))
		{
			if (state.weightCount)
			{
				logger.log("<bsdf> MaxChildCount of %d exceeded!",system::ILogger::ELL_ERROR,MetaBSDF::MaxChildCount);
				break;
			}
			state.weights[state.weightCount++] = std::stof(token);
		}
		success = true;
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_VARIANT_PROPERTY_CONSTRAINED_END;

	// blend
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(weight,FLOAT,std::is_same,BlendBSDF);

	// mask
	ADD_VARIANT_SPECTRUM_PROPERTY_CONSTRAINED(opacity,std::is_same,Mask);

	// twosided

	// phong
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(exponent,FLOAT,std::is_same,Phong);
	// COMMON: specularReflectance
	// COMMON: diffuseReflectance

	// ward
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_VARIANT_PROPERTY_CONSTRAINED(variant,STRING,std::is_same,Ward)
		static const core::unordered_map<std::string,Ward::Type,core::CaseInsensitiveHash,core::CaseInsensitiveEquals> StringToType =
		{
			{"ward",		Ward::Type::WARD},
			{"ward-duer",	Ward::Type::WARD_DUER},
			{"balanced",	Ward::Type::BALANCED}
		};
		auto found = StringToType.find(_property.getProperty<SPropertyElementData::Type::STRING>());
		if (found==StringToType.end())
			return;
		state.variant = found->second;
		success = true;
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_VARIANT_PROPERTY_CONSTRAINED_END;
	// COMMON: alphaU
	// COMMON: alphaV
	// COMMON: specularReflectance
	// COMMON: diffuseReflectance

	// TODO: set HK and IRAWAN parameters, sigmaS, sigmaT, albedo, filename, repeatU, repeatV

#undef ADD_VARIANT_SPECTRUM_PROPERTY_CONSTRAINED

	return retval;
}

float CElementBSDF::TransmissiveBase::findIOR(const std::string_view name, system::logger_opt_ptr logger)
{
	static const core::unordered_map<std::string_view,float,core::CaseInsensitiveHash,core::CaseInsensitiveEquals> NamedIndicesOfRefraction =
	{
		{"vacuum",					1.f},
		{"helium",					1.00004f},
		{"hydrogen",				1.00013f},
		{"air",						1.00028f},
		{"carbon dioxide",			1.00045f},
		{"water ice",				1.31f},
		{"water",					1.333f},
		{"acetone",					1.36f},
		{"ethanol",					1.361f},
		{"fused quartz",			1.458f},
		{"carbon tetrachloride",	1.461f},
		{"pyrex",					1.47f},
		{"glycerol",				1.4729f},
		{"acrylic glass",			1.49f},
		{"polypropylene",			1.49f},
		{"benzene",					1.501f},
		{"bk7",						1.5046f},
		{"silicone oil",			1.52045f},
		{"sodium chloride",			1.544f},
		{"amber",					1.55f},
		{"pet",						1.575f},
		{"bromine",					1.661f},
		{"diamond",					2.419f}
	};
	auto found = NamedIndicesOfRefraction.find(name);
	if (found == NamedIndicesOfRefraction.end())
		return NAN;
	return found->second;
}


CElementBSDF::AllConductor::AllConductor(const std::string_view material, system::logger_opt_ptr logger) : RoughSpecularBase(0.1)
{
// TODO fill this out with values from http://www.luxpop.com/HU_v173.cgi?OpCode=73 and https://github.com/mmp/pbrt-v3/blob/master/src/materials/metal.cpp or https://refractiveindex.info/?shelf=main&book=Cu&page=Johnson
	// we use Rec 709 for the Color primaries of this table, so Red ~= 615nm, Green ~= 535nm, Blue ~= 465nm
	static const core::unordered_map<std::string_view,hlsl::vector<hlsl::complex_t<float32_t>,3>,core::CaseInsensitiveHash,core::CaseInsensitiveEquals> NamedConductors =
	{
#define SPECTRUM_MACRO(R,G,B,X,Y,Z) {{R,X},{G,Y},{B,Z}}
		{"a-C",				SPECTRUM_MACRO(1.6855f, 1.065f, 1.727f,		0.0f, 0.009f, 0.0263f)},			// there is no "a-C", but "a-C:H; data from palik"
		{"Ag",				SPECTRUM_MACRO(0.059481f, 0.055090f, 0.046878f,		4.1367f, 3.4574f, 2.8028f)},
		{"Al",				SPECTRUM_MACRO(1.3404f, 0.95151f, 0.68603f,		7.3509f, 6.4542f, 5.6351f)},
		{"AlAs",			SPECTRUM_MACRO(3.1451f, 3.2636f, 3.4543f,		0.0012319f, 0.0039041f, 0.012940f)},
		{"AlAs_palik",		SPECTRUM_MACRO(3.145f, 3.273f, 3.570f,		0.0f, 0.000275f, 1.56f)},
		{"Au",				SPECTRUM_MACRO(0.21415f, 0.52329f, 1.3319f,		3.2508f, 2.2714f, 1.8693f)},
		{"Be",				SPECTRUM_MACRO(3.3884f, 3.2860f, 3.1238f,		3.1692f, 3.1301f, 3.1246f)},
		{"Be_palik",		SPECTRUM_MACRO(3.46f, 3.30f, 3.19f,		3.18f, 3.18f, 3.16f)},
		{"Cr",				SPECTRUM_MACRO(3.2246f, 2.6791f, 2.1411f,		4.2684f, 4.1664f, 3.9300f)},
		{"CsI",				SPECTRUM_MACRO(1.7834f, 1.7978f, 1.8182f,		0.0f, 0.0f, 0.0f)},
		{"CsI_palik",		SPECTRUM_MACRO(1.78006f, 1.79750f, 1.82315,		0.0f, 0.0f, 0.0f)},
		{"Cu",				SPECTRUM_MACRO(0.32075f,1.09860f,1.2469f,		3.17900f,2.59220f,2.4562)},
		{"Cu_palik",		SPECTRUM_MACRO(0.32000f, 1.04f, 1.16f,		3.15000f, 2.59f, 2.4f)},
		{"Cu20",			SPECTRUM_MACRO(2.975f, 3.17f, 3.075f,		0.122f, 0.23f, 0.525f)},  // out of range beyond 2.5 um refractiveindex.info and similar websites, so data applied is same as from palik's data
		{"Cu20_palik",	SPECTRUM_MACRO(2.975f, 3.17f, 3.075f,		0.122f, 0.23f, 0.525f)},
		{"d-C",			SPECTRUM_MACRO(2.4123f, 2.4246f, 2.4349f,		0.0f, 0.0f, 0.0f)},
		{"d-C_palik",		SPECTRUM_MACRO(2.4137f, 2.4272f, 2.4446f,		0.0f, 0.0f, 0.0f)},
		{"Hg",			SPECTRUM_MACRO(1.8847f, 1.4764f, 1.1306f,		5.1147f, 4.5410f, 3.9896f)},
		{"Hg_palik",		SPECTRUM_MACRO(1.850f, 1.460f, 1.100f,		5.100f, 4.600f, 3.990f)},
		//{"HgTe",			SPECTRUM_MACRO(,,,		,,)},						// lack of length wave range for our purpose https://www.researchgate.net/publication/3714159_Dispersion_of_refractive_index_in_degenerate_mercury_cadmium_telluride
		//{"HgTe_palik",		SPECTRUM_MACRO(,,,		,,)},					// the same in palik (wavelength beyond 2 um)			
		{"Ir",			SPECTRUM_MACRO(2.4200f, 2.0795f, 1.7965f,		5.0665f, 4.6125f, 4.1120f)},
		{"Ir_palik",		SPECTRUM_MACRO(2.44f, 2.17f, 1.87f,		4.52f, 4.24f, 3.79f)},
		{"K",			SPECTRUM_MACRO(0.052350f, 0.048270f, 0.042580f,		1.6732f, 1.3919f, 1.1195f)},
		{"K_palik",		SPECTRUM_MACRO(0.0525f, 0.0483f, 0.0427f,		1.67f, 1.39f, 1.12f)},
		{"Li",			SPECTRUM_MACRO(0.14872f, 0.14726f, 0.19236f,		2.9594f, 2.5129f, 2.1144f)},
		{"Li_palik",		SPECTRUM_MACRO(0.218f, 0.2093f, 0.229f,		2.848f, 2.369f, 2.226f)},
		{"MgO",			SPECTRUM_MACRO(1.7357f, 1.7419f, 1.7501f,		0.0f, 0.0f, 0.0f)},
		{"MgO_palik",		SPECTRUM_MACRO(1.7355f, 1.7414f, 1.74975f,		0.0f, 0.0f, 1.55f)},				 // Handbook of optical constants of solids vol 2 page 951, weird k compoment alone, no measurements and resoults
		{"Mo",			SPECTRUM_MACRO(0.76709f, 0.57441f, 0.46711f,		8.5005f, 7.2352f, 6.1383f)},	   	// https://refractiveindex.info/?shelf=main&book=Mo&page=Werner comparing with palik - weird
		{"Mo_palik",		SPECTRUM_MACRO(3.68f, 3.77f, 3.175f,		3.51f, 3.624f, 3.56f)},
		{"Na_palik",			SPECTRUM_MACRO(0.0522f, 0.061f, 0.0667f,		2.535f, 2.196f, 1.861f)},
		{"Nb",			SPECTRUM_MACRO(2.2775f, 2.2225f, 2.0050f,		3.2500f, 3.1325f, 3.0100f)},
		{"Nb_palik",			SPECTRUM_MACRO(2.869f, 2.9235f, 2.738f,		2.867f, 2.8764f, 2.8983f)},
		{"Ni_palik",			SPECTRUM_MACRO(1.921f, 1.744f, 1.651f,		3.615f, 3.168f, 2.753f)},
		{"Rh",			SPECTRUM_MACRO(2.8490f, 2.6410f, 2.4310f,		3.5450f, 3.3150f, 3.1190f)},
		{"Rh_palik",		SPECTRUM_MACRO(2.092f, 1.934f, 1.8256f,		5.472f, 4.902f, 4.5181f)},
		{"Se",			SPECTRUM_MACRO(1.4420f, 1.4759f, 1.4501f,		0.018713f, 0.10233f, 0.18418f)},
		{"Se_palik",		SPECTRUM_MACRO(3.346f, 3.013f, 3.068f,		0.6402f, 0.6711f, 0.553f)},
		{"SiC",			SPECTRUM_MACRO(2.6398f, 2.6677f, 2.7069f,		0.0f, 0.0f, 0.0f)},
		{"SiC_palik",		SPECTRUM_MACRO(2.6412f, 2.6684f, 2.7077f,		0.0f, 0.0f, 0.0f)},
		{"SnTe",			SPECTRUM_MACRO(3.059f, 1.813f, 1.687f,		5.144f, 4.177f, 3.555f)},			   // no data except palik's resources, so data same as palik
		{"SnTe_palik",		SPECTRUM_MACRO(3.059f, 1.813f, 1.687f,		5.144f, 4.177f, 3.555f)},
		{"Ta",			SPECTRUM_MACRO(1.0683f, 1.1379f, 1.2243f,		5.5047f, 4.7432f, 4.0988f)},
		{"Ta_palik",		SPECTRUM_MACRO(1.839f, 2.5875f, 2.8211f,		1.997f, 1.8683f, 2.0514f)},
		{"Te",			SPECTRUM_MACRO(4.1277f, 3.2968f, 2.6239f,		2.5658f, 2.8789f, 2.7673f)},
		{"Te_palik",		SPECTRUM_MACRO(5.8101f, 4.5213f, 3.3682f,		2.9428f, 3.7289f, 3.6783f)},
		{"ThF4",			SPECTRUM_MACRO(1.5113f, 1.5152f, 1.5205f,		0.0f, 0.0f, 0.0f)},
		{"ThF4_palik",		SPECTRUM_MACRO(1.520f, 1.5125f, 1.524f,		0.0f, 0.0f, 0.0f)},
		{"TiC",			SPECTRUM_MACRO(3.0460f, 2.9815f, 2.8864f,		2.6585f, 2.4714f, 2.3987f)},
		{"TiC_palik",		SPECTRUM_MACRO(3.0454f, 2.9763, 2.8674f,		2.6589f, 2.4695f, 2.3959f)},
		{"TiO2",			SPECTRUM_MACRO(2.1362f, 2.1729f, 2.2298f,		0.0f, 0.0f, 0.0f)},
		{"TiO2_palik",		SPECTRUM_MACRO(2.5925f, 2.676f, 2.78f,		0.0f, 0.0f, 0.0f)},
		{"VC",			SPECTRUM_MACRO(3.0033f, 2.8936f, 2.8138f,		2.4981f, 2.3046f, 2.1913f)},
		{"VC_palik",		SPECTRUM_MACRO(3.0038f, 2.8951f, 2.8184f,		2.4923f, 2.3107f, 2.1902f)},
		{"V_palik",		SPECTRUM_MACRO(3.512f, 3.671f, 3.2178f,		2.9337, 3.069f, 3.3667f)},
		{"VN",			SPECTRUM_MACRO(2.3429f, 2.2268f, 2.1550f,		2.4506f, 2.1345f, 1.8753f)},
		{"VN_palik",		SPECTRUM_MACRO(2.3418f, 2.2239f, 2.1539f,		2.4498f, 2.1371f, 1.8776f)},		
		{"W",		SPECTRUM_MACRO(0.96133f, 1.5474f, 2.1930f,		6.2902f, 5.1052f, 5.0325f)},
		{"none",			SPECTRUM_MACRO(0.f,0.f,0.f,		0.f,0.f,0.f)}
#undef SPECTRUM_MACRO
	};

	auto found = NamedConductors.find(material);
	if (found==NamedConductors.end())
	{
		_NBL_DEBUG_BREAK_IF(true);
		logger.log("Named material %s in <conductor> failed to be found, defaulting to \"none\"",system::ILogger::ELL_ERROR,material.data());
		found = NamedConductors.find("none");
		assert(found != NamedConductors.end());
	}

	const auto etaK = found->second;
	eta = SPropertyElementData(SPropertyElementData::Type::RGB,float32_t4{etaK.r.real(),etaK.g.real(),etaK.b.real(),0.f});
	k = SPropertyElementData(SPropertyElementData::Type::RGB,float32_t4{etaK.r.real(),etaK.g.real(),etaK.b.real(),0.f});
	extEta = TransmissiveBase::findIOR("air",logger);
}


bool CElementBSDF::processChildData(IElement* _child, const std::string& name, system::logger_opt_ptr logger)
{
	if (!_child)
		return true;

	using this_t = CElementBSDF;

#define SET_TEXTURE_CONSTRAINED(MEMBER,CONSTRAINT,... )	{#MEMBER,{.func = [](this_t* _this, IElement* _child, const system::logger_opt_ptr logger)->bool \
	{ \
		bool success = false; \
		auto _texture = static_cast<CElementTexture*>(_child); \
		_this->visit([&_texture,logger,&success](auto& state)->void \
			{ \
					if constexpr (CONSTRAINT<std::remove_reference_t<decltype(state)> __VA_OPT__(,) __VA_ARGS__>::value) \
					{ 
#define SET_TEXTURE_CONSTRAINED_END	} \
			} \
		); \
		return success; \
	} \
}}
#define SET_TEXTURE_CONSTRAINED_SIMPLE(MEMBER,CONSTRAINT,... )	SET_TEXTURE_CONSTRAINED(MEMBER,CONSTRAINT __VA_OPT__(,) __VA_ARGS__) \
	state. ## MEMBER = _texture; \
	success = true; \
SET_TEXTURE_CONSTRAINED_END

	// TODO: store this somewhere outside a global
	static const ProcessChildCallbackMap<CElementBSDF> TextureCallbacks =
	{
		SET_TEXTURE_CONSTRAINED_SIMPLE(transmittance,std::is_same,DiffuseTransmitter),
		SET_TEXTURE_CONSTRAINED_SIMPLE(reflectance,derived_from,AllDiffuse),
		SET_TEXTURE_CONSTRAINED_SIMPLE(diffuseReflectance,impl::has_diffuseReflectance),
		SET_TEXTURE_CONSTRAINED_SIMPLE(alpha,impl::has_alpha),
		SET_TEXTURE_CONSTRAINED_SIMPLE(alphaU,impl::can_have_isotropicNDF),
		SET_TEXTURE_CONSTRAINED_SIMPLE(alphaV,impl::can_have_isotropicNDF),
		SET_TEXTURE_CONSTRAINED_SIMPLE(specularReflectance,impl::has_specularReflectance),
		SET_TEXTURE_CONSTRAINED_SIMPLE(specularTransmittance,derived_from,TransmissiveBase),
		SET_TEXTURE_CONSTRAINED_SIMPLE(sigmaA,derived_from,AllCoating),
		SET_TEXTURE_CONSTRAINED(,is_any_of,BumpMap,NormalMap)
			state.texture = _texture;
			success = true;
		SET_TEXTURE_CONSTRAINED_END,
		SET_TEXTURE_CONSTRAINED_SIMPLE(weight,std::is_same,BlendBSDF),
		SET_TEXTURE_CONSTRAINED_SIMPLE(opacity,std::is_same,Mask),
		SET_TEXTURE_CONSTRAINED_SIMPLE(exponent,std::is_same,Phong)
	};
#undef SET_TEXTURE_CONSTRAINED
#undef SET_TEXTURE_CONSTRAINED_SIMPLE

	switch (_child->getType())
	{
		case IElement::Type::TEXTURE:
		{
			auto found = TextureCallbacks.find(name);
			if (found==TextureCallbacks.end())
				found = TextureCallbacks.find("");
			if (found==TextureCallbacks.end())
			{
				logger.log("No <bsdf> can have <texture> nested inside it with name \"%s\"!",system::ILogger::ELL_ERROR,name.c_str());
				return false;
			}
			if (!found->second(this,_child,logger))
			{
				logger.log(
					"Failed to parse <texture> with name \"%s\" nested inside <bsdf> of type %d!",
					system::ILogger::ELL_ERROR,name.c_str(),type
				);
				return false;
			}
			return true;
		}
		case IElement::Type::BSDF:
		{
			size_t maxChildCount = 0;
			{
				const auto* _this = this;
				visit([&maxChildCount,_this](const auto& state)->void
					{
						using state_t = std::remove_reference_t<decltype(state)>;
						if constexpr (std::is_base_of_v<MetaBSDF,state_t>)
							maxChildCount = state_t::MaxChildCount;
					}
				);
			}
			if (meta_common.childCount<maxChildCount)
			{
				auto _bsdf = static_cast<CElementBSDF*>(_child);
				meta_common.bsdf[meta_common.childCount++] = _bsdf;
				return true;
			}
			logger.log("<bsdf type=\"%d\"> cannot have more than %d other <bsdf>s nested inside it!",system::ILogger::ELL_ERROR,type,maxChildCount);
			return false;
		}
		default:
			logger.log("Unsupported <%s> nested inside <bsdf> only <texture> and <bsdf> are allowed!",system::ILogger::ELL_ERROR,_child->getLogName());
			return false;
	}
	return true;
}

bool CElementBSDF::onEndTag(CMitsubaMetadata* globalMetadata, system::logger_opt_ptr logger)
{
	NBL_EXT_MITSUBA_LOADER_ELEMENT_INVALID_TYPE_CHECK(true);
	
	// TODO: Validation
	{
	}

	return true;
}

}