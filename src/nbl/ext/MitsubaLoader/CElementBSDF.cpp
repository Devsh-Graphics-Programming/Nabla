// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/MitsubaLoader/ParserUtil.h"
#include "nbl/ext/MitsubaLoader/CElementFactory.h"

#include "nbl/ext/MitsubaLoader/CMitsubaMetadata.h"

#include <functional>

namespace nbl
{
namespace ext
{
namespace MitsubaLoader
{


template<>
CElementFactory::return_type CElementFactory::createElement<CElementBSDF>(const char** _atts, ParserManager* _util)
{
	const char* type;
	const char* id;
	std::string name;
	if (!IElement::getTypeIDAndNameStrings(type, id, name, _atts))
		return CElementFactory::return_type(nullptr, "");

	static const core::unordered_map<std::string, CElementBSDF::Type, core::CaseInsensitiveHash, core::CaseInsensitiveEquals> StringToType =
	{
		{"diffuse",			CElementBSDF::Type::DIFFUSE},
		{"roughdiffuse",	CElementBSDF::Type::ROUGHDIFFUSE},
		{"dielectric",		CElementBSDF::Type::DIELECTRIC},
		{"thindielectric",	CElementBSDF::Type::THINDIELECTRIC},
		{"roughdielectric",	CElementBSDF::Type::ROUGHDIELECTRIC},
		{"conductor",		CElementBSDF::Type::CONDUCTOR},
		{"roughconductor",	CElementBSDF::Type::ROUGHCONDUCTOR},
		{"plastic",			CElementBSDF::Type::PLASTIC},
		{"roughplastic",	CElementBSDF::Type::ROUGHPLASTIC},
		{"coating",			CElementBSDF::Type::COATING},
		{"roughcoating",	CElementBSDF::Type::ROUGHCOATING},
		{"bumpmap",			CElementBSDF::Type::BUMPMAP},
		{"phong",			CElementBSDF::Type::PHONG},
		{"ward",			CElementBSDF::Type::WARD},
		{"mixturebsdf",		CElementBSDF::Type::MIXTURE_BSDF},
		{"blendbsdf",		CElementBSDF::Type::BLEND_BSDF},
		{"mask",			CElementBSDF::Type::MASK},
		{"twosided",		CElementBSDF::Type::TWO_SIDED},
		{"difftrans",		CElementBSDF::Type::DIFFUSE_TRANSMITTER}//,
		//{"hk",				CElementBSDF::Type::HANRAHAN_KRUEGER},
		//{"irawan",			CElementBSDF::Type::IRAWAN_MARSCHNER}
	};

	auto found = StringToType.find(type);
	if (found==StringToType.end())
	{
		ParserLog::invalidXMLFileStructure("unknown type");
		_NBL_DEBUG_BREAK_IF(false);
		return CElementFactory::return_type(nullptr, "");
	}

	CElementBSDF* obj = _util->objects.construct<CElementBSDF>(id);
	if (!obj)
		return CElementFactory::return_type(nullptr, "");

	obj->type = found->second;
	// defaults
	switch (obj->type)
	{
		case CElementBSDF::Type::DIFFUSE:
			[[fallthrough]];
		case CElementBSDF::Type::ROUGHDIFFUSE:
			obj->diffuse = CElementBSDF::AllDiffuse();
			break;
		case CElementBSDF::Type::DIELECTRIC:
			[[fallthrough]];
		case CElementBSDF::Type::THINDIELECTRIC:
			[[fallthrough]];
		case CElementBSDF::Type::ROUGHDIELECTRIC:
			obj->dielectric = CElementBSDF::AllDielectric();
			break;
		case CElementBSDF::Type::CONDUCTOR:
			[[fallthrough]];
		case CElementBSDF::Type::ROUGHCONDUCTOR:
			obj->conductor = CElementBSDF::AllConductor();
			break;
		case CElementBSDF::Type::PLASTIC:
			[[fallthrough]];
		case CElementBSDF::Type::ROUGHPLASTIC:
			obj->plastic = CElementBSDF::AllPlastic();
			break;
		case CElementBSDF::Type::COATING:
			[[fallthrough]];
		case CElementBSDF::Type::ROUGHCOATING:
			obj->coating = CElementBSDF::AllCoating();
			break;
		case CElementBSDF::Type::BUMPMAP:
			obj->bumpmap = CElementBSDF::BumpMap();
			break;
		case CElementBSDF::Type::PHONG:
			obj->phong = CElementBSDF::Phong();
			break;
		case CElementBSDF::Type::WARD:
			obj->ward = CElementBSDF::Ward();
			break;
		case CElementBSDF::Type::MIXTURE_BSDF:
			obj->mixturebsdf = CElementBSDF::MixtureBSDF();
			break;
		case CElementBSDF::Type::BLEND_BSDF:
			obj->blendbsdf = CElementBSDF::BlendBSDF();
			break;
		case CElementBSDF::Type::MASK:
			obj->mask = CElementBSDF::Mask();
			break;
		case CElementBSDF::Type::TWO_SIDED:
			obj->twosided = CElementBSDF::TwoSided();
			break;
		case CElementBSDF::Type::DIFFUSE_TRANSMITTER:
			obj->difftrans = CElementBSDF::DiffuseTransmitter();
			break;
		//case CElementBSDF::Type::HANRAHAN_KRUEGER:
			//hk = CElementBSDF::HanrahanKrueger();
			//break;
		//case CElementBSDF::Type::IRAWAN_MARSCHNER:
			//irawan = CElementBSDF::IrawanMarschner();
			//break;
		default:
			break;
	}
	return CElementFactory::return_type(obj, std::move(name));
}

float CElementBSDF::TransmissiveBase::findIOR(const std::string& name)
{
	static const core::unordered_map<std::string, float, core::CaseInsensitiveHash, core::CaseInsensitiveEquals> NamedIndicesOfRefraction =
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


CElementBSDF::AllConductor::AllConductor(const std::string& material) : RoughSpecularBase(0.1)
{
// TODO fill this out with values from http://www.luxpop.com/HU_v173.cgi?OpCode=73 and https://github.com/mmp/pbrt-v3/blob/master/src/materials/metal.cpp or https://refractiveindex.info/?shelf=main&book=Cu&page=Johnson
	// we use Rec 709 for the Color primaries of this table, so Red ~= 615nm, Green ~= 535nm, Blue ~= 465nm
	static const core::unordered_map<std::string, std::pair<SPropertyElementData,SPropertyElementData>, core::CaseInsensitiveHash, core::CaseInsensitiveEquals> NamedConductors =
	{
#define SPECTRUM_MACRO(R,G,B,X,Y,Z) {SPropertyElementData(SPropertyElementData::Type::RGB,core::vectorSIMDf(R,G,B)),SPropertyElementData(SPropertyElementData::Type::RGB,core::vectorSIMDf(X,Y,Z))}
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
	if (found == NamedConductors.end())
	{
		_NBL_DEBUG_BREAK_IF(true);
		ParserLog::invalidXMLFileStructure("IoR Table lookup not implemented for material preset: " + material);
		found = NamedConductors.find("none");
		assert(found != NamedConductors.end());
	}

	eta = found->second.first;
	k = found->second.second;
	extEta = TransmissiveBase::findIOR("air");
}

bool CElementBSDF::addProperty(SNamedPropertyElement&& _property)
{
	bool error = false;
	auto dispatch = [&](auto func) -> void
	{
		switch (type)
		{
			case CElementBSDF::Type::DIFFUSE:
				[[fallthrough]];
			case CElementBSDF::Type::ROUGHDIFFUSE:
				func(diffuse);
				break;
			case CElementBSDF::Type::DIELECTRIC:
				[[fallthrough]];
			case CElementBSDF::Type::THINDIELECTRIC:
				[[fallthrough]];
			case CElementBSDF::Type::ROUGHDIELECTRIC:
				func(dielectric);
				break;
			case CElementBSDF::Type::CONDUCTOR:
				[[fallthrough]];
			case CElementBSDF::Type::ROUGHCONDUCTOR:
				func(conductor);
				break;
			case CElementBSDF::Type::PLASTIC:
				[[fallthrough]];
			case CElementBSDF::Type::ROUGHPLASTIC:
				func(plastic);
				break;
			case CElementBSDF::Type::COATING:
				[[fallthrough]];
			case CElementBSDF::Type::ROUGHCOATING:
				func(coating);
				break;
			case CElementBSDF::Type::BUMPMAP:
				func(bumpmap);
				break;
			case CElementBSDF::Type::PHONG:
				func(phong);
				break;
			case CElementBSDF::Type::WARD:
				func(ward);
				break;
			case CElementBSDF::Type::MIXTURE_BSDF:
				func(mixturebsdf);
				break;
			case CElementBSDF::Type::BLEND_BSDF:
				func(blendbsdf);
				break;
			case CElementBSDF::Type::MASK:
				func(mask);
				break;
			case CElementBSDF::Type::TWO_SIDED:
				func(twosided);
				break;
			case CElementBSDF::Type::DIFFUSE_TRANSMITTER:
				func(difftrans);
				break;
			//case CElementBSDF::Type::HANRAHAN_KRUEGER:
				//func(hk);
				//break;
			//case CElementBSDF::Type::IRAWAN_MARSCHNER:
				//func(irwan);
				//break;
			default:
				error = true;
				break;
		}
	};

#define SET_FLOAT(MEMBER, ... )		[&]() -> void { \
		dispatch([&](auto& state) -> void { \
			if constexpr (is_any_of<std::remove_reference<decltype(state)>::type,__VA_ARGS__>::value) \
			{ \
				switch (_property.type) { \
					case SPropertyElementData::Type::FLOAT: \
						state. ## MEMBER = SPropertyElementData(_property); \
						break; \
					default: \
						error = true; \
						break; \
				} \
			} \
		}); \
	}
#define SET_SPECTRUM(MEMBER, ... )		[&]() -> void { \
		dispatch([&](auto& state) -> void { \
			if constexpr (is_any_of<std::remove_reference<decltype(state)>::type,__VA_ARGS__>::value) \
			{ \
				switch (_property.type) { \
					case SPropertyElementData::Type::FLOAT: \
					case SPropertyElementData::Type::RGB: \
					case SPropertyElementData::Type::SRGB: \
					case SPropertyElementData::Type::SPECTRUM: \
						state. ## MEMBER = SPropertyElementData(_property); \
						break; \
					default: \
						error = true; \
						break; \
				} \
			} \
		}); \
	}
#define SET_PROPERTY_TEMPLATE(MEMBER,PROPERTY_TYPE, ... )		[&]() -> void { \
		dispatch([&](auto& state) -> void { \
			if constexpr (is_any_of<std::remove_reference<decltype(state)>::type,__VA_ARGS__>::value) \
			{ \
				if (_property.type!=PROPERTY_TYPE) { \
					error = true; \
					return; \
				} \
				state. ## MEMBER = _property.getProperty<PROPERTY_TYPE>(); \
			} \
		}); \
	}

	auto processReflectance = SET_SPECTRUM(reflectance,AllDiffuse);
	auto processDistribution = [&]() -> void
	{ 
		dispatch([&](auto& state) -> void {
			using state_type = std::remove_reference<decltype(state)>::type;

			if constexpr (std::is_base_of<RoughSpecularBase,state_type>::value)
			{
				static const core::unordered_map<std::string, RoughSpecularBase::NormalDistributionFunction, core::CaseInsensitiveHash, core::CaseInsensitiveEquals> StringToType =
				{
					{"beckmann",RoughSpecularBase::NormalDistributionFunction::BECKMANN},
					{"ggx",		RoughSpecularBase::NormalDistributionFunction::GGX},
					{"phong",	RoughSpecularBase::NormalDistributionFunction::PHONG},
					{"as",		RoughSpecularBase::NormalDistributionFunction::ASHIKHMIN_SHIRLEY}
				};

				auto found = StringToType.end();
				if (_property.type==SPropertyElementData::Type::STRING)
					found = StringToType.find(_property.getProperty<SPropertyElementData::Type::STRING>());
				if (found==StringToType.end())
				{
					error = true;
					return;
				}
				state.distribution = found->second;
			}
		});
	};
#define TRANSMISSIVE_TYPES AllDielectric,AllPlastic,AllCoating
#define SPECULAR_TYPES TRANSMISSIVE_TYPES,AllConductor
	auto processAlpha = SET_FLOAT(alpha, AllDiffuse,SPECULAR_TYPES);
	auto processAlphaU = SET_FLOAT(alphaU, SPECULAR_TYPES);
	auto processAlphaV = SET_FLOAT(alphaV, SPECULAR_TYPES);
	auto processUseFastApprox = SET_PROPERTY_TEMPLATE(useFastApprox,SPropertyElementData::Type::BOOLEAN,AllDiffuse);
	auto processIntIOR = [&]() -> void
	{ 
		dispatch([&](auto& state) -> void {
			using state_type = std::remove_reference<decltype(state)>::type;

			if constexpr (is_any_of<state_type,TRANSMISSIVE_TYPES>::value)
			{
				if (_property.type==SPropertyElementData::Type::FLOAT)
					state.intIOR = _property.getProperty<SPropertyElementData::Type::FLOAT>();
				else if (_property.type==SPropertyElementData::Type::STRING)
					state.intIOR = TransmissiveBase::findIOR(_property.getProperty<SPropertyElementData::Type::STRING>());
				else
					error = true;
			}
		});
	};
	auto processExtIOR = [&]() -> void
	{ 
		dispatch([&](auto& state) -> void {
			using state_type = std::remove_reference<decltype(state)>::type;

			if constexpr (is_any_of<state_type,TRANSMISSIVE_TYPES>::value)
			{
				if (_property.type==SPropertyElementData::Type::FLOAT)
					state.extIOR = _property.getProperty<SPropertyElementData::Type::FLOAT>();
				else if (_property.type==SPropertyElementData::Type::STRING)
					state.extIOR = TransmissiveBase::findIOR(_property.getProperty<SPropertyElementData::Type::STRING>());
				else
					error = true;
			}
		});
	};
	auto processSpecularReflectance = SET_SPECTRUM(specularReflectance, SPECULAR_TYPES,Phong,Ward);
	auto processDiffuseReflectance = SET_SPECTRUM(diffuseReflectance, AllDiffuse,AllPlastic,Phong,Ward);
	auto processSpecularTransmittance = SET_SPECTRUM(specularTransmittance, TRANSMISSIVE_TYPES);
#undef SPECULAR_TYPES
#undef TRANSMISSIVE_TYPES
	auto processMaterial = [&]() -> void
	{ 
		dispatch([&](auto& state) -> void {
			using state_type = std::remove_reference<decltype(state)>::type;

			if constexpr (std::is_same<state_type,AllConductor>::value)
			{
				if (_property.type == SPropertyElementData::Type::STRING)
					conductor = AllConductor(_property.getProperty<SPropertyElementData::Type::STRING>());
				else
					error = true;
			}/*
			else
			{
				if constexpr (std::is_same<state_type,HanrahanKrueger>::value)
				{
				}
			}*/
		});
	};
	auto processEta = SET_SPECTRUM(eta, AllConductor);
	auto processK = SET_SPECTRUM(k, AllConductor);
	auto processExtEta = [&]() -> void
	{ 
		dispatch([&](auto& state) -> void {
			using state_type = std::remove_reference<decltype(state)>::type;

			if constexpr (std::is_same<state_type,AllConductor>::value)
			{
				if (_property.type==SPropertyElementData::Type::FLOAT)
					state.extEta = _property.getProperty<SPropertyElementData::Type::FLOAT>();
				else if (_property.type==SPropertyElementData::Type::STRING)
					state.extEta = TransmissiveBase::findIOR(_property.getProperty<SPropertyElementData::Type::STRING>());
				else
					error = true;
			}
		});
	};
	auto processNonlinear = SET_PROPERTY_TEMPLATE(nonlinear, SPropertyElementData::Type::BOOLEAN, AllPlastic);
	auto processThickness = SET_PROPERTY_TEMPLATE(thickness, SPropertyElementData::Type::FLOAT, AllCoating);
	auto processSigmaA = SET_SPECTRUM(sigmaA, AllCoating);
	auto processExponent = SET_FLOAT(exponent, Phong);
	auto processVariant = [&]() -> void
	{ 
		dispatch([&](auto& state) -> void {
			using state_type = std::remove_reference<decltype(state)>::type;

			if constexpr (std::is_same<state_type,Ward>::value)
			{
				static const core::unordered_map<std::string,Ward::Type,core::CaseInsensitiveHash,core::CaseInsensitiveEquals> StringToType =
				{
					{"ward",		Ward::Type::WARD},
					{"ward-duer",	Ward::Type::WARD_DUER},
					{"balanced",	Ward::Type::BALANCED}
				};
				auto found = StringToType.end();
				if (_property.type==SPropertyElementData::Type::STRING)
					found = StringToType.find(_property.getProperty<SPropertyElementData::Type::STRING>());
				if (found==StringToType.end())
				{
					error = true;
					return;
				}
				state.variant = found->second;
			}
		});
	};
	auto processWeights = [&]() -> void
	{ 
		dispatch([&](auto& state) -> void {
			using state_type = std::remove_reference<decltype(state)>::type;

			if constexpr (std::is_same<state_type,MixtureBSDF>::value)
			{
				if (_property.type == SPropertyElementData::Type::STRING)
				{
					error = true;
					return;
				}

				std::istringstream sstr(_property.getProperty<SPropertyElementData::Type::STRING>());
				std::string token;
				while (std::getline(sstr, token, ','))
					state.weights[state.weightCount++] = std::stof(token);
			}
		});
	};
	auto processWeight = SET_FLOAT(weight, BlendBSDF);
	auto processOpacity = SET_SPECTRUM(opacity, Mask);
	auto processTransmittance = SET_SPECTRUM(transmittance, DiffuseTransmitter);
	// TODO: set HK and IRAWAN parameters
	/*
	auto processField = [&]() -> void
	{
		dispatch([&](auto& state) -> void
		{
			using state_type = std::remove_reference<decltype(state)>::type;
			if constexpr (std::is_same<state_type,FieldExtraction>::value)
			{
				if (_property.type != SPropertyElementData::Type::STRING)
				{
					error = true;
					return;
				}
				auto found = StringToType.find(_property.svalue);
				if (found!=StringToType.end())
					state.field = found->second;
				else
					state.field = FieldExtraction::Type::INVALID;
			}
		});
	};
	*/
#undef SET_FLOAT
#undef SET_SPECTRUM
#undef SET_PROPERTY_TEMPLATE
	const core::unordered_map<std::string, std::function<void()>, core::CaseInsensitiveHash, core::CaseInsensitiveEquals> SetPropertyMap =
	{
		{"reflectance",				processReflectance},
		{"distribution",			processDistribution},
		{"alpha",					processAlpha},
		{"alphaU",					processAlphaU},
		{"alphaV",					processAlphaV},
		{"useFastApprox",			processUseFastApprox},
		{"intIOR",					processIntIOR},
		{"extIOR",					processExtIOR},
		{"specularReflectance",		processSpecularReflectance},
		{"diffuseReflectance",		processDiffuseReflectance},
		{"specularTransmittance",	processSpecularTransmittance},
		{"material",				processMaterial},
		{"eta",						processEta},
		{"k",						processK},
		{"extEta",					processExtEta},
		{"nonlinear",				processNonlinear},
		{"thickness",				processThickness},
		{"sigmaA",					processSigmaA},
		{"exponent",				processExponent},
		{"variant",					processVariant},
		{"weights",					processWeights},
		{"weight",					processWeight},
		{"opacity",					processOpacity},
		{"transmittance",			processTransmittance}//,
		//{"sigmaS",				processSigmaS},
		//{"sigmaT",				processSigmaT},
		//{"albedo",				processAlbedo},
		//{"filename",				processFilename},
		//{"repeatU",				processRepeatU},
		//{"repeatV",				processRepeatV}
	};
	
	auto found = SetPropertyMap.find(_property.name);
	if (found==SetPropertyMap.end())
	{
		ParserLog::invalidXMLFileStructure("No BSDF can have such property set with name: "+_property.name);
		return false;
	}

	found->second();
	return !error;
}


bool CElementBSDF::processChildData(IElement* _child, const std::string& name)
{
	if (!_child)
		return true;

	switch (_child->getType())
	{
		case IElement::Type::TEXTURE:
			{
				auto _texture = static_cast<CElementTexture*>(_child);

				bool error = false;
				auto dispatch = [&](auto func) -> void
				{
					switch (type)
					{
						case CElementBSDF::Type::DIFFUSE:
							[[fallthrough]];
						case CElementBSDF::Type::ROUGHDIFFUSE:
							func(diffuse);
							break;
						case CElementBSDF::Type::DIELECTRIC:
							[[fallthrough]];
						case CElementBSDF::Type::THINDIELECTRIC:
							[[fallthrough]];
						case CElementBSDF::Type::ROUGHDIELECTRIC:
							func(dielectric);
							break;
						case CElementBSDF::Type::CONDUCTOR:
							[[fallthrough]];
						case CElementBSDF::Type::ROUGHCONDUCTOR:
							func(conductor);
							break;
						case CElementBSDF::Type::PLASTIC:
							[[fallthrough]];
						case CElementBSDF::Type::ROUGHPLASTIC:
							func(plastic);
							break;
						case CElementBSDF::Type::COATING:
							[[fallthrough]];
						case CElementBSDF::Type::ROUGHCOATING:
							func(coating);
							break;
						case CElementBSDF::Type::BUMPMAP:
							func(bumpmap);
							break;
						case CElementBSDF::Type::PHONG:
							func(phong);
							break;
						case CElementBSDF::Type::WARD:
							func(ward);
							break;
						case CElementBSDF::Type::MIXTURE_BSDF:
							func(mixturebsdf);
							break;
						case CElementBSDF::Type::BLEND_BSDF:
							func(blendbsdf);
							break;
						case CElementBSDF::Type::MASK:
							func(mask);
							break;
						case CElementBSDF::Type::TWO_SIDED:
							func(twosided);
							break;
						case CElementBSDF::Type::DIFFUSE_TRANSMITTER:
							func(difftrans);
							break;
						//case CElementBSDF::Type::HANRAHAN_KRUEGER:
							//func(hk);
							//break;
						//case CElementBSDF::Type::IRAWAN_MARSCHNER:
							//func(irwan);
							//break;
						default:
							error = true;
							break;
					}
				};
#define SET_TEXTURE(MEMBER, ... )		[&]() -> void { \
					dispatch([&](auto& state) -> void { \
						if constexpr (is_any_of<std::remove_reference<decltype(state)>::type,__VA_ARGS__>::value) \
						{ \
							state. ## MEMBER.value.type = SPropertyElementData::Type::INVALID; \
							state. ## MEMBER.texture = _texture; \
						} \
					}); \
				}

				auto processReflectance = SET_TEXTURE(reflectance, AllDiffuse);
#define TRANSMISSIVE_TYPES AllDielectric,AllPlastic,AllCoating
#define SPECULAR_TYPES TRANSMISSIVE_TYPES,AllConductor
				auto processAlpha = SET_TEXTURE(alpha, AllDiffuse,SPECULAR_TYPES);
				auto processAlphaU = SET_TEXTURE(alphaU, SPECULAR_TYPES);
				auto processAlphaV = SET_TEXTURE(alphaV, SPECULAR_TYPES);
				auto processSpecularReflectance = SET_TEXTURE(specularReflectance, SPECULAR_TYPES,Phong,Ward);
				auto processDiffuseReflectance = SET_TEXTURE(diffuseReflectance, AllDiffuse,AllPlastic,Phong,Ward);
				auto processSpecularTransmittance = SET_TEXTURE(specularTransmittance, TRANSMISSIVE_TYPES);
				auto processSigmaA = SET_TEXTURE(sigmaA, AllCoating);
				auto processExponent = SET_TEXTURE(exponent, Phong);
				auto processWeight = SET_TEXTURE(weight, BlendBSDF);
				auto processOpacity = SET_TEXTURE(opacity, Mask);
				auto processTransmittance = SET_TEXTURE(transmittance, DiffuseTransmitter);
#undef TRANSMISSIVE_TYPES
#undef SPECULAR_TYPES
#undef SET_TEXTURE

				const core::unordered_map<std::string, std::function<void()>, core::CaseInsensitiveHash, core::CaseInsensitiveEquals> SetChildMap =
				{
					{"reflectance",				processReflectance},
					{"alpha",					processAlpha},
					{"alphaU",					processAlphaU},
					{"alphaV",					processAlphaV},
					{"specularReflectance",		processSpecularReflectance},
					{"diffuseReflectance",		processDiffuseReflectance},
					{"specularTransmittance",	processSpecularTransmittance},
					{"sigmaA",					processSigmaA},
					{"exponent",				processExponent},
					{"weight",					processWeight},
					{"opacity",					processOpacity},
					{"transmittance",			processTransmittance}//,
					//{"sigmaS",				processSigmaS},
					//{"sigmaT",				processSigmaT},
					//{"albedo",				processAlbedo}
				};

				auto found = SetChildMap.find(name);
				if (found==SetChildMap.end())
				{
					switch (type)
					{
						case Type::BUMPMAP:
							bumpmap.texture = _texture;
							break;
						default:
							_NBL_DEBUG_BREAK_IF(true);
							ParserLog::invalidXMLFileStructure("No BSDF can have such property set with name: " + name);
							return false;
							break;
					}
				}
				else
					found->second();

				if (error)
					return false;
			}
			break;
		case IElement::Type::BSDF:
			{
				auto _bsdf = static_cast<CElementBSDF*>(_child);
				switch (type)
				{
					case Type::COATING:
						[[fallthrough]];
					case Type::ROUGHCOATING:
						if (coating.childCount < AllCoating::MaxChildCount)
							coating.bsdf[coating.childCount++] = _bsdf;
						else
							return false;
						break;
					case Type::BUMPMAP:
						if (bumpmap.childCount < BumpMap::MaxChildCount)
							bumpmap.bsdf[bumpmap.childCount++] = _bsdf;
						else
							return false;
						break;
					case Type::MIXTURE_BSDF:
						if (mixturebsdf.childCount < MixtureBSDF::MaxChildCount)
							mixturebsdf.bsdf[mixturebsdf.childCount++] = _bsdf;
						else
							return false;
						break;
					case Type::BLEND_BSDF:
						if (blendbsdf.childCount < BlendBSDF::MaxChildCount)
							blendbsdf.bsdf[blendbsdf.childCount++] = _bsdf;
						else
							return false;
						break;
					case Type::MASK:
						if (mask.childCount < Mask::MaxChildCount)
							mask.bsdf[mask.childCount++] = _bsdf;
						else
							return false;
						break;
					case Type::TWO_SIDED:
						if (twosided.childCount < TwoSided::MaxChildCount)
							twosided.bsdf[twosided.childCount++] = _bsdf;
						else
							return false;
						break;
					default:
						return false;
						break;
				}
			}
			break;
		default:
			return false;
			break;
	}
	return true;
}

bool CElementBSDF::onEndTag(asset::IAssetLoader::IAssetLoaderOverride* _override, CMitsubaMetadata* metadata)
{
	if (type == Type::INVALID)
	{
		ParserLog::invalidXMLFileStructure(getLogName() + ": type not specified");
		_NBL_DEBUG_BREAK_IF(true);
		return true;
	}
	
	// TODO: Validation
	{
	}

	return true;
}

}
}
}