#include "../../ext/MitsubaLoader/ParserUtil.h"
#include "../../ext/MitsubaLoader/CElementFactory.h"

#include "../../ext/MitsubaLoader/CGlobalMitsubaMetadata.h"

#include <functional>

namespace irr
{
namespace ext
{
namespace MitsubaLoader
{


template<>
IElement* CElementFactory::createElement<CElementBSDF>(const char** _atts, ParserManager* _util)
{
	const char* type;
	const char* id;
	if (!IElement::getTypeAndIDStrings(type, id, _atts))
		return nullptr;

	static const core::unordered_map<std::string, CElementBSDF::Type, core::CaseInsensitiveHash, core::CaseInsensitiveEquals> StringToType =
	{
		{"diffuse",			CElementBSDF::Type::DIFFUSE},
		{"roughdiffuse",	CElementBSDF::Type::ROUGHDIFFUSE},
		{"dielectric",		CElementBSDF::Type::PATH},
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
		_IRR_DEBUG_BREAK_IF(false);
		return nullptr;
	}

	CElementBSDF* obj = _util->objects.construct<CElementBSDF>(id);
	if (!obj)
		return nullptr;

	obj->type = found->second;
	// defaults
	switch (obj->type)
	{
		case CElementBSDF::Type::DIFFUSE:
			IRR_FALLTHROUGH;
		case CElementBSDF::Type::ROUGHDIFFUSE:
			diffuse = AllDiffuse();
			break;
		case CElementBSDF::Type::DIELECTRIC:
			IRR_FALLTHROUGH;
		case CElementBSDF::Type::THINDIELECTRIC:
			IRR_FALLTHROUGH;
		case CElementBSDF::Type::ROUGHDIELECTRIC:
			dieletric = AllDielectric();
			break;
		case CElementBSDF::Type::CONDUCTOR:
			IRR_FALLTHROUGH;
		case CElementBSDF::Type::ROUGHCONDUCTOR:
			conductor = AllConductor();
			break;
		case CElementBSDF::Type::PLASTIC:
			IRR_FALLTHROUGH;
		case CElementBSDF::Type::ROUGHPLASTIC:
			plastic = AllPlastic();
			break;
		case CElementBSDF::Type::COATING:
			IRR_FALLTHROUGH;
		case CElementBSDF::Type::ROUGHCOATING:
			coating = AllCoating();
			break;
		case CElementBSDF::Type::BUMPMAP:
			bumpmap = BumpMap();
			break;
		case CElementBSDF::Type::PHONG:
			phong = Phong();
			break;
		case CElementBSDF::Type::WARD:
			ward = Ward();
			break;
		case CElementBSDF::Type::MIXTURE_BSDF:
			mixturebsdf = MixtureBSDF();
			break;
		case CElementBSDF::Type::BLEND_BSDF:
			blendbsdf = BlendBSDF();
			break;
		case CElementBSDF::Type::MASK:
			mask = Mask();
			break;
		case CElementBSDF::Type::TWO_SIDED:
			twosided = TwoSided();
			break;
		case CElementBSDF::Type::DIFFUSE_TRANSMITTER:
			difftrans = DiffuseTransmitter();
			break;
		//case CElementBSDF::Type::HANRAHAN_KRUEGER:
			//hk = HanrahanKrueger();
			//break;
		//case CElementBSDF::Type::IRAWAN_MARSCHNER:
			//irawan = IrawanMarschner();
			//break;
		default:
			break;
	}
	return obj;
}


const core::unordered_map<std::string, float, core::CaseInsensitiveHash, core::CaseInsensitiveEquals> AllDielectric::NamedIndicesOfRefraction =
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

CElementBSDF::AllConductor::AllConductor(const std::string& material) : RoughSpecularBase(0.1)
{
// TODO fill this out with values from http://www.luxpop.com/HU_v173.cgi?OpCode=73 and https://github.com/mmp/pbrt-v3/blob/master/src/materials/metal.cpp or https://refractiveindex.info/?shelf=main&book=Cu&page=Johnson
	// we use Rec 709 for the Color primaries of this table, so Red ~= 615nm, Green ~= 535nm, Blue ~= 465nm
	static const core::unordered_map<std::string, std::pair<SPropertyElementData,SPropertyElementData>, core::CaseInsensitiveHash, core::CaseInsensitiveEquals> NamedConductors =
	{
#define SPECTRUM_MACRO(R,G,B,X,Y,Z) {SPropertyElementData(SPropertyElementData::Type::RGB,core::vectorSIMDf(R,G,B)),SPropertyElementData(SPropertyElementData::Type::RGB,core::vectorSIMDf(X,Y,Z))}
		//{"a-C",				SPECTRUM_MACRO(,,,		,,)},
		//{"Ag",				SPECTRUM_MACRO(,,,		,,)},
		//{"Al",				SPECTRUM_MACRO(,,,		,,)},
		//{"AlAs",			SPECTRUM_MACRO(,,,		,,)},
		//{"AlAs_palik",		SPECTRUM_MACRO(,,,		,,)},
		//{"Au",				SPECTRUM_MACRO(,,,		,,)},
		//{"Be",				SPECTRUM_MACRO(,,,		,,)},
		//{"Be_palik",		SPECTRUM_MACRO(,,,		,,)},
		//{"Cr",				SPECTRUM_MACRO(,,,		,,)},
		//{"CsI",				SPECTRUM_MACRO(,,,		,,)},
		//{"CsI_palik",		SPECTRUM_MACRO(,,,		,,)},
		{"Cu",				SPECTRUM_MACRO(0.32075f,1.09860f,1.2469f,		3.17900f,2.59220f,2.4562)},
		//{"Cu_palik",		SPECTRUM_MACRO(,,,		,,)},
		//{"Cu20",			SPECTRUM_MACRO(,,,		,,)},
		//{"Cu20_palik",	SPECTRUM_MACRO(,,,		,,)},
		//{"d-C",			SPECTRUM_MACRO(,,,		,,)},
		//{"d-C_palik",		SPECTRUM_MACRO(,,,		,,)},
		//{"Hg",			SPECTRUM_MACRO(,,,		,,)},
		//{"Hg_palik",		SPECTRUM_MACRO(,,,		,,)},
		//{"HgTe",			SPECTRUM_MACRO(,,,		,,)},
		//{"HgTe_palik",		SPECTRUM_MACRO(,,,		,,)},
		//{"Ir",			SPECTRUM_MACRO(,,,		,,)},
		//{"Ir_palik",		SPECTRUM_MACRO(,,,		,,)},
		//{"K",			SPECTRUM_MACRO(,,,		,,)},
		//{"K_palik",		SPECTRUM_MACRO(,,,		,,)},
		//{"Li",			SPECTRUM_MACRO(,,,		,,)},
		//{"Li_palik",		SPECTRUM_MACRO(,,,		,,)},
		//{"MgO",			SPECTRUM_MACRO(,,,		,,)},
		//{"MgO_palik",		SPECTRUM_MACRO(,,,		,,)},
		//{"Mo",			SPECTRUM_MACRO(,,,		,,)},
		//{"Mo_palik",		SPECTRUM_MACRO(,,,		,,)},
		//{"Na_palik",			SPECTRUM_MACRO(,,,		,,)},
		//{"Nb",			SPECTRUM_MACRO(,,,		,,)},
		//{"Nb_palik",			SPECTRUM_MACRO(,,,		,,)},
		//{"Ni_palik",			SPECTRUM_MACRO(,,,		,,)},
		//{"Rh",			SPECTRUM_MACRO(,,,		,,)},
		//{"Rh_palik",		SPECTRUM_MACRO(,,,		,,)},
		//{"Se",			SPECTRUM_MACRO(,,,		,,)},
		//{"Se_palik",		SPECTRUM_MACRO(,,,		,,)},
		//{"SiC",			SPECTRUM_MACRO(,,,		,,)},
		//{"SiC_palik",		SPECTRUM_MACRO(,,,		,,)},
		//{"SnTe",			SPECTRUM_MACRO(,,,		,,)},
		//{"SnTe_palik",		SPECTRUM_MACRO(,,,		,,)},
		//{"Ta",			SPECTRUM_MACRO(,,,		,,)},
		//{"Ta_palik",		SPECTRUM_MACRO(,,,		,,)},
		//{"Te",			SPECTRUM_MACRO(,,,		,,)},
		//{"Te_palik",		SPECTRUM_MACRO(,,,		,,)},
		//{"ThF4",			SPECTRUM_MACRO(,,,		,,)},
		//{"ThF4_palik",		SPECTRUM_MACRO(,,,		,,)},
		//{"TiC",			SPECTRUM_MACRO(,,,		,,)},
		//{"TiC_palik",		SPECTRUM_MACRO(,,,		,,)},
		//{"TiO2",			SPECTRUM_MACRO(,,,		,,)},
		//{"TiO2_palik",		SPECTRUM_MACRO(,,,		,,)},
		//{"VC",			SPECTRUM_MACRO(,,,		,,)},
		//{"VC_palik",		SPECTRUM_MACRO(,,,		,,)},
		//{"V_palik",		SPECTRUM_MACRO(,,,		,,)},
		//{"VN",			SPECTRUM_MACRO(,,,		,,)},
		//{"VN_palik",		SPECTRUM_MACRO(,,,		,,)},
		//{"W",		SPECTRUM_MACRO(,,,		,,)},
		{"none",			SPECTRUM_MACRO(0.f,0.f,0.f,		0.f,0.f,0.f)}
#undef SPECTRUM_MACRO
	};

	auto found = NamedConductors.find(material);
	if (found == NamedConductors.end())
	{
		_IRR_DEBUG_BREAK_IF(true);
		ParserLog::invalidXMLFileStructure("IoR Table lookup not implemented for material preset: " + material);
		found = NamedConductors.find("none");
		assert(found != NamedConductors.end());
	}

	eta = found->second->first;
	k = found->second->second;
	extEta = TransmissiveBase::NamedIndicesOfRefraction["air"];
}

bool CElementBSDF::addProperty(SPropertyElementData&& _property)
{
	bool error = false;
	auto dispatch = [&](auto func) -> void
	{
		switch (type)
		{
			case CElementBSDF::Type::DIFFUSE:
				IRR_FALLTHROUGH;
			case CElementBSDF::Type::ROUGHDIFFUSE:
				func(diffuse);
				break;
			case CElementBSDF::Type::DIELECTRIC:
				IRR_FALLTHROUGH;
			case CElementBSDF::Type::THINDIELECTRIC:
				IRR_FALLTHROUGH;
			case CElementBSDF::Type::ROUGHDIELECTRIC:
				func(dielectric);
				break;
			case CElementBSDF::Type::CONDUCTOR:
				IRR_FALLTHROUGH;
			case CElementBSDF::Type::ROUGHCONDUCTOR:
				func(conductor);
				break;
			case CElementBSDF::Type::PLASTIC:
				IRR_FALLTHROUGH;
			case CElementBSDF::Type::ROUGHPLASTIC:
				func(plastic);
				break;
			case CElementBSDF::Type::COATING:
				IRR_FALLTHROUGH;
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

#define SET_PROPERTY_TEMPLATE(MEMBER,PROPERTY_TYPE, ... )		[&]() -> void { \
		dispatch([&](auto& state) -> void { \
			IRR_PSEUDO_IF_CONSTEXPR_BEGIN(is_any_of<std::remove_reference<decltype(state)>::type,__VA_ARGS__>::value) \
			{ \
				if (_property.type!=PROPERTY_TYPE) { \
					error = true; \
					return; \
				} \
				state. ## MEMBER = _property.getProperty<PROPERTY_TYPE>(); \
			} \
			IRR_PSEUDO_IF_CONSTEXPR_END \
		}); \
	}

	auto processReflectance = SET_XYZ(reflectance,AllDiffuse);
	auto processDistribution = [&]() -> void
	{ 
		dispatch([&](auto& state) -> void {
			using state_type = std::remove_reference<decltype(state)>::type;

			IRR_PSEUDO_IF_CONSTEXPR_BEGIN(std::is_same<state_type,AllDielectric>::value)
			{
				found;
				if (_property.type==SPropertyElementData::Type::STRING)
					found = [_property.getProperty<SPropertyElementData::Type::STRING>()];
				if (found!=.end())
				{
					error = true;
					return;
				}
				distribution = found->second;
			}
			IRR_PSEUDO_IF_CONSTEXPR_END
		});
	};
	auto processAlpha = SET_XYZ(alpha,AllDiffuse);
	auto processAlphaU = SET_XYZ(alphaU,AllDiffuse);
	auto processAlphaV = SET_XYZ(alphaV,AllDiffuse);
	auto processUseFastApprox = SET_PROPERTY_TEMPLATE(useFastApprox,SPropertyElementData::Type::BOOLEAN,AllDiffuse);
	auto processIntIOR = [&]() -> void
	{ 
		dispatch([&](auto& state) -> void {
			using state_type = std::remove_reference<decltype(state)>::type;

			IRR_PSEUDO_IF_CONSTEXPR_BEGIN(std::is_same<state_type,AllDielectric>::value)
			{
				if (_property.type==SPropertyElementData::Type::FLOAT)
					state.intIOR = _property.getProperty<SPropertyElementData::Type::FLOAT>();
				else if (_property.type==SPropertyElementData::Type::STRING)
					state.intIOR = TransmissiveBase::NamedIndicesOfRefraction[_property.getProperty<SPropertyElementData::Type::STRING>()];
				else
					error = true;
			}
			IRR_PSEUDO_IF_CONSTEXPR_END
		});
	};
	auto processExtIOR = [&]() -> void
	{ 
		dispatch([&](auto& state) -> void {
			using state_type = std::remove_reference<decltype(state)>::type;

			IRR_PSEUDO_IF_CONSTEXPR_BEGIN(std::is_same<state_type,AllDielectric>::value)
			{
				if (_property.type==SPropertyElementData::Type::FLOAT)
					state.extIOR = _property.getProperty<SPropertyElementData::Type::FLOAT>();
				else if (_property.type==SPropertyElementData::Type::STRING)
					state.extIOR = TransmissiveBase::NamedIndicesOfRefraction[_property.getProperty<SPropertyElementData::Type::STRING>()];
				else
					error = true;
			}
			IRR_PSEUDO_IF_CONSTEXPR_END
		});
	};
	auto processSpecularReflectance = SET_XYZ(specularReflectance, AllDielectric);
	auto processDiffuseReflectance = SET_XYZ(diffuseReflectance, AllDielectric);
	auto processSpecularTransmittance = SET_XYZ(specularTransmittance, AllDielectric);
	auto processMaterial = [&]() -> void
	{ 
		dispatch([&](auto& state) -> void {
			using state_type = std::remove_reference<decltype(state)>::type;

			IRR_PSEUDO_IF_CONSTEXPR_BEGIN(std::is_same<state_type,AllConductor>::value)
			{
				found;
				if (_property.type==SPropertyElementData::Type::STRING)
					found = [_property.getProperty<SPropertyElementData::Type::STRING>()];
				if (found!=.end())
				{
					error = true;
					return;
				}

				eta = ;
				k = ;
			}/*
			IRR_PSEUDO_IF_CONSTEXPR_ELSE
			{
				IRR_PSEUDO_IF_CONSTEXPR_BEGIN(std::is_same<state_type,HanrahanKrueger>::value)
				{
				}
				IRR_PSEUDO_IF_CONSTEXPR_END
			}*/
			IRR_PSEUDO_IF_CONSTEXPR_END
		});
	};
	auto processEta = SET_XYZ(eta, AllConductor);
	auto processK = SET_XYZ(k, AllConductor);
	auto processExtEta = [&]() -> void
	{ 
		dispatch([&](auto& state) -> void {
			using state_type = std::remove_reference<decltype(state)>::type;

			IRR_PSEUDO_IF_CONSTEXPR_BEGIN(std::is_same<state_type,AllConductor>::value)
			{
				if (_property.type==SPropertyElementData::Type::FLOAT)
					state.extEta = _property.getProperty<SPropertyElementData::Type::FLOAT>();
				else if (_property.type==SPropertyElementData::Type::STRING)
					state.extEta = TransmissiveBase::NamedIndicesOfRefraction[_property.getProperty<SPropertyElementData::Type::STRING>()];
				else
					error = true;
			}
			IRR_PSEUDO_IF_CONSTEXPR_END
		});
	};
	auto processNonlinear = SET_PROPERTY_TEMPLATE(nonlinear, SPropertyElementData::Type::BOOLEAN, AllPlastic);
	auto processThickness = SET_PROPERTY_TEMPLATE(thickness, SPropertyElementData::Type::FLOAT, AllCoating);
	auto processSigmaA = SET_XYZ(sigmaA, AllCoating);
	auto processExponent = SET_XYZ(exponent, Phong);
	auto processVariant = [&]() -> void
	{ 
		dispatch([&](auto& state) -> void {
			using state_type = std::remove_reference<decltype(state)>::type;

			IRR_PSEUDO_IF_CONSTEXPR_BEGIN(std::is_same<state_type,Ward>::value)
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
				if (found!=StringToType.end())
				{
					error = true;
					return;
				}
				state.variant = found->second;
			}
			IRR_PSEUDO_IF_CONSTEXPR_END
		});
	};
	auto processWeights = [&]() -> void
	{ 
		dispatch([&](auto& state) -> void {
			using state_type = std::remove_reference<decltype(state)>::type;

			IRR_PSEUDO_IF_CONSTEXPR_BEGIN(std::is_same<state_type,MixtureBSDF>::value)
			{
				std::istringstream sstr;
				if (_property.type==SPropertyElementData::Type::STRING)
					sstr = _property.getProperty<SPropertyElementData::Type::STRING>();
				
				while (pop)
				{
					state.weights[state.weight++] = ;
				}
			}
			IRR_PSEUDO_IF_CONSTEXPR_END
		});
	};
	auto processWeight = SET_XYZ(weight, BlendBSDF);
	auto processOpacity = SET_XYZ(opacity, Mask);
	auto processTransmittance = SET_XYZ(transmittance, DiffuseTransmitter);
	// TODO: set HK and IRAWAN parameters
	/*
	auto processField = [&]() -> void
	{
		dispatch([&](auto& state) -> void
		{
			using state_type = std::remove_reference<decltype(state)>::type;
			IRR_PSEUDO_IF_CONSTEXPR_BEGIN(std::is_same<state_type,FieldExtraction>::value)
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
			IRR_PSEUDO_IF_CONSTEXPR_END
		});
	};
	*/
	static const core::unordered_map<std::string, std::function<void()>, core::CaseInsensitiveHash, core::CaseInsensitiveEquals> SetPropertyMap =
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
		_IRR_DEBUG_BREAK_IF(true);
		ParserLog::invalidXMLFileStructure("No BSDF can have such property set with name: "+_property.name);
		return false;
	}

	found->second();
	return !error;
}


bool CElementBSDF::processChildData(IElement* _child)
{
	if (!_child)
		return true;

	switch (_child->getType())
	{
		case IElement::Type::TEXTURE:
			{
				static const core::unordered_map<std::string, std::function<void()>, core::CaseInsensitiveHash, core::CaseInsensitiveEquals> SetPropertyMap =
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
					{"transmittance",			processTransmittance}//,
					//{"sigmaS",				processSigmaS},
					//{"sigmaT",				processSigmaT},
					//{"albedo",				processAlbedo}
				};

				auto _texture = static_cast<CElementTexture*>(_child);
				auto found = SetChildMap.find(_child.name);
				if (found==SetChildMap.end())
				{
					switch (type)
					{
						case Type::BUMPMAP:
							bumpmap.texture = _texture;
							break;
						default:
							_IRR_DEBUG_BREAK_IF(true);
							ParserLog::invalidXMLFileStructure("No BSDF can have such property set with name: " + _child.name);
							return false;
							break;
					}
				}
				else
					found->second();
			}
			break;
		case IElement::Type::BSDF:
			{
				auto _bsdf = static_cast<CElementBSDF*>(_child);
				switch (type)
				{
					case Type::COATING:
						IRR_FALLTHROUGH;
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
					case Type::TWOSIDED:
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

bool CElementBSDF::onEndTag(asset::IAssetLoader::IAssetLoaderOverride* _override, CGlobalMitsubaMetadata* globalMetadata)
{
	if (type == Type::INVALID)
	{
		ParserLog::invalidXMLFileStructure(getLogName() + ": type not specified");
		_IRR_DEBUG_BREAK_IF(true);
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