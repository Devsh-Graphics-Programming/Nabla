#include "../../ext/MitsubaLoader/ParserUtil.h"
#include "../../ext/MitsubaLoader/CElementFactory.h"

#include <functional>

namespace irr
{
namespace ext
{
namespace MitsubaLoader
{

template<>
CElementFactory::return_type CElementFactory::createElement<CElementEmitter>(const char** _atts, ParserManager* _util)
{
	const char* type;
	const char* id;
	std::string name;
	if (!IElement::getTypeIDAndNameStrings(type, id, name, _atts))
		return CElementFactory::return_type(nullptr,"");

	static const core::unordered_map<std::string, CElementEmitter::Type, core::CaseInsensitiveHash, core::CaseInsensitiveEquals> StringToType =
	{
		{"point",		CElementEmitter::Type::POINT},
		{"area",		CElementEmitter::Type::AREA},
		{"spot",		CElementEmitter::Type::SPOT},
		{"directional",	CElementEmitter::Type::DIRECTIONAL},
		{"collimated",	CElementEmitter::Type::COLLIMATED},/*
		{"sky",			CElementEmitter::Type::SKY},
		{"sun",			CElementEmitter::Type::SUN},
		{"sunsky",		CElementEmitter::Type::SUNSKY},*/
		{"envmap",		CElementEmitter::Type::ENVMAP},
		{"constant",	CElementEmitter::Type::CONSTANT}
	};

	auto found = StringToType.find(type);
	if (found==StringToType.end())
	{
		ParserLog::invalidXMLFileStructure("unknown type");
		_IRR_DEBUG_BREAK_IF(false);
		return CElementFactory::return_type(nullptr, "");
	}

	CElementEmitter* obj = _util->objects.construct<CElementEmitter>(id);
	if (!obj)
		return CElementFactory::return_type(nullptr, "");

	obj->type = found->second;
	// defaults
	switch (obj->type)
	{
		case CElementEmitter::Type::POINT:
			obj->point = CElementEmitter::Point();
			break;
		case CElementEmitter::Type::AREA:
			obj->area = CElementEmitter::Area();
			break;
		case CElementEmitter::Type::SPOT:
			obj->spot = CElementEmitter::Spot();
			break;
		case CElementEmitter::Type::DIRECTIONAL:
			obj->directional = CElementEmitter::Directional();
			break;
		case CElementEmitter::Type::COLLIMATED:
			obj->collimated = CElementEmitter::Collimated();
			break;/*
		case CElementEmitter::Type::SKY:
			obj->sky = CElementEmitter::Sky();
			break;
		case CElementEmitter::Type::SUN:
			obj->ply = CElementEmitter::Sun();
			break;
		case CElementEmitter::Type::SUNSKY:
			obj->serialized = CElementEmitter::SunSky();
			break;*/
		case CElementEmitter::Type::ENVMAP:
			obj->envmap = CElementEmitter::EnvMap();
			break;
		case CElementEmitter::Type::CONSTANT:
			obj->constant = CElementEmitter::Constant();
			break;
		default:
			break;
	}
	return CElementFactory::return_type(obj, std::move(name));
}

bool CElementEmitter::addProperty(SNamedPropertyElement&& _property)
{
	bool error = false;
	auto dispatch = [&](auto func) -> void
	{
		switch (type)
		{
			case Type::POINT:
				func(point);
				break;
			case Type::AREA:
				func(area);
				break;
			case Type::SPOT:
				func(spot);
				break;
			case Type::DIRECTIONAL:
				func(directional);
				break;
			case Type::COLLIMATED:
				func(collimated);
				break;/*
			case Type::SKY:
				func(sky);
				break;
			case Type::SUN:
				func(sun);
				break;
			case Type::SUNSKY:
				func(sunsky);
				break;*/
			case Type::ENVMAP:
				func(envmap);
				break;
			case Type::CONSTANT:
				func(constant);
				break;
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
#define SET_SPECTRUM(MEMBER, ... )		[&]() -> void { \
		dispatch([&](auto& state) -> void { \
			IRR_PSEUDO_IF_CONSTEXPR_BEGIN(is_any_of<std::remove_reference<decltype(state)>::type,__VA_ARGS__>::value) \
			{ \
				switch (_property.type) { \
					case SPropertyElementData::Type::FLOAT: \
						state. ## MEMBER.x = state. ## MEMBER.y = state. ## MEMBER.z = _property.getProperty<SPropertyElementData::Type::FLOAT>(); \
						break; \
					case SPropertyElementData::Type::RGB: \
						state. ## MEMBER = _property.getProperty<SPropertyElementData::Type::RGB>(); \
						break; \
					case SPropertyElementData::Type::SRGB: \
						state. ## MEMBER = _property.getProperty<SPropertyElementData::Type::SRGB>(); \
						break; \
					case SPropertyElementData::Type::SPECTRUM: \
						state. ## MEMBER = _property.getProperty<SPropertyElementData::Type::SPECTRUM>(); \
						break; \
					default: \
						error = true; \
						break; \
				} \
			} \
			IRR_PSEUDO_IF_CONSTEXPR_END \
		}); \
	}

	auto setSamplingWeight = SET_PROPERTY_TEMPLATE(samplingWeight, SNamedPropertyElement::Type::FLOAT, Point,Area,Spot,Directional,Collimated,/*Sky,Sun,SunSky,*/EnvMap,Constant);
	auto setIntensity = SET_SPECTRUM(intensity, Point,Spot);
	auto setPosition = [&]() -> void {
		if (_property.type!=SNamedPropertyElement::Type::POINT || type!=Type::POINT)
		{
			error = true;
			return;
		}
		transform.matrix.setTranslation(_property.vvalue);
	};
	auto setRadiance = SET_SPECTRUM(radiance, Area,Constant);
	auto setCutoffAngle = SET_PROPERTY_TEMPLATE(cutoffAngle, SNamedPropertyElement::Type::FLOAT, Spot);
	auto setBeamWidth = SET_PROPERTY_TEMPLATE(beamWidth, SNamedPropertyElement::Type::FLOAT, Spot);
	auto setDirection = [&]() -> void {
		if (_property.type != SNamedPropertyElement::Type::VECTOR || type != Type::DIRECTIONAL)
		{
			error = true;
			return;
		}
		core::vectorSIMDf up(0.f);
		float maxDot = _property.vvalue[0];
		uint32_t index = 0u;
		for (auto i=1u; i<3u; i++)
		if (_property.vvalue[i] < maxDot)
		{
			maxDot = _property.vvalue[i];
			index = i;
		}
		up[index] = 1.f;
		// hope it works
		core::matrix3x4SIMD tmp;
		core::matrix3x4SIMD::buildCameraLookAtMatrixRH(core::vectorSIMDf(),-_property.vvalue,up).getInverse(tmp);
		transform.matrix = core::matrix4SIMD(tmp);
		_IRR_DEBUG_BREAK_IF(true); // no idea if matrix is correct
	};
	auto setPower = SET_SPECTRUM(power, Collimated);
	auto setFilename = [&]() -> void
	{ 
		dispatch([&](auto& state) -> void {
			using state_type = std::remove_reference<decltype(state)>::type;

			IRR_PSEUDO_IF_CONSTEXPR_BEGIN(std::is_same<state_type,EnvMap>::value)
			{
				envmap.filename = std::move(_property);
			}
			IRR_PSEUDO_IF_CONSTEXPR_END
		});
	};
	auto setScale = SET_PROPERTY_TEMPLATE(scale, SNamedPropertyElement::Type::FLOAT, EnvMap);
	auto setGamma = SET_PROPERTY_TEMPLATE(gamma, SNamedPropertyElement::Type::FLOAT, EnvMap);
	//auto setCache = SET_PROPERTY_TEMPLATE(cache, SNamedPropertyElement::Type::BOOLEAN, EnvMap);
#undef SET_SPECTRUM
#undef SET_PROPERTY_TEMPLATE
	const core::unordered_map<std::string, std::function<void()>, core::CaseInsensitiveHash, core::CaseInsensitiveEquals> SetPropertyMap =
	{
		{"samplingWeight",	setSamplingWeight},
		{"intensity",		setIntensity},
		{"position",		setPosition},
		{"radiance",		setRadiance},
		{"cutoffAngle",		setCutoffAngle},
		{"beamWidth",		setBeamWidth},
		{"direction",		setDirection},
		{"power",			setPower},/*
		{"turbidity",		setTurbidity},
		{"",				set},
		{"sunRadiusScale",	setSunRadiusScale},*/
		{"filename",		setFilename},
		{"scale",			setScale},
		{"gamma",			setGamma}//,
		//{"cache",			setCache},
	};

	auto found = SetPropertyMap.find(_property.name);
	if (found==SetPropertyMap.end())
	{
		_IRR_DEBUG_BREAK_IF(true);
		ParserLog::invalidXMLFileStructure("No Emitter can have such property set with name: " + _property.name);
		return false;
	}

	found->second();
	return !error;
}

bool CElementEmitter::onEndTag(asset::IAssetLoader::IAssetLoaderOverride* _override, CGlobalMitsubaMetadata* globalMetadata)
{
	// TODO: some more validation
	switch (type)
	{
		case Type::INVALID:
			ParserLog::invalidXMLFileStructure(getLogName() + ": type not specified");
			_IRR_DEBUG_BREAK_IF(true);
			return true;
			break;
		case Type::SPOT:
			if (std::isnan(spot.beamWidth))
				spot.beamWidth = spot.cutoffAngle * 0.75f;
		default:
			break;
	}

	switch (type)
	{
		case Type::AREA:
			break;
		default:
			globalMetadata->emitters.push_back(*this);
			break;
	}


	return true;
}

}
}
}