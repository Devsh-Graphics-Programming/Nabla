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

	auto setSamplingWeight = SET_PROPERTY_TEMPLATE(samplingWeight, SNamedPropertyElement::Type::FLOAT, Point,Area,Spot,Directional,Collimated,/*Sky,Sun,SunSky,*/EnvMap,Constant);
	auto setIntensity = [&]() -> void {
	};
	auto setPosition = [&]() -> void {
		if (_property.type!=SNamedPropertyElement::Type::POINT || type!=Type::POINT)
		{
			error = true;
			return;
		}
		transform.matrix.setTranslation(-_property.vvalue);
	};
	auto setDirection = [&]() -> void {
		if (_property.type != SNamedPropertyElement::Type::VECTOR || type != Type::DIRECTIONAL)
		{
			error = true;
			return;
		}
		transform.matrix = core::matrix4SIMD::buildCameraLookAtMatrixLH(-_property.vvalue);
	};
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
			// TODO: add to global emitters
			break;
	}


	return true;
}

}
}
}