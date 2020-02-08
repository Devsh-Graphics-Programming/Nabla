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
CElementFactory::return_type CElementFactory::createElement<CElementSensor>(const char** _atts, ParserManager* _util)
{
	const char* type;
	const char* id;
	std::string name;
	if (!IElement::getTypeIDAndNameStrings(type, id, name, _atts))
		return CElementFactory::return_type(nullptr,"");

	static const core::unordered_map<std::string, CElementSensor::Type, core::CaseInsensitiveHash, core::CaseInsensitiveEquals> StringToType =
	{
		{"perspective",			CElementSensor::Type::PERSPECTIVE},
		{"thinlens",			CElementSensor::Type::THINLENS},
		{"orthographic",		CElementSensor::Type::ORTHOGRAPHIC},
		{"telecentric",			CElementSensor::Type::TELECENTRIC},
		{"spherical",			CElementSensor::Type::SPHERICAL},
		{"irradiancemeter",		CElementSensor::Type::IRRADIANCEMETER},
		{"radiancemeter",		CElementSensor::Type::RADIANCEMETER},
		{"fluencemeter",		CElementSensor::Type::FLUENCEMETER}/*,
		{"perspective_rdist",	CElementSensor::PERSPECTIVE_RDIST}*/
	};

	auto found = StringToType.find(type);
	if (found==StringToType.end())
	{
		ParserLog::invalidXMLFileStructure("unknown type");
		_IRR_DEBUG_BREAK_IF(false);
		return CElementFactory::return_type(nullptr, "");
	}

	CElementSensor* obj = _util->objects.construct<CElementSensor>(id);
	if (!obj)
		return CElementFactory::return_type(nullptr, "");

	obj->type = found->second;
	// defaults
	switch (obj->type)
	{
		case CElementSensor::Type::PERSPECTIVE:
			obj->perspective = CElementSensor::PerspectivePinhole();
			break;
		case CElementSensor::Type::THINLENS:
			obj->thinlens = CElementSensor::PerspectiveThinLens();
			break;
		case CElementSensor::Type::ORTHOGRAPHIC:
			obj->orthographic = CElementSensor::Orthographic();
			break;
		case CElementSensor::Type::TELECENTRIC:
			obj->telecentric = CElementSensor::TelecentricLens();
			break;
		case CElementSensor::Type::SPHERICAL:
			obj->spherical = CElementSensor::SphericalCamera();
			break;
		case CElementSensor::Type::IRRADIANCEMETER:
			obj->irradiancemeter = CElementSensor::IrradianceMeter();
			break;
		case CElementSensor::Type::RADIANCEMETER:
			obj->radiancemeter = CElementSensor::RadianceMeter();
			break;
		case CElementSensor::Type::FLUENCEMETER:
			obj->fluencemeter = CElementSensor::FluenceMeter();
			break;
		default:
			break;
	}
	return CElementFactory::return_type(obj, std::move(name));
}

bool CElementSensor::addProperty(SNamedPropertyElement&& _property)
{
	bool error = false;
	auto dispatch = [&](auto func) -> void
	{
		switch (type)
		{
			case CElementSensor::Type::PERSPECTIVE:
				func(perspective);
				break;
			case CElementSensor::Type::THINLENS:
				func(thinlens);
				break;
			case CElementSensor::Type::ORTHOGRAPHIC:
				func(orthographic);
				break;
			case CElementSensor::Type::TELECENTRIC:
				func(telecentric);
				break;
			case CElementSensor::Type::SPHERICAL:
				func(spherical);
				break;
			case CElementSensor::Type::IRRADIANCEMETER:
				func(irradiancemeter);
				break;
			case CElementSensor::Type::RADIANCEMETER:
				func(radiancemeter);
				break;
			case CElementSensor::Type::FLUENCEMETER:
				func(fluencemeter);
				break;
			default:
				error = true;
				break;
		}
	};

#define SET_PROPERTY_TEMPLATE(MEMBER,PROPERTY_TYPE,BASE)		[&]() -> void { \
		dispatch([&](auto& state) -> void { \
			IRR_PSEUDO_IF_CONSTEXPR_BEGIN(std::is_base_of<BASE,std::remove_reference<decltype(state)>::type >::value) \
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

	auto setFov = SET_PROPERTY_TEMPLATE(fov,SNamedPropertyElement::Type::FLOAT,PerspectivePinhole);
	auto setFovAxis = [&]() -> void
	{
		dispatch([&](auto& state) -> void
		{
			using state_type = std::remove_reference<decltype(state)>::type;
			IRR_PSEUDO_IF_CONSTEXPR_BEGIN(std::is_base_of<PerspectivePinhole,state_type>::value)
			{
				if (_property.type != SNamedPropertyElement::Type::STRING)
				{
					error = true;
					return;
				}
				static const core::unordered_map<std::string,PerspectivePinhole::FOVAxis,core::CaseInsensitiveHash,core::CaseInsensitiveEquals> StringToType =
				{
					{"x",		PerspectivePinhole::FOVAxis::X},
					{"y",		PerspectivePinhole::FOVAxis::Y},
					{"diagonal",PerspectivePinhole::FOVAxis::DIAGONAL},
					{"smaller",	PerspectivePinhole::FOVAxis::SMALLER},
					{"larger",	PerspectivePinhole::FOVAxis::LARGER}
				};
				auto found = StringToType.find(_property.svalue);
				if (found!=StringToType.end())
					state.fovAxis = found->second;
				else
					state.fovAxis = PerspectivePinhole::FOVAxis::INVALID;
			}
			IRR_PSEUDO_IF_CONSTEXPR_END
		});
	};
	auto setShutterOpen		= SET_PROPERTY_TEMPLATE(shutterOpen,SNamedPropertyElement::Type::FLOAT,ShutterSensor);
	auto setShutterClose	= SET_PROPERTY_TEMPLATE(shutterClose,SNamedPropertyElement::Type::FLOAT,ShutterSensor);
	auto setNearClip		= SET_PROPERTY_TEMPLATE(nearClip,SNamedPropertyElement::Type::FLOAT,CameraBase);
	auto setFarClip			= SET_PROPERTY_TEMPLATE(farClip,SNamedPropertyElement::Type::FLOAT,CameraBase);
	auto setFocusDistance	= SET_PROPERTY_TEMPLATE(focusDistance,SNamedPropertyElement::Type::FLOAT,DepthOfFieldBase);
	auto setApertureRadius	= SET_PROPERTY_TEMPLATE(apertureRadius,SNamedPropertyElement::Type::FLOAT,DepthOfFieldBase);
	//auto setKc			= SET_PROPERTY_TEMPLATE(apertureRadius,SNamedPropertyElement::Type::STRING,PerspectivePinholeRadialDistortion);

	const core::unordered_map<std::string, std::function<void()>, core::CaseInsensitiveHash, core::CaseInsensitiveEquals> SetPropertyMap =
	{
		//{"focalLength",	noIdeaHowToProcessValue},
		{"fov",				setFov},
		{"fovAxis",			setFovAxis},
		{"shutterOpen",		setShutterOpen},
		{"shuttterClose",	setShutterClose},
		{"nearClip",		setNearClip},
		{"farClip",			setFarClip},
		{"focusDistance",	setFocusDistance},
		{"apertureRadius",	setApertureRadius}/*,
		{"kc",				setKc}*/
	};
	

	auto found = SetPropertyMap.find(_property.name);
	if (found==SetPropertyMap.end())
	{
		_IRR_DEBUG_BREAK_IF(true);
		ParserLog::invalidXMLFileStructure("No Integrator can have such property set with name: "+_property.name);
		return false;
	}

	found->second();
	return !error;
}

bool CElementSensor::onEndTag(asset::IAssetLoader::IAssetLoaderOverride* _override, CGlobalMitsubaMetadata* globalMetadata)
{
	if (type == Type::INVALID)
	{
		ParserLog::invalidXMLFileStructure(getLogName() + ": type not specified");
		_IRR_DEBUG_BREAK_IF(true);
		return true;
	}

	// TODO: some validation

	// add to global metadata
	globalMetadata->sensors.push_back(*this);

	return true;
}

}
}
}