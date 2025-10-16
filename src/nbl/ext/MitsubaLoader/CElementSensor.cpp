// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


#include "nbl/ext/MitsubaLoader/CElementSensor.h"
#include "nbl/ext/MitsubaLoader/ParserUtil.h"
#include "nbl/ext/MitsubaLoader/ElementMacros.h"

#include <functional>

template<typename D, typename B>
struct derived_from : std::is_base_of<B,D> {};

namespace nbl::ext::MitsubaLoader
{

auto CElementSensor::compAddPropertyMap() -> AddPropertyMap<CElementSensor>
{
	using this_t = CElementSensor;
	AddPropertyMap<CElementSensor> retval;

//	auto setUp = SET_PROPERTY_TEMPLATE(up, SNamedPropertyElement::Type::VECTOR, ShutterSensor);
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_PROPERTY(VECTOR,"clipPlane")
		{
			if (_property.getVectorDimension()!=4)
			{
				return false;
			}
			constexpr std::string_view Name = "clipPlane";
			const std::string_view sv(_property.name);
			if (sv.length()!=Name.length()+1 || sv.find(Name)!=0)
			{
				return false;
			}
			const auto index = std::atoi(sv.data()+Name.length());
			if (index>MaxClipPlanes)
			{
				return false;
			}
			// everyone inherits from this
			_this->perspective.clipPlanes[index] = _property.vvalue;
			return true;
		}
	});

//	auto setShiftX = SET_PROPERTY_TEMPLATE(shiftX, SNamedPropertyElement::Type::FLOAT, PerspectivePinhole);
//	auto setShiftY = SET_PROPERTY_TEMPLATE(shiftY, SNamedPropertyElement::Type::FLOAT, PerspectivePinhole);
//	auto setFov = SET_PROPERTY_TEMPLATE(fov, SNamedPropertyElement::Type::FLOAT, PerspectivePinhole);
	retval.template registerCallback<derived_from,PerspectivePinhole>(SNamedPropertyElement::Type::STRING,"fovAxis",[](CElementSensor* _this, SNamedPropertyElement&& _property, const system::logger_opt_ptr logger)->bool
		{
			auto& state = _this->perspective;
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
			return true;
		}
	);

	return retval;
}

bool CElementSensor::addProperty(SNamedPropertyElement&& _property, system::logger_opt_ptr logger)
{
	if (type >= Type::INVALID)
		return false;
	bool error = false;

#if 0

#define SET_PROPERTY_TEMPLATE(MEMBER,PROPERTY_TYPE,BASE)		[&]() -> void { \
		dispatch([&](auto& state) -> void { \
			if constexpr (std::is_base_of<BASE,std::remove_reference<decltype(state)>::type >::value) \
			{ \
				if (_property.type!=PROPERTY_TYPE) { \
					error = true; \
					return; \
				} \
				state. ## MEMBER = _property.getProperty<PROPERTY_TYPE>(); \
			} \
		}); \
	}
	
	auto setUp = SET_PROPERTY_TEMPLATE(up,SNamedPropertyElement::Type::VECTOR,ShutterSensor);
	auto setShutterOpen		= SET_PROPERTY_TEMPLATE(shutterOpen,SNamedPropertyElement::Type::FLOAT,ShutterSensor);
	auto setShutterClose	= SET_PROPERTY_TEMPLATE(shutterClose,SNamedPropertyElement::Type::FLOAT,ShutterSensor);
	auto setMoveSpeed		= SET_PROPERTY_TEMPLATE(moveSpeed,SNamedPropertyElement::Type::FLOAT,ShutterSensor);
	auto setZoomSpeed		= SET_PROPERTY_TEMPLATE(zoomSpeed,SNamedPropertyElement::Type::FLOAT,ShutterSensor);
	auto setRotateSpeed		= SET_PROPERTY_TEMPLATE(rotateSpeed,SNamedPropertyElement::Type::FLOAT,ShutterSensor);
	auto setNearClip		= SET_PROPERTY_TEMPLATE(nearClip,SNamedPropertyElement::Type::FLOAT,CameraBase);
	auto setFarClip			= SET_PROPERTY_TEMPLATE(farClip,SNamedPropertyElement::Type::FLOAT,CameraBase);
	auto setFocusDistance	= SET_PROPERTY_TEMPLATE(focusDistance,SNamedPropertyElement::Type::FLOAT,DepthOfFieldBase);
	auto setApertureRadius	= SET_PROPERTY_TEMPLATE(apertureRadius,SNamedPropertyElement::Type::FLOAT,DepthOfFieldBase);
	//auto setKc			= SET_PROPERTY_TEMPLATE(apertureRadius,SNamedPropertyElement::Type::STRING,PerspectivePinholeRadialDistortion);

	const core::unordered_map<std::string, std::function<void()>, core::CaseInsensitiveHash, core::CaseInsensitiveEquals> SetPropertyMap =
	{
		//{"focalLength",	noIdeaHowToProcessValue},
		{"up",				setUp},
		{"clipPlane0",		setClipPlane},
		{"clipPlane1",		setClipPlane},
		{"clipPlane2",		setClipPlane},
		{"clipPlane3",		setClipPlane},
		{"clipPlane4",		setClipPlane},
		{"clipPlane5",		setClipPlane},
		// UPDATE WHENEVER `MaxClipPlanes` changes!
		{"shiftX",			setShiftX},
		{"shiftY",			setShiftY},
		{"fov",				setFov},
		{"fovAxis",			setFovAxis},
		{"shutterOpen",		setShutterOpen},
		{"shuttterClose",	setShutterClose},
		{"moveSpeed",		setMoveSpeed},
		{"zoomSpeed",		setZoomSpeed},
		{"rotateSpeed",		setRotateSpeed},
		{"nearClip",		setNearClip},
		{"farClip",			setFarClip},
		{"focusDistance",	setFocusDistance},
		{"apertureRadius",	setApertureRadius}
//,		{"kc",				setKc}
	};
	

	auto found = SetPropertyMap.find(_property.name);
	if (found==SetPropertyMap.end())
	{
		_NBL_DEBUG_BREAK_IF(true);
		ParserLog::invalidXMLFileStructure("No Integrator can have such property set with name: "+_property.name);
		return false;
	}

	found->second();
	return !error;
#endif
	assert(false);
	return false;
}

bool CElementSensor::onEndTag(CMitsubaMetadata* meta, system::logger_opt_ptr logger)
{
	NBL_EXT_MITSUBA_LOADER_ELEMENT_INVALID_TYPE_CHECK(true);

	// TODO: some validation

	// add to global metadata
	meta->m_global.m_sensors.push_back(*this);

	return true;
}

}