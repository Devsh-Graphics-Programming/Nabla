// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


#include "nbl/ext/MitsubaLoader/CElementSensor.h"
#include "nbl/ext/MitsubaLoader/ParserUtil.h"
#include "nbl/ext/MitsubaLoader/ElementMacros.h"

#include <functional>


namespace nbl::ext::MitsubaLoader
{

auto CElementSensor::compAddPropertyMap() -> AddPropertyMap<CElementSensor>
{
	using this_t = CElementSensor;
	AddPropertyMap<CElementSensor> retval;

	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(up,VECTOR,derived_from,ShutterSensor);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(shiftX,FLOAT,derived_from,PerspectivePinhole);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(shiftY,FLOAT,derived_from,PerspectivePinhole);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(fov,FLOAT,derived_from,PerspectivePinhole);
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_PROPERTY_CONSTRAINED("fovAxis",STRING,derived_from,PerspectivePinhole)
		{
			auto& state = _this->perspective;
			// TODO: check if this gives problem with delay loads
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

	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(up,VECTOR,derived_from,ShutterSensor);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(shutterOpen,FLOAT,derived_from,ShutterSensor);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(shutterClose,FLOAT,derived_from,ShutterSensor);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(moveSpeed,FLOAT,derived_from,ShutterSensor);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(zoomSpeed,FLOAT,derived_from,ShutterSensor);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(rotateSpeed,FLOAT,derived_from,ShutterSensor);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(nearClip,FLOAT,derived_from,CameraBase);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(farClip,FLOAT,derived_from,CameraBase);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(focusDistance,FLOAT,derived_from,DepthOfFieldBase);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(apertureRadius,FLOAT,derived_from,DepthOfFieldBase);

	// special
	auto setClipPlane = [](this_t* _this, SNamedPropertyElement&& _property, const system::logger_opt_ptr logger)->bool
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
	};
	for (auto i=0; i<MaxClipPlanes; i++)
		retval.registerCallback(SNamedPropertyElement::Type::VECTOR,"clipPlane"+std::to_string(i),{.func=setClipPlane});

	// TODOs:
	//auto setKc			= SET_PROPERTY_TEMPLATE(apertureRadius,SNamedPropertyElement::Type::STRING,PerspectivePinholeRadialDistortion);
	//{"focalLength",	noIdeaHowToProcessValue},

	return retval;
}

bool CElementSensor::addProperty(SNamedPropertyElement&& _property, system::logger_opt_ptr logger)
{	
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