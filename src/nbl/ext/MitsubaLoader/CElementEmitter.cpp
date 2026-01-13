// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/ext/MitsubaLoader/ParserUtil.h"
#include "nbl/ext/MitsubaLoader/CElementEmitter.h"

#include "nbl/ext/MitsubaLoader/ElementMacros.h"

#include "nbl/type_traits.h" // legacy stuff for `is_any_of`
#include <functional>

#include "nbl/builtin/hlsl/math/linalg/transform.hlsl"
#include "glm/gtc/matrix_transform.hpp"


namespace nbl::ext::MitsubaLoader
{

auto CElementEmitter::compAddPropertyMap() -> AddPropertyMap<CElementEmitter>
{
	using this_t = CElementEmitter;
	AddPropertyMap<CElementEmitter> retval;

	// funky transform setting
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_PROPERTY("position",POINT)
		{
			if (_this->type!=Type::POINT && _this->type!=Type::COLLIMATED)
				return false;
			for (auto r=0; r<3; r++)
				_this->transform.matrix[r][3] = _property.vvalue[r];
			return true;
		}
	});
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_PROPERTY("direction",VECTOR)
		{
			// for point lights direction gets concatenated with IES rotation
			if (_this->type!=Type::POINT && _this->type!=Type::DIRECTIONAL && _this->type!=Type::COLLIMATED)
				return false;
			hlsl::float32_t3 up = {0,0,0};
			{
				uint32_t index = 0u;
				{
					float maxDot = std::abs(_property.vvalue[0]);
					for (auto i=1u; i<3u; i++)
					{
						float thisAbs = std::abs(_property.vvalue[i]);
						if (thisAbs < maxDot)
						{
							maxDot = thisAbs;
							index = i;
						}
					}
				}
				up[index] = hlsl::sign(_property.vvalue[index]);
			}
			// TODO: check if correct
			const hlsl::float32_t3 target = (-_property.vvalue).xyz;
			// TODO: after the rm-core matrix PR we need to get rid of the tranpose (I transpose only because of GLM and HLSL mixup)
			const auto lookAtGLM = reinterpret_cast<const hlsl::float32_t4x4&>(glm::lookAtRH<float>({},target,up));
			const auto lookAt = hlsl::transpose(lookAtGLM);
			// turn lookat into a rotation matrix
			const auto rotation = hlsl::inverse<hlsl::float32_t3x3>(hlsl::float32_t3x3(lookAt));
			//_NBL_DEBUG_BREAK_IF(true); // no idea if matrix is correct, looks okay
			for (auto r=0; r<3; r++)
				_this->transform.matrix[r].xyz = rotation[r];
			return true;
		}
	});

	// base
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(samplingWeight,FLOAT,derived_from,SampledEmitter);

// spectrum setting
#define ADD_VARIANT_SPECTRUM_PROPERTY_CONSTRAINED(MEMBER,CONSTRAINT,...) { \
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_VARIANT_PROPERTY_CONSTRAINED(MEMBER,FLOAT,CONSTRAINT __VA_OPT__(,) __VA_ARGS__) \
		state. ## MEMBER.x = state. ## MEMBER.y = state. ## MEMBER.z = _property.getProperty<SPropertyElementData::Type::FLOAT>(); \
		success = true; \
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_VARIANT_PROPERTY_CONSTRAINED_END; \
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_VARIANT_PROPERTY_CONSTRAINED(MEMBER,RGB,CONSTRAINT __VA_OPT__(,) __VA_ARGS__) \
		state. ## MEMBER = _property.getProperty<SPropertyElementData::Type::RGB>(); \
		success = true; \
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_VARIANT_PROPERTY_CONSTRAINED_END; \
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_VARIANT_PROPERTY_CONSTRAINED(MEMBER,SRGB,CONSTRAINT __VA_OPT__(,) __VA_ARGS__) \
		state. ## MEMBER = _property.getProperty<SPropertyElementData::Type::SRGB>(); \
		success = true; \
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_VARIANT_PROPERTY_CONSTRAINED_END; \
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_VARIANT_PROPERTY_CONSTRAINED(MEMBER,SPECTRUM,CONSTRAINT __VA_OPT__(,) __VA_ARGS__) \
		state. ## MEMBER = _property.getProperty<SPropertyElementData::Type::SPECTRUM>(); \
		success = true; \
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_VARIANT_PROPERTY_CONSTRAINED_END; \
}

	// delta
	ADD_VARIANT_SPECTRUM_PROPERTY_CONSTRAINED(intensity,derived_from,DeltaDistributionEmitter);
	// point covered by delta

	// non zero solid angle
	ADD_VARIANT_SPECTRUM_PROPERTY_CONSTRAINED(radiance,derived_from,SolidAngleEmitter);
	// area covered by solid angle

	// directional
	ADD_VARIANT_SPECTRUM_PROPERTY_CONSTRAINED(irradiance,std::is_same,Directional);

	// collimated
	ADD_VARIANT_SPECTRUM_PROPERTY_CONSTRAINED(power,std::is_same,Collimated);

#undef ADD_VARIANT_SPECTRUM_PROPERTY_CONSTRAINED

	// environment map
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_PROPERTY_CONSTRAINED("filename",STRING,std::is_same,EnvMap)
		{
			setLimitedString("filename",_this->envmap.filename,_property,logger); return true;
		}
	);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(scale,FLOAT,std::is_same,EnvMap);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(gamma,FLOAT,std::is_same,EnvMap);

#undef ADD_SPECTRUM
	return retval;
}

bool CElementEmitter::processChildData(IElement* _child, const std::string& name, system::logger_opt_ptr logger)
{
	if (!_child)
		return true;

	switch (_child->getType())
	{
		case IElement::Type::TRANSFORM:
			{
				auto tform = static_cast<CElementTransform*>(_child);
				if (name!="toWorld")
				{
					logger.log("The <transform> nested inside <emitter> needs to be named \"toWorld\"",system::ILogger::ELL_ERROR);
					return false;
				}
				//toWorldType = IElement::Type::TRANSFORM;
				switch (type)
				{
					case Type::POINT:
						[[fallthrough]];
					case Type::DIRECTIONAL:
						[[fallthrough]];
					case Type::COLLIMATED:
						[[fallthrough]];
					case Type::AREA:
						[[fallthrough]];
						/*
					case Type::SKY:
						[[fallthrough]];
					case Type::SUN:
						[[fallthrough]];
					case Type::SUNSKY:
						[[fallthrough]];*/
					case Type::ENVMAP:
						transform = *tform;
						return true;
					default:
						logger.log("<emitter type=\"%d\"> does not support <transform name=\"toWorld\">",system::ILogger::ELL_ERROR,type);
						return false;
				}
			}
			break;/*
		case IElement::Type::ANIMATION:
			auto anim = static_cast<CElementAnimation>(_child);
			if (anim->name!="toWorld")
				return false;
			toWorlType = IElement::Type::ANIMATION;
			animation = anim;
			return true;
			break;*/
		case IElement::Type::EMISSION_PROFILE:
			if (type!=Type::AREA && type!=Type::POINT)
			{
				logger.log("<emitter type=\"%d\"> does not support nested emission profiles, only Point and Area lights do",system::ILogger::ELL_ERROR,type);
				return false;
			}
			area.emissionProfile = static_cast<CElementEmissionProfile*>(_child);
			return true;
		default:
			break;
	}
	logger.log("<emitter type=\"%d\"> does not support nested <%s> elements",system::ILogger::ELL_ERROR,type,_child->getLogName());
	return false;
}

bool CElementEmitter::onEndTag(CMitsubaMetadata* globalMetadata, system::logger_opt_ptr logger)
{
	// TODO: some more validation
	switch (type)
	{
		case Type::INVALID:
			logger.log("<emitter>'s type not specified!",system::ILogger::ELL_ERROR);
			_NBL_DEBUG_BREAK_IF(true);
			return true;
			break;
		case Type::AREA:
			break;
		default:
			// TODO: slap into the scene instead!
//			globalMetadata->m_global.m_emitters.push_back(*this);
			break;
	}

	return true;
}

}