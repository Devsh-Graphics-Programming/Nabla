// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/ext/MitsubaLoader/ParserUtil.h"
#include "nbl/ext/MitsubaLoader/CElementShape.h"

#include "nbl/ext/MitsubaLoader/ElementMacros.h"

#include "nbl/type_traits.h" // legacy stuff for `is_any_of`
#include <functional>

namespace nbl::ext::MitsubaLoader
{
	
auto CElementShape::compAddPropertyMap() -> AddPropertyMap<CElementShape>
{
	using this_t = CElementShape;
	AddPropertyMap<CElementShape> retval;


	// base
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(flipNormals,BOOLEAN,derived_from,Base);
	// cube has nothing

	// sphere
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(center,POINT,std::is_same,Sphere);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(radius,FLOAT,is_any_of,Sphere,Cylinder/*,Hair*/);

	// cylinder
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(p0,POINT,std::is_same,Cylinder);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(p1,POINT,std::is_same,Cylinder);
	// COMMON: radius

	// LoadedFromFileBase
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_PROPERTY_CONSTRAINED("filename",STRING,derived_from,LoadedFromFileBase)
		{
			setLimitedString("filename",_this->serialized.filename,std::move(_property),logger); return true;
		}
	);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(faceNormals,BOOLEAN,derived_from,LoadedFromFileBase);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(maxSmoothAngle,FLOAT,derived_from,LoadedFromFileBase);

	// Obj
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(flipTexCoords,BOOLEAN,std::is_same,Obj);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(collapse,BOOLEAN,std::is_same,Obj);

	// Ply
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(srgb,BOOLEAN,std::is_same,Ply);

	// Serialized
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(shapeIndex,INTEGER,std::is_same,Serialized);

	return retval;
}

bool CElementShape::processChildData(IElement* _child, const std::string& name, system::logger_opt_ptr logger)
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
					logger.log("The <transform> nested inside <shape> needs to be named \"toWorld\"",system::ILogger::ELL_ERROR);
					return false;
				}
				//toWorldType = IElement::Type::TRANSFORM;
				transform = *tform;
				return true;
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
		case IElement::Type::SHAPE:
			{
				auto child = static_cast<CElementShape*>(_child);
				switch (type)
				{
					case Type::SHAPEGROUP:
						if (child->type == Type::INVALID || child->type == Type::SHAPEGROUP)
						{
							logger.log("<shape type=\"shapegroup\"> cannot be nested inside each other or have INVALID shapes nested inside.",system::ILogger::ELL_ERROR);
							return false;
						}
						if (shapegroup.childCount==ShapeGroup::MaxChildCount)
						{
							logger.log("The <shape type=\"shapegroup\">'s MaxChildCount of %d exceeded!",system::ILogger::ELL_ERROR,ShapeGroup::MaxChildCount);
							return false;
						}
						shapegroup.children[shapegroup.childCount++] = child;
						return true;
					case Type::INSTANCE:
						if (child->type!=Type::SHAPEGROUP)
						{
							logger.log("Only <shape type=\"shapegroup\"> can be nested inside <shape type=\"instance\">",system::ILogger::ELL_ERROR);
							return false;
						}
						if (instance.parent)
							logger.log("<shape type=\"instance\"> 's parent already set to %s, overriding",system::ILogger::ELL_WARNING,instance.parent->id.c_str());
						instance.parent = child; // yeah I know its messed up, but its the XML child, not the Abstract Syntax Tree (or Scene Tree) parent
						return true;
					default:
						logger.log("Only <shape type=\"shapegroup\"> and <shape type=\"instance\"> support nesting other <shape>s inside",system::ILogger::ELL_ERROR);
						return false;
				}
			}
			break;
		case IElement::Type::BSDF:
			if (bsdf)
				logger.log("<shape>'s BSDF already set to %s, overriding",system::ILogger::ELL_WARNING,bsdf->id.c_str());
			bsdf = static_cast<CElementBSDF*>(_child);
			if (bsdf->type != CElementBSDF::Type::INVALID)
				return true;
			break;
		case IElement::Type::EMITTER:
			if (emitter)
				logger.log("<shape>'s Emitter already set to %s, overriding",system::ILogger::ELL_WARNING,emitter->id.c_str());
			emitter = static_cast<CElementEmitter*>(_child);
			if (emitter->type != CElementEmitter::Type::INVALID)
				return true;
			break;
		default:
			break;
	}
	logger.log("Invalid or unsupported child with ID %s and Name %s nested inside of <shape>",system::ILogger::ELL_ERROR,_child->id.c_str(),name.c_str());
	return false;
}

bool CElementShape::onEndTag(CMitsubaMetadata* globalMetadata, system::logger_opt_ptr logger)
{
	NBL_EXT_MITSUBA_LOADER_ELEMENT_INVALID_TYPE_CHECK(true);

	// TODO: some validation

	return true;
}

}