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
CElementFactory::return_type CElementFactory::createElement<CElementShape>(const char** _atts, ParserManager* _util)
{
	const char* type;
	const char* id;
	std::string name;
	if (!IElement::getTypeIDAndNameStrings(type, id, name, _atts))
		return CElementFactory::return_type(nullptr,"");

	static const core::unordered_map<std::string, CElementShape::Type, core::CaseInsensitiveHash, core::CaseInsensitiveEquals> StringToType =
	{
		{"cube",		CElementShape::Type::CUBE},
		{"sphere",		CElementShape::Type::SPHERE},
		{"cylinder",	CElementShape::Type::CYLINDER},
		{"rectangle",	CElementShape::Type::RECTANGLE},
		{"disk",		CElementShape::Type::DISK},
		{"obj",			CElementShape::Type::OBJ},
		{"ply",			CElementShape::Type::PLY},
		{"serialized",	CElementShape::Type::SERIALIZED},
		{"shapegroup",	CElementShape::Type::SHAPEGROUP},
		{"instance",	CElementShape::Type::INSTANCE}/*,
		{"hair",		CElementShape::Type::HAIR},
		{"heightfield",	CElementShape::Type::HEIGHTFIELD}*/
	};

	auto found = StringToType.find(type);
	if (found==StringToType.end())
	{
		ParserLog::invalidXMLFileStructure("unknown type");
		_IRR_DEBUG_BREAK_IF(false);
		return CElementFactory::return_type(nullptr, "");
	}

	CElementShape* obj = _util->objects.construct<CElementShape>(id);
	if (!obj)
		return CElementFactory::return_type(nullptr, "");

	obj->type = found->second;
	// defaults
	switch (obj->type)
	{
		case CElementShape::Type::CUBE:
			obj->cube = CElementShape::Base();
			break;
		case CElementShape::Type::SPHERE:
			obj->sphere = CElementShape::Sphere();
			break;
		case CElementShape::Type::CYLINDER:
			obj->cylinder = CElementShape::Cylinder();
			break;
		case CElementShape::Type::RECTANGLE:
			obj->rectangle = CElementShape::Base();
			break;
		case CElementShape::Type::DISK:
			obj->disk = CElementShape::Base();
			break;
		case CElementShape::Type::OBJ:
			obj->obj = CElementShape::Obj();
			break;
		case CElementShape::Type::PLY:
			obj->ply = CElementShape::Ply();
			break;
		case CElementShape::Type::SERIALIZED:
			obj->serialized = CElementShape::Serialized();
			break;
		case CElementShape::Type::SHAPEGROUP:
			obj->shapegroup = CElementShape::ShapeGroup();
			break;
		case CElementShape::Type::INSTANCE:
			obj->instance = CElementShape::Instance();
			break;
		default:
			break;
	}
	return CElementFactory::return_type(obj, std::move(name));
}

bool CElementShape::addProperty(SNamedPropertyElement&& _property)
{
	bool error = false;
	auto dispatch = [&](auto func) -> void
	{
		switch (type)
		{
			case CElementShape::Type::CUBE:
				func(cube);
				break;
			case CElementShape::Type::SPHERE:
				func(sphere);
				break;
			case CElementShape::Type::CYLINDER:
				func(cylinder);
				break;
			case CElementShape::Type::RECTANGLE:
				func(rectangle);
				break;
			case CElementShape::Type::DISK:
				func(disk);
				break;
			case CElementShape::Type::OBJ:
				func(obj);
				break;
			case CElementShape::Type::PLY:
				func(ply);
				break;
			case CElementShape::Type::SERIALIZED:
				func(serialized);
				break;
			case CElementShape::Type::SHAPEGROUP:
				func(shapegroup);
				break;
			case CElementShape::Type::INSTANCE:
				func(instance);
				break;/*
			case CElementShape::Type::HAIR:
				func(hair);
				break;
			case CElementShape::Type::HEIGHTFIELD:
				func(heightfield);
				break;*/
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

	auto setFlipNormals = SET_PROPERTY_TEMPLATE(flipNormals,SNamedPropertyElement::Type::BOOLEAN, Base,Sphere,Cylinder,Obj,Ply,Serialized/*,Heightfield*/);
	auto setCenter = SET_PROPERTY_TEMPLATE(center,SNamedPropertyElement::Type::POINT, Sphere);
	auto setRadius = SET_PROPERTY_TEMPLATE(radius,SNamedPropertyElement::Type::FLOAT, Sphere,Cylinder/*,Hair*/);
	auto setP0 = SET_PROPERTY_TEMPLATE(p0,SNamedPropertyElement::Type::POINT, Cylinder);
	auto setP1 = SET_PROPERTY_TEMPLATE(p1,SNamedPropertyElement::Type::POINT, Cylinder);
	auto setFilename = [&]() -> void
	{
		dispatch([&](auto& state) -> void {
			using state_type = std::remove_reference<decltype(state)>::type;

			IRR_PSEUDO_IF_CONSTEXPR_BEGIN(is_any_of<state_type, Obj,Ply,Serialized/*,Hair,Heightfield*/>::value)
			{
				state.filename = std::move(_property);
			}
			IRR_PSEUDO_IF_CONSTEXPR_END
		});
	};
	auto setFaceNormals	= SET_PROPERTY_TEMPLATE(faceNormals,SNamedPropertyElement::Type::BOOLEAN, Obj,Ply,Serialized);
	auto setMaxSmoothAngle	= SET_PROPERTY_TEMPLATE(maxSmoothAngle,SNamedPropertyElement::Type::FLOAT, Obj,Ply,Serialized);
	auto setFlipTexCoords = SET_PROPERTY_TEMPLATE(flipTexCoords,SNamedPropertyElement::Type::BOOLEAN, Obj);
	auto setCollapse = SET_PROPERTY_TEMPLATE(collapse,SNamedPropertyElement::Type::BOOLEAN, Obj);
	auto setSRGB = SET_PROPERTY_TEMPLATE(srgb,SNamedPropertyElement::Type::BOOLEAN, Ply);
	auto setShapeIndex = SET_PROPERTY_TEMPLATE(shapeIndex,SNamedPropertyElement::Type::INTEGER, Serialized);

	const core::unordered_map<std::string, std::function<void()>, core::CaseInsensitiveHash, core::CaseInsensitiveEquals> SetPropertyMap =
	{
		{"flipNormals",		setFlipNormals},
		{"center",			setCenter},
		{"radius",			setRadius},
		{"p0",				setP0},
		{"p1",				setP1},
		{"filename",		setFilename},
		{"faceNormals",		setFaceNormals},
		{"maxSmoothAngle",	setMaxSmoothAngle},
		{"flipTexCoords",	setFlipTexCoords},
		{"collapse",		setCollapse},
		{"srgb",			setSRGB},
		{"shapeIndex",		setShapeIndex}/*,
		{"",				set}*/
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

bool CElementShape::processChildData(IElement* _child, const std::string& name)
{
	if (!_child)
		return true;
	switch (_child->getType())
	{
		case IElement::Type::TRANSFORM:
			{
				auto tform = static_cast<CElementTransform*>(_child);
				if (name!="toWorld")
					return false;
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
						if (child->type==Type::INVALID || child->type==Type::SHAPEGROUP)
							return false;
						if (shapegroup.childCount == ShapeGroup::MaxChildCount)
						{
							ParserLog::invalidXMLFileStructure("Maximum shape-group children exceeded.");
							return false;
						}
						shapegroup.children[shapegroup.childCount++] = child;
						return true;
						break;
					case Type::INSTANCE:
						if (child->type != Type::SHAPEGROUP)
							return false;
						instance.parent = child; // yeah I kknow its fucked up, but its the XML child, but Abstract Syntax Tree (or Scene Tree) parent
						return true;
						break;
					default:
						break;
				}
			}
			break;
		case IElement::Type::BSDF:
			bsdf = static_cast<CElementBSDF*>(_child);
			if (bsdf->type != CElementBSDF::Type::INVALID)
				return true;
			break;
		case IElement::Type::EMITTER:
			emitter = static_cast<CElementEmitter*>(_child);
			if (emitter->type != CElementEmitter::Type::INVALID)
				return true;
			break;
		default:
			break;
	}
	return false;
}

bool CElementShape::onEndTag(asset::IAssetLoader::IAssetLoaderOverride* _override, CGlobalMitsubaMetadata* globalMetadata)
{
	if (type == Type::INVALID)
	{
		ParserLog::invalidXMLFileStructure(getLogName() + ": type not specified");
		_IRR_DEBUG_BREAK_IF(true);
		return true;
	}

	// TODO: some validation

	return true;
}

}
}
}