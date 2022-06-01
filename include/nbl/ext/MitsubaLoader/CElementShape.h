// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __C_ELEMENT_SHAPE_H_INCLUDED__
#define __C_ELEMENT_SHAPE_H_INCLUDED__

#include "nbl/ext/MitsubaLoader/IElement.h"
#include "nbl/ext/MitsubaLoader/CElementTransform.h"
#include "nbl/ext/MitsubaLoader/CElementBSDF.h"
#include "nbl/ext/MitsubaLoader/CElementEmitter.h"


namespace nbl
{
namespace ext
{
namespace MitsubaLoader
{


class NBL_API CElementShape : public IElement
{
	public:
		enum Type
		{
			INVALID,
			CUBE,
			SPHERE,
			CYLINDER,
			RECTANGLE,
			DISK,
			OBJ,
			PLY,
			SERIALIZED,
			SHAPEGROUP,
			INSTANCE//,
			//HAIR,
			//HEIGHTFIELD
		};
	struct Base
	{
		bool flipNormals = false;
	};
		struct Sphere : Base
		{
			core::vectorSIMDf center = core::vectorSIMDf(0,0,0);
			float radius = 1.f;
		};
		struct Cylinder : Base
		{
			core::vectorSIMDf p0 = core::vectorSIMDf(0,0,0);
			core::vectorSIMDf p1 = core::vectorSIMDf(0,0,1);
			float radius = 1.f;
		};
	struct LoadedFromFileBase : Base
	{
		SPropertyElementData	filename;
		//! Use face normals (any per-vertex normals will be discarded)
		bool					faceNormals = false;
		float					maxSmoothAngle = NAN;
	};
		struct Obj : LoadedFromFileBase
		{
			bool	flipTexCoords = true;
			bool	collapse = false;
		};
		struct Ply : LoadedFromFileBase
		{
			bool	flipNormals = false;
			bool	srgb = true;
		};
		struct Serialized : LoadedFromFileBase
		{
			int32_t	shapeIndex;
			bool	flipNormals;
		};
		struct ShapeGroup
		{
			_NBL_STATIC_INLINE_CONSTEXPR size_t MaxChildCount = 128u;
			size_t childCount = 0u;
			CElementShape* children[MaxChildCount] = { nullptr };
		};
		struct Instance
		{
			CElementShape* parent = nullptr;
		};/*
		struct Hair : Base
		{
			SPropertyElementData	filename;
			float					raidus = 0.025f;
			float					angleThreshold = 1.f;
			float					reduction = 0.f;
		};
		struct HeightField
		{
			SPropertyElementData filename;
			boolean shadingNormals;
			boolean flipNormals;
			int32_t width;
			int32_t height;
			float	scale;
			CElementTexture* texture;
		};*/

		CElementShape(const char* id) : IElement(id), type(Type::INVALID), /*toWorldType(IElement::Type::TRANSFORM),*/ transform(), bsdf(nullptr), emitter(nullptr)
		{
		}
		CElementShape(const CElementShape& other) : IElement(""), transform(), bsdf(nullptr), emitter(nullptr)
		{
			operator=(other);
		}
		CElementShape(CElementShape&& other) : IElement(""), transform(), bsdf(nullptr), emitter(nullptr)
		{
			operator=(std::move(other));
		}
		virtual ~CElementShape()
		{
		}

		inline CElementShape& operator=(const CElementShape& other)
		{
			IElement::operator=(other);
			transform = other.transform;
			type = other.type;
			switch (type)
			{
				case Type::CUBE:
					cube = other.cube;
					break;
				case Type::SPHERE:
					sphere = other.sphere;
					break;
				case Type::CYLINDER:
					cylinder = other.cylinder;
					break;
				case Type::RECTANGLE:
					rectangle = other.rectangle;
					break;
				case Type::DISK:
					disk = other.disk;
					break;
				case Type::OBJ:
					obj = other.obj;
					break;
				case Type::PLY:
					ply = other.ply;
					break;
				case Type::SERIALIZED:
					serialized = other.serialized;
					break;
				case Type::SHAPEGROUP:
					shapegroup = other.shapegroup;
					break;
				case Type::INSTANCE:
					instance = other.instance;
					break;/*
				case Type::HAIR:
					hair = other.hair;
					break;
				case Type::HEIGHTFIELD:
					heightfield = other.heightfield;
					break;*/
				default:
					break;
			}
			bsdf = other.bsdf;
			emitter = other.emitter;
			return *this;
		}
		inline CElementShape& operator=(CElementShape&& other)
		{
			IElement::operator=(std::move(other));
			std::swap(transform,other.transform);
			std::swap(type,other.type);
			switch (type)
			{
				case Type::CUBE:
					std::swap(cube,other.cube);
					break;
				case Type::SPHERE:
					std::swap(sphere,other.sphere);
					break;
				case Type::CYLINDER:
					std::swap(cylinder,other.cylinder);
					break;
				case Type::RECTANGLE:
					std::swap(rectangle,other.rectangle);
					break;
				case Type::DISK:
					std::swap(disk,other.disk);
					break;
				case Type::OBJ:
					std::swap(obj,other.obj);
					break;
				case Type::PLY:
					std::swap(ply,other.ply);
					break;
				case Type::SERIALIZED:
					std::swap(serialized,other.serialized);
					break;
				case Type::SHAPEGROUP:
					std::swap(shapegroup,other.shapegroup);
					break;
				case Type::INSTANCE:
					std::swap(instance,other.instance);
					break;/*
				case Type::HAIR:
					std::swap(hair,other.hair);
					break;
				case Type::HEIGHTFIELD:
					std::swap(heightfield,other.heightfield);
					break;*/
				default:
					break;
			}
			std::swap(bsdf,other.bsdf);
			std::swap(emitter,other.emitter);
			return *this;
		}

		bool addProperty(SNamedPropertyElement&& _property) override;
		bool onEndTag(asset::IAssetLoader::IAssetLoaderOverride* _override, CMitsubaMetadata* globalMetadata) override;
		IElement::Type getType() const override { return IElement::Type::SHAPE; }
		std::string getLogName() const override { return "shape"; }

		
		inline core::matrix3x4SIMD getAbsoluteTransform() const
		{
			auto local = transform.matrix.extractSub3x4();
			// TODO restore at some point (and make it actually work??)
			// note: INSTANCE can only contain SHAPEGROUP and the latter doesnt have its own transform

			//if (type==CElementShape::INSTANCE && instance.parent)
			//	return core::concatenateBFollowedByA(local,instance.parent->getAbsoluteTransform());
			return local;
		}

		inline CElementEmitter obtainEmitter() const
		{
			if (emitter)
				return *emitter;
			if (type==CElementShape::INSTANCE && instance.parent && instance.parent->emitter)
				return *instance.parent->emitter;

			return CElementEmitter("");
		}


		bool processChildData(IElement* _child, const std::string& name) override;

		//
		Type type;
		CElementTransform transform;/*
		IElement::Type toWorldType;
		// nullptr means identity matrix
		union
		{
			CElementTransform* transform;
			CElementAnimation* animation;
		};*/
		union
		{
			Base			cube;
			Sphere			sphere;
			Cylinder		cylinder;
			Base			rectangle;
			Base			disk;
			Obj				obj;
			Ply				ply;
			Serialized		serialized;
			ShapeGroup		shapegroup;
			Instance		instance;
			//Hair			hair;
			//Heightfield	heightfield;
		};
		// optionals
		CElementBSDF*	bsdf;
		CElementEmitter*emitter;
};


}
}
}

#endif