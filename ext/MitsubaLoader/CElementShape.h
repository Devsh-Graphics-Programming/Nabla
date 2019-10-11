#ifndef __C_ELEMENT_SHAPE_H_INCLUDED__
#define __C_ELEMENT_SHAPE_H_INCLUDED__

#include "../../ext/MitsubaLoader/IElement.h"
#include "../../ext/MitsubaLoader/CElementTransform.h"
#include "../../ext/MitsubaLoader/CElementBSDF.h"
#include "../../ext/MitsubaLoader/CElementEmitter.h"


namespace irr
{
namespace ext
{
namespace MitsubaLoader
{

class CGlobalMitsubaMetadata;

class CElementShape : public IElement
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
			_IRR_STATIC_INLINE_CONSTEXPR size_t MaxChildCount = 128u;
			size_t childCount = 0u;
			CElementShape* children[MaxChildCount] = { nullptr };
		};
		struct Instance
		{
			ShapeGroup shapegroup;
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

		bool addProperty(SNamedPropertyElement&& _property) override;
		bool onEndTag(asset::IAssetLoader::IAssetLoaderOverride* _override, CGlobalMitsubaMetadata* globalMetadata) override;
		IElement::Type getType() const override { return IElement::Type::SHAPE; }
		std::string getLogName() const override { return "shape"; }

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