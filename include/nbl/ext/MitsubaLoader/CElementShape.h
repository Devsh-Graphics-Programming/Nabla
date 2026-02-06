// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXT_MISTUBA_LOADER_C_ELEMENT_SHAPE_H_INCLUDED_
#define _NBL_EXT_MISTUBA_LOADER_C_ELEMENT_SHAPE_H_INCLUDED_


#include "nbl/ext/MitsubaLoader/IElement.h"
#include "nbl/ext/MitsubaLoader/CElementTransform.h"
#include "nbl/ext/MitsubaLoader/CElementBSDF.h"
#include "nbl/ext/MitsubaLoader/CElementEmitter.h"

#include "nbl/builtin/hlsl/math/linalg/basic.hlsl"


namespace nbl::ext::MitsubaLoader
{


class CElementShape final : public IElement
{
	public:
		enum Type : uint8_t
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
		static inline core::unordered_map<core::string,Type,core::CaseInsensitiveHash,core::CaseInsensitiveEquals> compStringToTypeMap()
		{
			return {
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
		}

	struct Base
	{
		bool flipNormals = false;
	};
		struct Cube : Base
		{
			constexpr static inline Type VariantType = Type::CUBE;
		};
		struct Sphere : Base
		{
			constexpr static inline Type VariantType = Type::SPHERE;
			
			hlsl::float32_t3 center = {0,0,0};
			float radius = 1.f;
		};
		struct Cylinder : Base
		{
			constexpr static inline Type VariantType = Type::CYLINDER;

			hlsl::float32_t3 p0 = {0,0,0};
			hlsl::float32_t3 p1 = {0,0,1};
			float radius = 1.f;
		};
	struct LoadedFromFileBase : Base
	{
		constexpr static inline uint16_t MaxPathLen = 1024u;

		char	filename[MaxPathLen];
		//! Use face normals (any per-vertex normals will be discarded)
		bool	faceNormals = false;
		float	maxSmoothAngle = NAN;
	};
		struct Obj : LoadedFromFileBase
		{
			constexpr static inline Type VariantType = Type::OBJ;

			bool	flipTexCoords = true;
			bool	collapse = false;
		};
		struct Ply : LoadedFromFileBase
		{
			constexpr static inline Type VariantType = Type::PLY;

			bool	srgb = true;
		};
		struct Serialized : LoadedFromFileBase
		{
			constexpr static inline Type VariantType = Type::SERIALIZED;

			int32_t	shapeIndex;
		};
		// geometries basically
		struct ShapeGroup
		{
			constexpr static inline Type VariantType = Type::SHAPEGROUP;
			constexpr static inline size_t MaxChildCount = 128u;

			size_t childCount = 0u;
			CElementShape* children[MaxChildCount] = { nullptr };
		};
		struct Instance
		{
			constexpr static inline Type VariantType = Type::INSTANCE;

			CElementShape* parent = nullptr;
		};/*
		struct Hair : Base
		{
			SPropertyElementData	filename;
			float					raidus = 0.025f;
			float					angleThreshold = 1.f;
			float					reduction = 0.f;
		};
		struct HeightField : Base
		{
			SPropertyElementData filename;
			boolean shadingNormals;
			int32_t width;
			int32_t height;
			float	scale;
			CElementTexture* texture;
		};*/

		//
		using variant_list_t = core::type_list<
			Sphere,
			Cylinder,
			Obj,
			Ply,
			Serialized,
			ShapeGroup,
			Instance
		>;
		//
		static AddPropertyMap<CElementShape> compAddPropertyMap();

		inline CElementShape(const char* id) : IElement(id), type(Type::INVALID), /*toWorldType(IElement::Type::TRANSFORM),*/ transform(), bsdf(nullptr), emitter(nullptr)
		{
		}
		inline CElementShape(const CElementShape& other) : IElement(""), transform(), bsdf(nullptr), emitter(nullptr)
		{
			operator=(other);
		}
		inline CElementShape(CElementShape&& other) : IElement(""), transform(), bsdf(nullptr), emitter(nullptr)
		{
			operator=(std::move(other));
		}
		inline ~CElementShape()
		{
		}

		template<typename Visitor>
		inline void visit(Visitor&& visitor)
		{
			switch (type)
			{
				case CElementShape::Type::CUBE:
					visitor(cube);
					break;
				case CElementShape::Type::SPHERE:
					visitor(sphere);
					break;
				case CElementShape::Type::CYLINDER:
					visitor(cylinder);
					break;
				case CElementShape::Type::RECTANGLE:
					visitor(rectangle);
					break;
				case CElementShape::Type::DISK:
					visitor(disk);
					break;
				case CElementShape::Type::OBJ:
					visitor(obj);
					break;
				case CElementShape::Type::PLY:
					visitor(ply);
					break;
				case CElementShape::Type::SERIALIZED:
					visitor(serialized);
					break;
				case CElementShape::Type::SHAPEGROUP:
					visitor(shapegroup);
					break;
				case CElementShape::Type::INSTANCE:
					visitor(instance);
					break;/*
				case CElementShape::Type::HAIR:
					visitor(hair);
					break;
				case CElementShape::Type::HEIGHTFIELD:
					visitor(heightfield);
					break;*/
				default:
					break;
			}
		}
		template<typename Visitor>
		inline void visit(Visitor&& visitor) const
		{
			const_cast<CElementShape*>(this)->visit([&]<typename T>(T& var)->void
				{
					visitor(const_cast<const T&>(var));
				}
			);
		}

		inline CElementShape& operator=(const CElementShape& other)
		{
			IElement::operator=(other);
			transform = other.transform;
			type = other.type;
			IElement::copyVariant(this,&other);
			bsdf = other.bsdf;
			emitter = other.emitter;
			return *this;
		}

		bool onEndTag(CMitsubaMetadata* globalMetadata, system::logger_opt_ptr logger) override;

		constexpr static inline auto ElementType = IElement::Type::SHAPE;
		inline IElement::Type getType() const override { return ElementType; }
		inline std::string getLogName() const override { return "shape"; }

		
		inline hlsl::float32_t3x4 getTransform() const
		{
			// explicit truncation
			auto local = hlsl::float32_t3x4(transform.matrix);

			// SHAPEGROUP cannot have its own transformation
			assert(type!=Type::SHAPEGROUP || hlsl::math::linalg::diagonal<hlsl::float32_t3x4>(1)==local);

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


		bool processChildData(IElement* _child, const std::string& name, system::logger_opt_ptr logger) override;

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
			Cube			cube;
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
#endif
