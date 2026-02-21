// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXT_MISTUBA_LOADER_C_ELEMENT_EMITTER_H_INCLUDED_
#define _NBL_EXT_MISTUBA_LOADER_C_ELEMENT_EMITTER_H_INCLUDED_


#include "nbl/ext/MitsubaLoader/CElementTexture.h"
#include "nbl/ext/MitsubaLoader/CElementEmissionProfile.h"

#include <cmath>


namespace nbl::ext::MitsubaLoader
{
	
class CElementEmitter : public IElement
{
	public:
		enum Type
		{
			INVALID,
			POINT,
			AREA,
			//SPOT, // deprecated, use POINT with an IES profile instead!
			DIRECTIONAL,
			COLLIMATED,
			//SKY,
			//SUN,
			//SUNSKY,
			ENVMAP,
			CONSTANT
		};
		static inline core::unordered_map<core::string,Type,core::CaseInsensitiveHash,core::CaseInsensitiveEquals> compStringToTypeMap()
		{
			return {
				{"point",		CElementEmitter::Type::POINT},
				{"area",		CElementEmitter::Type::AREA},
				{"directional",	CElementEmitter::Type::DIRECTIONAL},
				{"collimated",	CElementEmitter::Type::COLLIMATED},/*
				{"sky",			CElementEmitter::Type::SKY},
				{"sun",			CElementEmitter::Type::SUN},
				{"sunsky",		CElementEmitter::Type::SUNSKY},*/
				{"envmap",		CElementEmitter::Type::ENVMAP},
				{"constant",	CElementEmitter::Type::CONSTANT}
			};
		}


	struct SampledEmitter
	{
		inline SampledEmitter() : samplingWeight(1.f) {}

		float samplingWeight;
	};
	struct DeltaDistributionEmitter : SampledEmitter
	{
		// Watts
		hlsl::float32_t3 intensity = {1.f,1.f,1.f};
	};
	struct SolidAngleEmitter : SampledEmitter
	{
		// Watts Steradian^-1
		hlsl::float32_t3 radiance = {1.f,1.f,1.f};
	};
	struct EmissionProfileEmitter
	{
		CElementEmissionProfile* emissionProfile = nullptr;
	};
		struct Point : DeltaDistributionEmitter, EmissionProfileEmitter
		{
			constexpr static inline Type VariantType = Type::POINT;
		};
		struct Area : SolidAngleEmitter, EmissionProfileEmitter
		{
			constexpr static inline Type VariantType = Type::AREA;
		};
		struct Directional : SampledEmitter
		{
			constexpr static inline Type VariantType = Type::DIRECTIONAL;

			hlsl::float32_t3 irradiance = {1.f,1.f,1.f}; // Watts Meter^-2
		};
		struct Collimated : SampledEmitter
		{
			constexpr static inline Type VariantType = Type::COLLIMATED;

			hlsl::float32_t3 power = {1.f,1.f,1.f}; // Watts
		};/*
		struct Sky : SampledEmitter
		{
			float turbidity = 3.f;
			hlsl::float32_t3 albedo = {0.15f,0.15f,0.15f};
			hlsl::float32_t3 sunDirection = calculate default from tokyo japan at 15:00 on 10.07.2010;
			float stretch = 1.f; // must be [1,2]
			int32_t resolution = 512;
			float scale = 1.f;
		};
		struct Sun : SampledEmitter
		{
			float turbidity = 3.f;
			hlsl::float32_t3 sunDirection = calculate default from tokyo japan at 15:00 on 10.07.2010;
			int32_t resolution = 512;
			float scale = 1.f;
			float sunRadiusScale = 1.f;

		};
		struct SunSky : Sky
		{
			float sunRadiusScale = 1.f;
		};*/
		struct EnvMap : SampledEmitter
		{
			constexpr static inline Type VariantType = Type::ENVMAP;
			constexpr static inline uint16_t MaxPathLen = 1024u;

			char	filename[MaxPathLen];
			float	scale = 1.f;
			float	gamma = NAN;
			//bool cache = false;
		};
		struct Constant : SolidAngleEmitter
		{
			constexpr static inline Type VariantType = Type::CONSTANT;
		};

		//
		using variant_list_t = core::type_list<
			Point,
			Area,
			Directional,
			Collimated,
//			Sky,
//			Sun,
//			SunSky,
			EnvMap,
			Constant
		>;
		//
		static AddPropertyMap<CElementEmitter> compAddPropertyMap();

		//
		inline CElementEmitter(const char* id) : IElement(id), type(Type::INVALID), /*toWorldType(IElement::Type::TRANSFORM),*/ transform()
		{
		}
		inline CElementEmitter() : CElementEmitter("") {}
		inline CElementEmitter(const CElementEmitter& other) : IElement(""), transform()
		{
			operator=(other);
		}
		virtual ~CElementEmitter()
		{
		}

		template<typename Visitor>
		inline void visit(Visitor&& func)
		{
			switch (type)
			{
				case Type::POINT:
					func(point);
					break;
				case Type::AREA:
					func(area);
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
					break;
			}
		}
		template<typename Visitor>
		inline void visit(Visitor&& visitor) const
		{
			const_cast<CElementEmitter*>(this)->visit([&]<typename T>(T& var)->void
				{
					visitor(const_cast<const T&>(var));
				}
			);
		}

		inline CElementEmitter& operator=(const CElementEmitter& other)
		{
			IElement::operator=(other);
			transform = other.transform;
			type = other.type;
			IElement::copyVariant(this,&other);
			return *this;
		}

		bool onEndTag(CMitsubaMetadata* globalMetadata, system::logger_opt_ptr logger) override;
		
		constexpr static inline auto ElementType = IElement::Type::EMITTER;
		inline IElement::Type getType() const override { return ElementType; }
		std::string getLogName() const override { return "emitter"; }

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
			Point		point;
			Area		area;
			Directional	directional;
			Collimated	collimated;/*
			Sky			sky;
			Sun			sun;
			SunSky		sunsky;*/
			EnvMap		envmap;
			Constant	constant;
		};
};

}
#endif