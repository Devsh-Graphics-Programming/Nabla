// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXT_MISTUBA_LOADER_C_ELEMENT_SENSOR_H_INCLUDED_
#define _NBL_EXT_MISTUBA_LOADER_C_ELEMENT_SENSOR_H_INCLUDED_


#include "nbl/ext/MitsubaLoader/IElement.h"
#include "nbl/ext/MitsubaLoader/CElementTransform.h"
#include "nbl/ext/MitsubaLoader/CElementFilm.h"
#include "nbl/ext/MitsubaLoader/CElementSampler.h"


namespace nbl::ext::MitsubaLoader
{

class CElementSensor final : public IElement
{
	public:
		enum Type : uint8_t
		{
			PERSPECTIVE,
			THINLENS,
			ORTHOGRAPHIC,
			TELECENTRIC,
			SPHERICAL,
			IRRADIANCEMETER,
			RADIANCEMETER,
			FLUENCEMETER,
			PERSPECTIVE_RDIST,
			INVALID
		};

		constexpr static inline uint8_t MaxClipPlanes = 6u;

		struct ShutterSensor
		{
			hlsl::float32_t3 up = hlsl::float32_t3(0,1,0);
			hlsl::float32_t4 clipPlanes[MaxClipPlanes] = {};
			float moveSpeed = core::nan<float>();
			float zoomSpeed = core::nan<float>();
			float rotateSpeed = core::nan<float>();
			float shutterOpen = 0.f;
			float shutterClose = 0.f;
		};
		struct CameraBase : ShutterSensor
		{
			float nearClip = 0.01f;
			float farClip = 10000.f;
		};
		struct PerspectivePinhole : CameraBase
		{
			constexpr static inline Type VariantType = Type::PERSPECTIVE;

			enum class FOVAxis
			{
				INVALID,
				X,
				Y,
				DIAGONAL,
				SMALLER,
				LARGER
			};

			void setFoVFromFocalLength(float focalLength)
			{
				_NBL_DEBUG_BREAK_IF(true); // TODO
			}

			float shiftX = 0.f;
			float shiftY = 0.f;
			float fov = 53.2f;
			FOVAxis fovAxis = FOVAxis::X;
		};
		struct Orthographic : CameraBase
		{
			constexpr static inline Type VariantType = Type::ORTHOGRAPHIC;
		};
		struct DepthOfFieldBase
		{
			float apertureRadius = 0.f;
			float focusDistance = 0.f;
		};
		struct PerspectiveThinLens : PerspectivePinhole, DepthOfFieldBase
		{
			constexpr static inline Type VariantType = Type::THINLENS;
		};
		struct TelecentricLens : Orthographic, DepthOfFieldBase
		{
			constexpr static inline Type VariantType = Type::TELECENTRIC;
		};
		struct SphericalCamera : CameraBase
		{
			constexpr static inline Type VariantType = Type::SPHERICAL;
		};
		struct IrradianceMeter : ShutterSensor
		{
			constexpr static inline Type VariantType = Type::IRRADIANCEMETER;
		};
		struct RadianceMeter : ShutterSensor
		{
			constexpr static inline Type VariantType = Type::RADIANCEMETER;
		};
		struct FluenceMeter : ShutterSensor
		{
			constexpr static inline Type VariantType = Type::FLUENCEMETER;
		};/*
		struct PerspectivePinholeRadialDistortion : PerspectivePinhole
		{
			kc;
		};*/

		using variant_list_t = core::type_list<
			PerspectivePinhole,
			PerspectiveThinLens,
			Orthographic,
			TelecentricLens,
			SphericalCamera,
			IrradianceMeter,
			RadianceMeter,
			FluenceMeter
		>;
		static inline core::unordered_map<core::string,Type,core::CaseInsensitiveHash,core::CaseInsensitiveEquals> compStringToTypeMap()
		{
			return {
				{"perspective",			Type::PERSPECTIVE},
				{"thinlens",			Type::THINLENS},
				{"orthographic",		Type::ORTHOGRAPHIC},
				{"telecentric",			Type::TELECENTRIC},
				{"spherical",			Type::SPHERICAL},
				{"irradiancemeter",		Type::IRRADIANCEMETER},
				{"radiancemeter",		Type::RADIANCEMETER},
				{"fluencemeter",		Type::FLUENCEMETER}/*,
				{"perspective_rdist",	PERSPECTIVE_RDIST}*/
			};
		}
		static AddPropertyMap<CElementSensor> compAddPropertyMap();

		inline CElementSensor(const char* id) : IElement(id), type(Type::INVALID), /*toWorldType(IElement::Type::TRANSFORM),*/ transform(), film(""), sampler("")
		{
		}
		inline CElementSensor(const CElementSensor& other) : IElement(""), transform(), film(""), sampler("")
		{
			operator=(other);
		}
		inline ~CElementSensor()
		{
		}

		template<typename Visitor>
		inline void visit(Visitor&& visitor)
		{
			switch (type)
			{
				case CElementSensor::Type::PERSPECTIVE:
					visitor(perspective);
					break;
				case CElementSensor::Type::THINLENS:
					visitor(thinlens);
					break;
				case CElementSensor::Type::ORTHOGRAPHIC:
					visitor(orthographic);
					break;
				case CElementSensor::Type::TELECENTRIC:
					visitor(telecentric);
					break;
				case CElementSensor::Type::SPHERICAL:
					visitor(spherical);
					break;
				case CElementSensor::Type::IRRADIANCEMETER:
					visitor(irradiancemeter);
					break;
				case CElementSensor::Type::RADIANCEMETER:
					visitor(radiancemeter);
					break;
				case CElementSensor::Type::FLUENCEMETER:
					visitor(fluencemeter);
					break;
				default:
					break;
			}
		}
		template<typename Visitor>
		inline void visit(Visitor&& visitor) const
		{
			const_cast<CElementSensor*>(this)->visit([&]<typename T>(T& var)->void
				{
					visitor(const_cast<const T&>(var));
				}
			);
		}

		inline CElementSensor& operator=(const CElementSensor& other)
		{
			IElement::operator=(other);
			type = other.type;
			transform = other.transform;
			IElement::copyVariant(this,&other);
			film = other.film;
			sampler = other.sampler;
			return *this;
		}

		bool onEndTag(CMitsubaMetadata* globalMetadata, system::logger_opt_ptr logger) override;

		constexpr static inline auto ElementType = IElement::Type::SENSOR;
		inline IElement::Type getType() const override { return ElementType; }
		inline std::string getLogName() const override { return "sensor"; }

		inline bool processChildData(IElement* _child, const std::string& name, system::logger_opt_ptr logger) override
		{
			if (!_child)
				return true;
			switch (_child->getType())
			{
				case IElement::Type::TRANSFORM:
					{
						auto tform = static_cast<CElementTransform*>(_child);
						if (name != "toWorld")
						{
							logger.log("The <transform> nested inside <sensor> needs to be named \"toWorld\"",system::ILogger::ELL_ERROR);
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
				case IElement::Type::FILM:
					film = *static_cast<CElementFilm*>(_child);
					if (film.type!=CElementFilm::Type::INVALID)
						return true;
					break;
				case IElement::Type::SAMPLER:
					sampler = *static_cast<CElementSampler*>(_child);
					if (sampler.type!=CElementSampler::Type::INVALID)
						return true;
					break;
				default:
					break;
			}
			logger.log("Only valid nested children inside <sensor> are: VALID <transform>, <film>, and <sampler>. The <animation> is not supported yet.",system::ILogger::ELL_ERROR);
			return false;
		}

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
			PerspectivePinhole	perspective;
			PerspectiveThinLens	thinlens;
			Orthographic		orthographic;
			TelecentricLens		telecentric;
			SphericalCamera		spherical;
			IrradianceMeter		irradiancemeter;
			RadianceMeter		radiancemeter;
			FluenceMeter		fluencemeter;
			//PerspectivePinholeRadialDistortion perspective_rdist;
		};
		CElementFilm	film;
		CElementSampler	sampler;
};


}
#endif