// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __C_ELEMENT_SENSOR_H_INCLUDED__
#define __C_ELEMENT_SENSOR_H_INCLUDED__

#include "nbl/ext/MitsubaLoader/IElement.h"
#include "nbl/ext/MitsubaLoader/CElementTransform.h"
#include "nbl/ext/MitsubaLoader/CElementFilm.h"
#include "nbl/ext/MitsubaLoader/CElementSampler.h"

namespace nbl
{
namespace ext
{
namespace MitsubaLoader
{
class CElementSensor : public IElement
{
public:
    enum Type
    {
        INVALID,
        PERSPECTIVE,
        THINLENS,
        ORTHOGRAPHIC,
        TELECENTRIC,
        SPHERICAL,
        IRRADIANCEMETER,
        RADIANCEMETER,
        FLUENCEMETER,
        PERSPECTIVE_RDIST
    };
    struct ShutterSensor
    {
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
            _NBL_DEBUG_BREAK_IF(true);  // TODO
        }

        float fov = 53.2f;
        FOVAxis fovAxis = FOVAxis::X;
    };
    struct Orthographic : CameraBase
    {
    };
    struct DepthOfFieldBase
    {
        float apertureRadius = 0.f;
        float focusDistance = 0.f;
    };
    struct PerspectiveThinLens : PerspectivePinhole, DepthOfFieldBase
    {
    };
    struct TelecentricLens : Orthographic, DepthOfFieldBase
    {
    };
    struct SphericalCamera : CameraBase
    {
    };
    struct IrradianceMeter : ShutterSensor
    {
    };
    struct RadianceMeter : ShutterSensor
    {
    };
    struct FluenceMeter : ShutterSensor
    {
    }; /*
		struct PerspectivePinholeRadialDistortion : PerspectivePinhole
		{
			kc;
		};*/

    CElementSensor(const char* id)
        : IElement(id), type(Type::INVALID), /*toWorldType(IElement::Type::TRANSFORM),*/ transform(), film(""), sampler("")
    {
    }
    CElementSensor(const CElementSensor& other)
        : IElement(""), transform(), film(""), sampler("")
    {
        operator=(other);
    }
    virtual ~CElementSensor()
    {
    }

    inline CElementSensor& operator=(const CElementSensor& other)
    {
        IElement::operator=(other);
        type = other.type;
        transform = other.transform;
        switch(type)
        {
            case CElementSensor::Type::PERSPECTIVE:
                perspective = other.perspective;
                break;
            case CElementSensor::Type::THINLENS:
                thinlens = other.thinlens;
                break;
            case CElementSensor::Type::ORTHOGRAPHIC:
                orthographic = other.orthographic;
                break;
            case CElementSensor::Type::TELECENTRIC:
                telecentric = other.telecentric;
                break;
            case CElementSensor::Type::SPHERICAL:
                spherical = other.spherical;
                break;
            case CElementSensor::Type::IRRADIANCEMETER:
                irradiancemeter = other.irradiancemeter;
                break;
            case CElementSensor::Type::RADIANCEMETER:
                radiancemeter = other.radiancemeter;
                break;
            case CElementSensor::Type::FLUENCEMETER:
                fluencemeter = other.fluencemeter;
                break;
            default:
                break;
        }
        film = other.film;
        sampler = other.sampler;
        return *this;
    }

    bool addProperty(SNamedPropertyElement&& _property) override;
    bool onEndTag(asset::IAssetLoader::IAssetLoaderOverride* _override, CMitsubaMetadata* globalMetadata) override;
    IElement::Type getType() const override { return IElement::Type::SENSOR; }
    std::string getLogName() const override { return "sensor"; }

    bool processChildData(IElement* _child, const std::string& name) override
    {
        if(!_child)
            return true;
        switch(_child->getType())
        {
            case IElement::Type::TRANSFORM: {
                auto tform = static_cast<CElementTransform*>(_child);
                if(name != "toWorld")
                    return false;
                //toWorldType = IElement::Type::TRANSFORM;
                transform = *tform;
                return true;
            }
            break; /*
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
                if(film.type != CElementFilm::Type::INVALID)
                    return true;
                break;
            case IElement::Type::SAMPLER:
                sampler = *static_cast<CElementSampler*>(_child);
                if(sampler.type != CElementSampler::Type::INVALID)
                    return true;
                break;
        }
        return false;
    }

    //
    Type type;
    CElementTransform transform; /*
		IElement::Type toWorldType;
		// nullptr means identity matrix
		union
		{
			CElementTransform* transform;
			CElementAnimation* animation;
		};*/
    union
    {
        PerspectivePinhole perspective;
        PerspectiveThinLens thinlens;
        Orthographic orthographic;
        TelecentricLens telecentric;
        SphericalCamera spherical;
        IrradianceMeter irradiancemeter;
        RadianceMeter radiancemeter;
        FluenceMeter fluencemeter;
        //PerspectivePinholeRadialDistortion perspective_rdist;
    };
    CElementFilm film;
    CElementSampler sampler;
};

}
}
}

#endif