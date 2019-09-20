#ifndef __C_ELEMENT_SENSOR_H_INCLUDED__
#define __C_ELEMENT_SENSOR_H_INCLUDED__

#include "../../ext/MitsubaLoader/IElement.h"
#include "../../ext/MitsubaLoader/CElementFilm.h"
#include "../../ext/MitsubaLoader/CElementSampler.h"

#include "irrlicht.h"

namespace irr { namespace ext { namespace MitsubaLoader {

enum class ESensorType
{
	NONE,
	PERSPECTIVE,
	THINLENS,
	ORTHOGRAPHIC,
	TELECENTRIC,
	SPHERICAL,
	IRRADIANCEMETER,
	RADIANCEMETER,
	FLUENCEMETER
};

enum class EFOVAxis
{
	X,
	Y,
	DIAGONAL,
	SMALLER,
	LARGER
};

struct SSensorMetadata
{
	SSensorMetadata()
		:type(ESensorType::NONE) {};

	SSensorMetadata(const SSensorMetadata& other)
		: type(other.type), shutterOpen(other.shutterOpen), shutterClose(other.shutterClose)
	{
		switch (other.type)
		{
		case ESensorType::PERSPECTIVE:
			perspectiveData = other.perspectiveData;
			break;

		case ESensorType::THINLENS:
			thinlensData = other.thinlensData;
			break;

		case ESensorType::ORTHOGRAPHIC:
			perspectiveData = other.perspectiveData;
			break;

		case ESensorType::TELECENTRIC:
			telecentricData = other.telecentricData;
			break;

		case ESensorType::SPHERICAL:
			sphericalData = other.sphericalData;
			break;

		case ESensorType::RADIANCEMETER:
			radiancemeterData = other.radiancemeterData;
			break;

		case ESensorType::FLUENCEMETER:
			fluencemeterData = other.fluencemeterData;
			break;

		default:
			assert(false);
		}
	}

	~SSensorMetadata()
	{

	}

	SSamplerMetadata samperData;
	SFilmMetadata filmData;

	ESensorType type;

	float shutterOpen = 0.0f;
	float shutterClose = 0.0f;

	union
	{
		struct PerspectiveData
		{
			core::matrix4SIMD toWorld;
			std::string focalLength;
			EFOVAxis fovAxis;
			float fov;
			float nearClip;
			float farClip;

		} perspectiveData;

		struct ThinlensData
		{
			core::matrix4SIMD toWorld;
			std::string focalLength;
			EFOVAxis fovAxis;
			float fov;
			float nearClip;
			float farClip;
			float apertureRadius;
			float focusDistance;

		} thinlensData;

		struct OrthographicData
		{
			core::matrix4SIMD toWorld;
			float nearClip;
			float farClip;

		} orthographicData;

		struct TelecentricData
		{
			core::matrix4SIMD toWorld;
			float nearClip;
			float farClip;
			float apertureRadius;
			float focusDistance;

		} telecentricData;

		struct SphericalData
		{
			core::matrix4SIMD toWorld;

		} sphericalData;

		struct RadiancemeterData
		{
			core::matrix4SIMD toWorld;

		} radiancemeterData;

		struct FluencemeterData
		{
			core::matrix4SIMD toWorld;

		} fluencemeterData;
	};
};

class CElementSensor : public IElement
{
public:
	virtual bool processAttributes(const char** _atts) override;
	virtual bool processChildData(IElement* _child) override;
	virtual bool onEndTag(asset::IAssetLoader::IAssetLoaderOverride* _override) override;
	virtual IElement::Type getType() const override { return IElement::Type::SENSOR; }
	virtual std::string getLogName() const override { return "sensor"; }

	SSensorMetadata getMetadata() { return data; }

private:
	bool processSharedDataProperty(const SPropertyElementData& _property);
	bool processPerspectiveSensorProperties();
	bool processThinlensSensorProperties();
	bool processOrthographicSensorProperties();
	bool processTelecentricSensorProperties();

private:
	SSensorMetadata data;
	core::matrix4SIMD transform;

};



}
}
}

#endif