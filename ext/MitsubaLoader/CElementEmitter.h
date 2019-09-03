#ifndef __C_ELEMENT_EMITTER_H_INCLUDED__
#define __C_ELEMENT_EMITTER_H_INCLUDED__

#include "../../ext/MitsubaLoader/IElement.h"
#include "irrlicht.h"

namespace irr { namespace ext { namespace MitsubaLoader {

enum class EEmitterType
{
	NONE,
	POINT,
	AREA,
	SPOT,
	DIRECTIONAL,
	COLLIMATED,
	SKY,
	SUN,
	SUNSKY,
	ENVMAP,
	CONSTANT
};

struct SEmitterMetadata
{
	SEmitterMetadata()
		:type(EEmitterType::NONE) {};

	SEmitterMetadata(const SEmitterMetadata& other)
		: type(other.type), shutterOpen(other.shutterOpen), shutterClose(other.shutterClose)
	{
		switch (other.type)
		{
		case EEmitterType::POINT:
			pointData = other.pointData;
			break;

		case EEmitterType::AREA:
			areaData = other.areaData;
			break;

		case EEmitterType::SPOT:
			spotData = other.spotData;
			break;

		case EEmitterType::DIRECTIONAL:
			directionalData = other.directionalData;
			break;

		case EEmitterType::COLLIMATED:
			collimatedData = other.collimatedData;
			break;

		case EEmitterType::SKY:
			skyData = other.skyData;
			break;

		case EEmitterType::SUN:
			sunData = other.sunData;
			break;

		case EEmitterType::SUNSKY:
			sunskyData = other.sunskyData;
			break;

		case EEmitterType::ENVMAP:
			envmapData = other.envmapData;
			break;

		case EEmitterType::CONSTANT:
			constantData = other.constantData;
			break;

		default:
			assert(false);
		}
	}

	~SEmitterMetadata()
	{

	}

	EEmitterType type;

	float shutterOpen = 0.0f;
	float shutterClose = 0.0f;

	union
	{
		struct PointData
		{
			core::vector3df_SIMD position;
			video::SColorf intensity;
			float samplingWeight;

		} pointData;

		struct AreaData
		{
			video::SColorf radiance;
			float samplingWeight;

		} areaData;

		struct SpotData
		{
			core::matrix4SIMD toWorld;
			video::SColorf intensity;
			float cutoffAngle;
			float beamWidth;
			//texture
			float samplingWeight;

		} spotData;

		struct DirectionalData
		{
			core::matrix4SIMD toWorld;
			video::SColorf irradiance;
			core::vector3df_SIMD direction;
			float samplingWeight;

		} directionalData;

		struct CollimatedData
		{
			core::matrix4SIMD toWorld;
			video::SColorf power;
			float samplingWeight;

		} collimatedData;

		struct SkyData
		{
			core::matrix4SIMD toWorld;
			float samplingWeight;
			float turbidity;
			video::SColorf albedo;
			int day;
			int month;
			int year;
			float second;
			float minute;
			float hour;
			float latitude;
			float longitude;
			float timezone;
			core::vector3df_SIMD sunDirection;
			float stretch;
			int resolution;
			float scale;

		} skyData;

		struct SunData
		{
			float samplingWeight;
			float turbidity;
			int day;
			int month;
			int year;
			float second;
			float minute;
			float hour;
			float latitude;
			float longitude;
			float timezone;
			core::vector3df_SIMD sunDirection;
			float scale;
			float sunRadiusScale;

		} sunData;

		struct SunskyData
		{
			float turbidity;
			video::SColorf albedo;
			int day;
			int month;
			int year;
			float second;
			float minute;
			float hour;
			float latitude;
			float longitude;
			float timezone;
			core::vector3df_SIMD sunDirection;
			float stretch;
			float sunScale;
			float skyScale;
			float sunRadiusScale;
			int resolution;

		} sunskyData;

		struct EnvmapData
		{
			core::matrix4SIMD toWorld;
			std::string filename;
			float scale;
			float gamma;
			bool cache;
			float samplingWeight;

		} envmapData;

		struct ConstantData
		{
			video::SColorf radiance;
			float samplingWeight;

		} constantData;
	};
};

class CElementEmitter : public IElement
{
public:
	virtual bool processAttributes(const char** _atts) override;
	virtual bool processChildData(IElement* _child) override;
	virtual bool onEndTag(asset::IAssetManager& _assetManager) override;
	virtual IElement::Type getType() const override { return IElement::Type::EMITTER; }
	virtual std::string getLogName() const override { return "emitter"; }

	SEmitterMetadata getMetadata() { return data; }

private:
	bool processSharedDataProperty(const SPropertyElementData& _property);
	bool processPointEmitterProperties();
	bool processAreaEmitterProperties();
	bool processSpotEmitterProperties();
	bool processDirectionalEmitterProperties();
	bool processCollimatedEmitterProperties();
	bool processConstantEmitterProperties();

private:
	SEmitterMetadata data;
	core::matrix4SIMD transform;

};



}
}
}

#endif