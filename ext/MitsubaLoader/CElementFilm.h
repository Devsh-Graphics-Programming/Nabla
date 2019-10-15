#ifndef __C_ELEMENT_FILM_H_INCLUDED__
#define __C_ELEMENT_FILM_H_INCLUDED__

#include "../../ext/MitsubaLoader/IElement.h"
#include "irrlicht.h"

namespace irr { namespace ext { namespace MitsubaLoader {

	/*The options are float16, float32, or
		uint32. (Default: float16).*/

enum class ETonemapMethod
{
	GAMMA,
	REINHARD
};

enum class EComponentFormat
{
	FLOAT16,
	FLOAT32,
	UINT32
};

enum class EHDRFileFormat
{
	OPENEXR,
	RGBE,
	PFM
};

enum class ELDRFileFormat
{
	PNG,
	JPEG
};

enum class EMFileFormat
{
	MATLAB,
	MATHEMATICA,
	NUMPY
};

enum class EFilmType
{
	NONE,
	HDR_FILM,
	TILED_HDR_FILM,
	LDR_FILM,
	M_FILM
};

enum class EPixelFormat
{
	LUMINANCE, 
	LUMINANCE_ALPHA, 
	RGB, 
	RGBA,
	XYZ,
	XYZA,
	SPECTRUM, 
	SPECTRUM_ALPHA
};

struct SFilmMetadata
{
	SFilmMetadata() 
		:type(EFilmType::NONE) {};

	SFilmMetadata(const SFilmMetadata& other)
		:type(other.type), width(other.width), height(other.height),
		isCropUsed(other.isCropUsed), cropOffsetX(other.cropOffsetX), cropOffsetY(other.cropOffsetY),
		cropWidth(other.cropWidth), cropHeight(other.cropHeight), pixelFormat(other.pixelFormat)
	{
		switch (other.type)
		{
		case EFilmType::HDR_FILM:
			hdrFilmData = other.hdrFilmData;
			break;

		case EFilmType::TILED_HDR_FILM:
			tiledHdrFilmData = other.tiledHdrFilmData;
			break;

		case EFilmType::LDR_FILM:
			ldrFilmData = other.ldrFilmData;
			break;

		case EFilmType::M_FILM:
			mFilmData = other.mFilmData;
			break;
		}
	}

	SFilmMetadata& operator=(const SFilmMetadata& other)
	{
		type = other.type;
		width = other.width; 
		height = other.height;
		isCropUsed = other.isCropUsed;
		cropOffsetX = other.cropOffsetX; 
		cropOffsetY = other.cropOffsetY;
		cropWidth = other.cropWidth; 
		cropHeight = other.cropHeight; 
		pixelFormat = other.pixelFormat;

		switch (other.type)
		{
		case EFilmType::HDR_FILM:
			hdrFilmData = other.hdrFilmData;
			break;

		case EFilmType::TILED_HDR_FILM:
			tiledHdrFilmData = other.tiledHdrFilmData;
			break;

		case EFilmType::LDR_FILM:
			ldrFilmData = other.ldrFilmData;
			break;

		case EFilmType::M_FILM:
			mFilmData = other.mFilmData;
			break;
		}

		return *this;
	}

	~SFilmMetadata()
	{

	}

	EFilmType type;

	int width = 768;
	int height = 576;

	bool isCropUsed = false;
	int cropOffsetX = 0.0f;
	int cropOffsetY = 0.0f;
	int cropWidth = 0.0f;
	int cropHeight = 0.0f;

	//rfilter

	EPixelFormat pixelFormat = EPixelFormat::RGB;
	
	union
	{
		struct HdrFilmData
		{
			EHDRFileFormat fileFormat;
			EComponentFormat componentFormat;
			bool attachLog;
			bool banner;
			bool highQualityEdges;
		} hdrFilmData;

		struct TiledHdrFilmData
		{
			EComponentFormat componentFormat;
		} tiledHdrFilmData;

		struct LdrFilmData
		{
			ELDRFileFormat fileFormat;
			ETonemapMethod tonemapMethod;
			float gamma;
			float exposure;
			float key;
			float burn;
			bool banner;
			bool highQualityEdges;

		} ldrFilmData;

		struct MFilmData
		{
			EMFileFormat fileFormat;
			int digits;
			std::string variable;
			bool highQualityEdges;
			
		} mFilmData;
	};
};

class CElementFilm : public IElement
{
public:
	virtual bool processAttributes(const char** _atts) override;
	virtual bool onEndTag(asset::IAssetManager* _assetManager) override;
	virtual IElement::Type getType() const override { return IElement::Type::FILM; }
	virtual std::string getLogName() const override { return "film"; }

	SFilmMetadata getMetadata() const { return data; };

private:
	bool processSharedDataProperty(const SPropertyElementData& _property);
	bool processHDRFilmProperties();
	bool processTiledHDRFilmProperties();
	bool processLDRFilmProperties();
	bool processMFilmProperties();

private:
	SFilmMetadata data;

};



}
}
}

#endif