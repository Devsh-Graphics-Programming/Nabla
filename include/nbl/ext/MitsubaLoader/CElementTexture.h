// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXT_MITSUBA_LOADER_C_ELEMENT_TEXTURE_H_INCLUDED_
#define _NBL_EXT_MITSUBA_LOADER_C_ELEMENT_TEXTURE_H_INCLUDED_

#include "nbl/ext/MitsubaLoader/PropertyElement.h"
#include "nbl/ext/MitsubaLoader/IElement.h"

namespace nbl::ext::MitsubaLoader
{

class CElementTexture : public IElement
{
	public:
		struct FloatOrTexture
		{
			FloatOrTexture(CElementTexture* _tex)
			{
				value.type = SPropertyElementData::Type::INVALID;
				texture = _tex;
			}
			FloatOrTexture(float _value)
			{
				value.type = SPropertyElementData::Type::FLOAT;
				value.fvalue = _value;
				texture = nullptr;
			}
			FloatOrTexture(const SPropertyElementData& _other) : FloatOrTexture(nullptr)
			{
				operator=(_other);
			}
			FloatOrTexture(SPropertyElementData&& _other) : FloatOrTexture(nullptr)
			{
				operator=(std::move(_other));
			}
			inline FloatOrTexture& operator=(const SPropertyElementData& _other)
			{
				return operator=(SPropertyElementData(_other));
			}
			inline FloatOrTexture& operator=(SPropertyElementData&& _other)
			{
				switch (_other.type)
				{
					case SPropertyElementData::Type::INVALID:
					case SPropertyElementData::Type::FLOAT:
						value = std::move(_other);
						break;
					default:
						_NBL_DEBUG_BREAK_IF(true);
						break;
				}
				return *this;
			}

			SPropertyElementData value;
			CElementTexture* texture; // only used if value.type==INVALID
		};
		struct SpectrumOrTexture : FloatOrTexture
		{
			SpectrumOrTexture(CElementTexture* _tex) : FloatOrTexture(_tex) {}
			SpectrumOrTexture(float _value) : FloatOrTexture(_value) {}
			SpectrumOrTexture(const SPropertyElementData& _other) : SpectrumOrTexture(nullptr)
			{
				operator=(_other);
			}
			SpectrumOrTexture(SPropertyElementData&& _other) : SpectrumOrTexture(nullptr)
			{
				operator=(std::move(_other));
			}
			inline SpectrumOrTexture& operator=(const SPropertyElementData& _other)
			{
				return operator=(SPropertyElementData(_other));
			}
			inline SpectrumOrTexture& operator=(SPropertyElementData&& _other)
			{
				switch (_other.type)
				{
					case SPropertyElementData::Type::INVALID:
					case SPropertyElementData::Type::FLOAT:
					case SPropertyElementData::Type::RGB:
					case SPropertyElementData::Type::SRGB:
					case SPropertyElementData::Type::SPECTRUM:
					case SPropertyElementData::Type::BLACKBODY:
						value = std::move(_other);
						break;
					default:
						_NBL_DEBUG_BREAK_IF(true);
						break;
				}
				return *this;
			}
		};

		enum Type
		{
			INVALID,
			BITMAP,
			//CHECKERBOARD,
			//GRID,
			SCALE,
			//VERTEXCOLOR,
			//WIREFRAME,
			//CURVATURE
		};
		struct Bitmap
		{
			enum WRAP_MODE
			{
				REPEAT,
				MIRROR,
				CLAMP,
				ZERO,
				ONE
			};
			enum FILTER_TYPE
			{
				EWA,
				TRILINEAR,
				NEAREST
			};
			enum CHANNEL
			{
				INVALID=0,
				R,
				G,
				B,
				A/*, needs special conversions
				X,
				Y,
				Z*/
			};

			SPropertyElementData filename;
			WRAP_MODE wrapModeU = REPEAT;
			WRAP_MODE wrapModeV = REPEAT;
			float gamma = NAN;
			FILTER_TYPE filterType = EWA;
			float maxAnisotropy = 20.f;
			//bool cache = false;
			float uoffset = 0.f;
			float voffset = 0.f;
			float uscale = 1.f;
			float vscale = 1.f;
			CHANNEL channel = INVALID;
		};
	struct MetaTexture
	{
		CElementTexture*	texture;
	};
		struct Scale : MetaTexture
		{
			float	scale;
		};

		CElementTexture(const char* id) : IElement(id), type(Type::INVALID)
		{
		}
		CElementTexture(const CElementTexture& other) : CElementTexture("")
		{
			operator=(other);
		}
		CElementTexture(CElementTexture&& other) : CElementTexture("")
		{
			operator=(std::move(other));
		}
		virtual ~CElementTexture()
		{
		}
		
		inline CElementTexture& operator=(const CElementTexture& other)
		{
			IElement::operator=(other);
			type = other.type;
			switch (type)
			{
				case CElementTexture::Type::BITMAP:
					bitmap = other.bitmap;
					break;
				//case CElementTexture::Type::CHECKERBOARD:
					//checkerboard = CheckerBoard();
					//break;
				//case CElementTexture::Type::GRID:
					//grid = Grid();
					//break;
				case CElementTexture::Type::SCALE:
					scale = other.scale;
					break;
				//case CElementTexture::Type::VERTEXCOLOR:
					//vertexcolor = VertexColor();
					//break;
				//case CElementTexture::Type::WIREFRAME:
					//wireframe = Wireframe();
					//break;
				//case CElementTexture::Type::CURVATURE:
					//curvature = Curvature();
					//break;
				default:
					break;
			}
			return *this;
		}
		inline CElementTexture& operator=(CElementTexture&& other)
		{
			IElement::operator=(other);
			type = other.type;
			switch (type)
			{
				case CElementTexture::Type::BITMAP:
					std::swap(bitmap,other.bitmap);
					break;
				//case CElementTexture::Type::CHECKERBOARD:
					//checkerboard = CheckerBoard();
					//break;
				//case CElementTexture::Type::GRID:
					//grid = Grid();
					//break;
				case CElementTexture::Type::SCALE:
					std::swap(scale,other.scale);
					break;
				//case CElementTexture::Type::VERTEXCOLOR:
					//vertexcolor = VertexColor();
					//break;
				//case CElementTexture::Type::WIREFRAME:
					//wireframe = Wireframe();
					//break;
				//case CElementTexture::Type::CURVATURE:
					//curvature = Curvature();
					//break;
				default:
					break;
			}
			return *this;
		}

		bool addProperty(SNamedPropertyElement&& _property) override;
		bool onEndTag(asset::IAssetLoader::IAssetLoaderOverride* _override, CMitsubaMetadata* globalMetadata) override;
		IElement::Type getType() const override { return IElement::Type::TEXTURE; }
		std::string getLogName() const override { return "texture"; }

		bool processChildData(IElement* _child, const std::string& name) override;

		//
		Type type;
		union
		{
			Bitmap		bitmap;
			Scale		scale;
		};
};

}

#endif