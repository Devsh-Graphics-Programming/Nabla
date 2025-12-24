// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXT_MISTUBA_LOADER_C_ELEMENT_TEXTURE_H_INCLUDED_
#define _NBL_EXT_MISTUBA_LOADER_C_ELEMENT_TEXTURE_H_INCLUDED_


#include "nbl/ext/MitsubaLoader/PropertyElement.h"
#include "nbl/ext/MitsubaLoader/IElement.h"


namespace nbl::ext::MitsubaLoader
{

class CElementTexture : public IElement
{
	public:
		struct FloatOrTexture
		{
			inline FloatOrTexture(CElementTexture* _tex)
			{
				value = std::numeric_limits<float>::quiet_NaN();
				texture = _tex;
			}
			inline FloatOrTexture(const float _value)
			{
				value = _value;
				texture = nullptr;
			}
			inline FloatOrTexture(const FloatOrTexture&) = default;

			inline FloatOrTexture& operator=(const FloatOrTexture&) = default;

			float value = 0.f;
			CElementTexture* texture = nullptr;
		};
		struct SpectrumOrTexture
		{
			inline SpectrumOrTexture(CElementTexture* _tex)
			{
				value.type = SPropertyElementData::Type::INVALID;
				texture = _tex;
			}
			inline SpectrumOrTexture(const SPropertyElementData& _other) : SpectrumOrTexture(nullptr)
			{
				operator=(_other);
			}
			inline SpectrumOrTexture(SPropertyElementData&& _other) : SpectrumOrTexture(nullptr)
			{
				operator=(std::move(_other));
			}
			inline SpectrumOrTexture(const float _value) : SpectrumOrTexture(SPropertyElementData{_value}) {}

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
				texture = nullptr;
				return *this;
			}

			SPropertyElementData value = {};
			CElementTexture* texture = nullptr;
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
		static inline core::unordered_map<core::string,Type,core::CaseInsensitiveHash,core::CaseInsensitiveEquals> compStringToTypeMap()
		{
			return {
				{"bitmap",			CElementTexture::Type::BITMAP},
				{"scale",			CElementTexture::Type::SCALE}
			};
		}

		struct Bitmap
		{
			constexpr static inline Type VariantType = Type::BITMAP;
			constexpr static inline uint16_t MaxPathLen = 1024u;

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

			char		filename[MaxPathLen];
			WRAP_MODE	wrapModeU = REPEAT;
			WRAP_MODE	wrapModeV = REPEAT;
			float		gamma = NAN;
			FILTER_TYPE filterType = EWA;
			float		maxAnisotropy = 20.f;
			//bool cache = false;
			float		uoffset = 0.f;
			float		voffset = 0.f;
			float		uscale = 1.f;
			float		vscale = 1.f;
			CHANNEL		channel = INVALID;
		};
	struct MetaTexture
	{
		CElementTexture*	texture;
	};
		struct Scale : MetaTexture
		{
			constexpr static inline Type VariantType = Type::SCALE;

			// only monochrome scaling for now!
			float scale = 1.f;
		};

		//
		using variant_list_t = core::type_list<Bitmap,Scale>;
		//
		static AddPropertyMap<CElementTexture> compAddPropertyMap();

		//
		inline CElementTexture(const char* id) : IElement(id), type(Type::INVALID)
		{
		}
		inline CElementTexture(const CElementTexture& other) : CElementTexture("")
		{
			operator=(other);
		}
		inline virtual ~CElementTexture()
		{
		}

		template<typename Visitor>
		inline void visit(Visitor&& func)
		{
			switch (type)
			{
				case CElementTexture::Type::BITMAP:
					func(bitmap);
					break;
				case CElementTexture::Type::SCALE:
					func(scale);
					break;
				default:
					break;
			}
		}
		template<typename Visitor>
		inline void visit(Visitor&& visitor) const
		{
			const_cast<CElementTexture*>(this)->visit([&]<typename T>(T& var)->void
				{
					visitor(const_cast<const T&>(var));
				}
			);
		}
		
		inline CElementTexture& operator=(const CElementTexture& other)
		{
			IElement::operator=(other);
			type = other.type;
			IElement::copyVariant(this,&other);
			return *this;
		}

		bool onEndTag(CMitsubaMetadata* globalMetadata, system::logger_opt_ptr logger) override;

		constexpr static inline auto ElementType = IElement::Type::TEXTURE;
		inline IElement::Type getType() const override { return ElementType; }
		inline std::string getLogName() const override { return "texture"; }

		bool processChildData(IElement* _child, const std::string& name, system::logger_opt_ptr logger) override;

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