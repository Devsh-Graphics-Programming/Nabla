// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __C_ELEMENT_BSDF_H_INCLUDED__
#define __C_ELEMENT_BSDF_H_INCLUDED__

#include "nbl/ext/MitsubaLoader/CElementTexture.h"

namespace nbl
{
namespace ext
{
namespace MitsubaLoader
{


class NBL_API CElementBSDF : public IElement
{
	public:
		enum Type
		{
			INVALID,
			DIFFUSE, // Lambertian
			ROUGHDIFFUSE, // Oren-Nayar
			DIELECTRIC, // delta distributions with real IoR 
			THINDIELECTRIC,
			ROUGHDIELECTRIC, // distributions with real IoR
			CONDUCTOR,
			ROUGHCONDUCTOR,
			PLASTIC,
			ROUGHPLASTIC,
			COATING,
			ROUGHCOATING,
			BUMPMAP,
			PHONG,
			WARD,
			MIXTURE_BSDF,
			BLEND_BSDF,
			MASK,
			TWO_SIDED,
			DIFFUSE_TRANSMITTER,
			//HANRAHAN_KRUEGER,
			//IRAWAN_MARSCHNER
		};
		struct DiffuseTransmitter
		{
			DiffuseTransmitter() : transmittance(0.5f) {}

			inline DiffuseTransmitter& operator=(const DiffuseTransmitter& other)
			{
				transmittance = other.transmittance;
				return *this;
			}

			CElementTexture::SpectrumOrTexture transmittance;
		};
		struct AllDiffuse
		{
			AllDiffuse() : reflectance(0.5f), alpha(0.2f), useFastApprox(false) {}
			~AllDiffuse() {}

			inline AllDiffuse& operator=(const AllDiffuse& other)
			{
				reflectance = other.reflectance;
				alpha = other.alpha;
				useFastApprox = other.useFastApprox;
				return *this;
			}

			union // to support the unholy undocumented feature of Mitsuba
			{
				CElementTexture::SpectrumOrTexture	reflectance;
				CElementTexture::SpectrumOrTexture	diffuseReflectance;
			};
			CElementTexture::FloatOrTexture		alpha; // not the parameter from Oren-Nayar
			bool								useFastApprox;
		};
	struct RoughSpecularBase
	{
		enum NormalDistributionFunction : uint32_t
		{
			BECKMANN,
			GGX,
			PHONG,
			ASHIKHMIN_SHIRLEY
		};

		RoughSpecularBase(float defaultAlpha) : distribution(BECKMANN), specularReflectance(1.f)
		{
			alpha = defaultAlpha;
		}
		virtual ~RoughSpecularBase() {}

		inline RoughSpecularBase& operator=(const RoughSpecularBase& other)
		{
			distribution = other.distribution;
			switch (distribution)
			{
				case ASHIKHMIN_SHIRLEY:
					alphaU = other.alphaU;
					alphaV = other.alphaV;
					break;
				default:
					alpha = other.alpha;
					break;
			}
			specularReflectance = other.specularReflectance;
			return *this;
		}

		NormalDistributionFunction distribution;
		union
		{
			CElementTexture::FloatOrTexture alpha;
			struct
			{
				CElementTexture::FloatOrTexture alphaU;
				CElementTexture::FloatOrTexture alphaV;
			};
		};
		CElementTexture::SpectrumOrTexture specularReflectance;
	};
	struct TransmissiveBase
	{
		static float findIOR(const std::string& name);

		TransmissiveBase(float _intIOR, float _extIOR) : intIOR(_intIOR), extIOR(_extIOR), specularTransmittance(1.f) {}
		TransmissiveBase(const std::string& _intIOR, const std::string& _extIOR) : TransmissiveBase(findIOR(_intIOR), findIOR(_extIOR)) {}

		inline TransmissiveBase& operator=(const TransmissiveBase& other)
		{
			intIOR = other.intIOR;
			extIOR = other.extIOR;
			specularTransmittance = other.specularTransmittance;
			return *this;
		}

		float intIOR;
		float extIOR;
		CElementTexture::SpectrumOrTexture specularTransmittance;
	};
		struct AllDielectric : RoughSpecularBase, TransmissiveBase
		{
			AllDielectric() : RoughSpecularBase(0.1f), TransmissiveBase("bk7","air") {}
			AllDielectric(float intIOR, float extIOR) : RoughSpecularBase(0.1f), TransmissiveBase(intIOR,extIOR) {}
			AllDielectric(const std::string& intIOR, const std::string& extIOR) : RoughSpecularBase(0.1f), TransmissiveBase(intIOR,extIOR) {}

			inline AllDielectric& operator=(const AllDielectric& other)
			{
				RoughSpecularBase::operator=(other);
				TransmissiveBase::operator=(other);
				return *this;
			}
		};
		struct AllConductor : RoughSpecularBase
		{
			AllConductor() : AllConductor("cu") {}
			AllConductor(const std::string& material);
			AllConductor(SPropertyElementData&& _eta, SPropertyElementData&& _k) : RoughSpecularBase(0.1f), eta(_eta), k(_k), extEta(TransmissiveBase::findIOR("air")) {}

			inline AllConductor& operator=(const AllConductor& other)
			{
				RoughSpecularBase::operator=(other);
				eta = other.eta;
				k = other.k;
				extEta = other.extEta;
				return *this;
			}

			SPropertyElementData eta,k;
			float extEta;
		};
		struct AllPlastic : RoughSpecularBase, TransmissiveBase
		{
			AllPlastic() : RoughSpecularBase(0.1f), TransmissiveBase("polypropylene", "air"), nonlinear(false) {}
			AllPlastic(float intIOR, float extIOR) : RoughSpecularBase(0.1f), TransmissiveBase(intIOR, extIOR), nonlinear(false) {}
			AllPlastic(const std::string& intIOR, const std::string& extIOR) : RoughSpecularBase(0.1f), TransmissiveBase(intIOR, extIOR), nonlinear(false) {}

			inline AllPlastic& operator=(const AllPlastic& other)
			{
				RoughSpecularBase::operator=(other);
				TransmissiveBase::operator=(other);
				nonlinear = other.nonlinear;
				diffuseReflectance = other.diffuseReflectance;
				return *this;
			}

			bool nonlinear;
			CElementTexture::SpectrumOrTexture diffuseReflectance = 0.5f;
		};/*
		struct HanrahanKrueger
		{
			class NBL_API CPhaseElement
			{
			};
			HanrahanKrueger(const std::string& material);
			HanrahanKrueger() : HanrahanKrueger("skin1") {}

			bool tNOTs = false;
			union
			{
				struct
				{
					SpectrumOrTexture sigmaS;
					SpectrumOrTexture sigmaA;
				};
				struct 
				{
					SpectrumOrTexture sigmaT;
					SpectrumOrTexture albedo;
				};
			};
			float thickness = 1.f;
			CPhaseElement phase;
		};*/
	struct MetaBSDF
	{
		_NBL_STATIC_INLINE_CONSTEXPR size_t MaxChildCount = 32u;
		size_t childCount = 0u;
		CElementBSDF* bsdf[MaxChildCount] = { nullptr };
	};
		struct AllCoating : MetaBSDF, RoughSpecularBase, TransmissiveBase
		{
			_NBL_STATIC_INLINE_CONSTEXPR size_t MaxChildCount = 1u;

			AllCoating() : MetaBSDF(), RoughSpecularBase(0.1f), TransmissiveBase("bk7","air"), thickness(1.f), sigmaA(0.f) {}

			inline AllCoating& operator=(const AllCoating& other)
			{
				RoughSpecularBase::operator=(other);
				TransmissiveBase::operator=(other);
				MetaBSDF::operator=(other);
				thickness = other.thickness;
				sigmaA = other.sigmaA;
				return *this;
			}

			float thickness;
			CElementTexture::SpectrumOrTexture sigmaA;
		};
		struct BumpMap : MetaBSDF
		{
			CElementTexture* texture;
			bool wasNormal;
		};
		struct MixtureBSDF : MetaBSDF
		{
			uint32_t weightCount = 0u;
			float weights[MetaBSDF::MaxChildCount] = { 1.f };
		};
		struct BlendBSDF : MetaBSDF
		{
			_NBL_STATIC_INLINE_CONSTEXPR size_t MaxChildCount = 2u;

			BlendBSDF() : weight(0.5f) {}

			CElementTexture::FloatOrTexture weight;
		};
		struct Mask : MetaBSDF
		{
			_NBL_STATIC_INLINE_CONSTEXPR size_t MaxChildCount = 1u;

			Mask() : opacity(0.5f) {}

			CElementTexture::SpectrumOrTexture opacity;
		};
		struct TwoSided : MetaBSDF
		{
			_NBL_STATIC_INLINE_CONSTEXPR size_t MaxChildCount = 1u;
		};
		// legacy and evil
		struct Phong
		{
			CElementTexture::FloatOrTexture exponent = 30.f;
			CElementTexture::SpectrumOrTexture specularReflectance = 0.2f;
			CElementTexture::SpectrumOrTexture diffuseReflectance = 0.5f;
		};
		struct Ward
		{
			enum Type : uint32_t
			{
				WARD,
				WARD_DUER,
				BALANCED
			};
			Type variant = BALANCED;
			CElementTexture::FloatOrTexture alphaU = 0.1f;
			CElementTexture::FloatOrTexture alphaV = 0.1f;
			CElementTexture::SpectrumOrTexture specularReflectance = 0.2f;
			CElementTexture::SpectrumOrTexture diffuseReflectance = 0.5f;
		};

		CElementBSDF(const char* id) : IElement(id), type(Type::INVALID)
		{
		}
		virtual ~CElementBSDF()
		{
		}

		inline CElementBSDF& operator=(const CElementBSDF& other)
		{
			IElement::operator=(other);
			type = other.type;
			switch (type)
			{
				case CElementBSDF::Type::DIFFUSE:
					[[fallthrough]];
				case CElementBSDF::Type::ROUGHDIFFUSE:
					diffuse = other.diffuse;
					break;
				case CElementBSDF::Type::DIELECTRIC:
					[[fallthrough]];
				case CElementBSDF::Type::THINDIELECTRIC:
					[[fallthrough]];
				case CElementBSDF::Type::ROUGHDIELECTRIC:
					dielectric = other.dielectric;
					break;
				case CElementBSDF::Type::CONDUCTOR:
					[[fallthrough]];
				case CElementBSDF::Type::ROUGHCONDUCTOR:
					conductor = other.conductor;
					break;
				case CElementBSDF::Type::PLASTIC:
					[[fallthrough]];
				case CElementBSDF::Type::ROUGHPLASTIC:
					plastic = other.plastic;
					break;
				case CElementBSDF::Type::COATING:
					[[fallthrough]];
				case CElementBSDF::Type::ROUGHCOATING:
					coating = other.coating;
					break;
				case CElementBSDF::Type::BUMPMAP:
					bumpmap = other.bumpmap;
					break;
				case CElementBSDF::Type::PHONG:
					phong = other.phong;
					break;
				case CElementBSDF::Type::WARD:
					ward = other.ward;
					break;
				case CElementBSDF::Type::MIXTURE_BSDF:
					mixturebsdf = other.mixturebsdf;
					break;
				case CElementBSDF::Type::BLEND_BSDF:
					blendbsdf = other.blendbsdf;
					break;
				case CElementBSDF::Type::MASK:
					mask = other.mask;
					break;
				case CElementBSDF::Type::TWO_SIDED:
					twosided = other.twosided;
					break;
				case CElementBSDF::Type::DIFFUSE_TRANSMITTER:
					difftrans = other.difftrans;
					break;
				//case CElementBSDF::Type::HANRAHAN_KRUEGER:
					//hk = HanrahanKrueger();
					//break;
				//case CElementBSDF::Type::IRAWAN_MARSCHNER:
					//irawan = IrawanMarschner();
					//break;
				default:
					break;
			}
			return *this;
		}

		bool addProperty(SNamedPropertyElement&& _property) override;
		bool onEndTag(asset::IAssetLoader::IAssetLoaderOverride* _override, CMitsubaMetadata* globalMetadata) override;
		IElement::Type getType() const override { return IElement::Type::BSDF; }
		std::string getLogName() const override { return "bsdf"; }

		bool processChildData(IElement* _child, const std::string& name) override;

		bool isMeta() const
		{
			switch (type)
			{
			case COATING: [[fallthrough]];
			case ROUGHCOATING: [[fallthrough]];
			case TWO_SIDED: [[fallthrough]];
			case MASK: [[fallthrough]];
			case BLEND_BSDF: [[fallthrough]];
			case MIXTURE_BSDF: [[fallthrough]];
			case BUMPMAP:
				return true;
			default:
				return false;
			}
		}

		//
		Type type;
		union
		{
			AllDiffuse			diffuse;
			DiffuseTransmitter	difftrans;
			AllDielectric		dielectric;
			AllConductor		conductor;
			AllPlastic			plastic;
			AllCoating			coating;
			BumpMap				bumpmap;
			Phong				phong;
			Ward				ward;
			MixtureBSDF			mixturebsdf;
			BlendBSDF			blendbsdf;
			Mask				mask;
			TwoSided			twosided;
			//a not confusing way (extra union member) to access members common for all structs inheriting from MetaBSDF
			MetaBSDF			meta_common;
			//HanrahanKrueger	hk;
			//IrawanMarschner	irawan;
		};
};



}
}
}

#endif