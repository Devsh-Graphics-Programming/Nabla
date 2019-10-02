#ifndef __C_ELEMENT_BSDF_H_INCLUDED__
#define __C_ELEMENT_BSDF_H_INCLUDED__

#include "../../ext/MitsubaLoader/CElementTexture.h"

namespace irr
{
namespace ext
{
namespace MitsubaLoader
{

class CGlobalMitsubaMetadata;

class CElementBSDF : public IElement
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
	struct FloatOrTexture
	{
		SPropertyElementData value;
		CElementTexture* texture; // only used if value.type==INVALID
	};
	struct SpectrumOrTexture : FloatOrTexture
	{
	};
		struct DiffuseTransmitter
		{
			SpectrumOrTexture transmittance = 0.5f;
		};
		struct AllDiffuse
		{
			SpectrumOrTexture reflectance = 0.5f;
			SpectrumOrTexture alpha = 0.2f; // not the parameter from Oren-Nayar
			bool useFastApprox;
		};
	struct RoughSpecularBase
	{
		enum NormalDistributionFunction
		{
			BECKMANN,
			GGX,
			PHONG,
			ASHIKHMIN_SHIRLEY
		};

		RoughSpecularBase(float defaultAlpha) : distribution(BECKMANN)
		{
			alpha = defaultAlpha;
		}

		NormalDistributionFunction distribution;
		union
		{
			FloatOrTexture alpha;
			struct
			{
				FloatOrTexture alphaU;
				FloatOrTexture alphaV;
			};
		};
		SpectrumOrTexture specularReflectance = 1.f;
	};
	struct TransmissiveBase
	{
		static const core::unordered_map<std::string, float, core::CaseInsensitiveHash, core::CaseInsensitiveEquals> NamedIndicesOfRefraction;

		TransmissiveBase(float intIOR, float extIOR) : intIOR(_intIOR), extIOR(_extIOR), specularTransmittance(1.f) {}
		TransmissiveBase(const std::string& intIOR, const std::string& extIOR)
		{
			intIOR = NamedIndicesOfRefraction[intIOR];
			extIOR = NamedIndicesOfRefraction[extIOR];
			specularTransmittance = 1.f;
		}

		float intIOR;
		float extIOR;
		SpectrumOrTexture specularTransmittance;
	};
		struct AllDielectric : RoughSpecularBase, TransmissiveBase
		{
			AllDielectric() : RoughSpecularBase(0.1f), TransmissiveBase("bk7","air") {}
			AllDielectric(float intIOR, float extIOR) : RoughSpecularBase(0.1f), TransmissiveBase(intIOR,extIOR) {}
			AllDielectric(const std::string& intIOR, const std::string& extIOR) : RoughSpecularBase(0.1f), TransmissiveBase(intIOR,extIOR) {}
		};
		struct AllConductor : RoughSpecularBase
		{
			AllConductor() : AllConductor("cu") {}
			AllConductor(const std::string& material);
			AllConductor(SPropertyElementData&& _eta, SPropertyElementData&& _k) : eta(_eta), k(_k), eta(TransmissiveBase::NamedIndicesOfRefraction["air"]) {}

			SPropertyElementData eta,k;
			float extEta;
		};
		struct AllPlastic : RoughSpecularBase, TransmissiveBase
		{
			AllDielectric() : RoughSpecularBase(0.1f), TransmissiveBase("polypropylene", "air"), nonlinear(false) {}
			AllDielectric(float intIOR, float extIOR) : RoughSpecularBase(0.1f), TransmissiveBase(intIOR, extIOR), nonlinear(false) {}
			AllDielectric(const std::string& intIOR, const std::string& extIOR) : RoughSpecularBase(0.1f), TransmissiveBase(intIOR, extIOR), nonlinear(false) {}

			bool nonlinear;
			SpectrumOrTexture diffuseReflectance = 0.5f;
		};/*
		struct HanrahanKrueger
		{
			class CPhaseElement
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
		_IRR_STATIC_INLINE_CONSTEXPR size_t MaxChildCount = 32u;
		size_t childCount = 0u;
		CElementBSDF* bsdf[MaxChildCount] = nullptr;
	};
		struct AllCoating : RoughSpecularBase, TransmissiveBase, MetaBSDF
		{
			_IRR_STATIC_INLINE_CONSTEXPR size_t MaxChildCount = 1u;
			float thickness;
			SpectrumOrTexture sigmaA;
		};
		struct BumpMap : MetaBSDF
		{
			CElementTexture* texture;
		};
		struct MixtureBSDF : MetaBSDF
		{
			float weights[MetaBSDF::MaxChildCount] = 1.f;
		};
		struct BlendBSDF : MetaBSDF
		{
			_IRR_STATIC_INLINE_CONSTEXPR size_t MaxChildCount = 2u;
			FloatOrTexture weight = 0.5f;
		};
		struct Mask : MetaBSDF
		{
			_IRR_STATIC_INLINE_CONSTEXPR size_t MaxChildCount = 1u;
			SpectrumOrTexture opacity;
		};
		struct TwoSided : MetaBSDF
		{
			_IRR_STATIC_INLINE_CONSTEXPR size_t MaxChildCount = 1u;
		};
		// legacy and evil
		struct Phong
		{
			float exponent = 30.f;
			SpectrumOrTexture specularReflectance = 0.2f;
			SpectrumOrTexture diffuseReflectance = 0.5f;
		};
		struct Ward
		{
			enum Type
			{
				WARD,
				WARD_DUER,
				BALANCED
			};
			Type variant = BALANCED;
			FloatOrTexture alphaU = 0.1f;
			FloatOrTexture alphaV = 0.1f;
			SpectrumOrTexture specularReflectance = 0.2f;
			SpectrumOrTexture diffuseReflectance = 0.5f;
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
					IRR_FALLTHROUGH;
				case CElementBSDF::Type::ROUGHDIFFUSE:
					diffuse = other.diffuse;
					break;
				case CElementBSDF::Type::DIELECTRIC:
					IRR_FALLTHROUGH;
				case CElementBSDF::Type::THINDIELECTRIC:
					IRR_FALLTHROUGH;
				case CElementBSDF::Type::ROUGHDIELECTRIC:
					dieletric = other.dielectric;
					break;
				case CElementBSDF::Type::CONDUCTOR:
					IRR_FALLTHROUGH;
				case CElementBSDF::Type::ROUGHCONDUCTOR:
					conductor = other.conductor;
					break;
				case CElementBSDF::Type::PLASTIC:
					IRR_FALLTHROUGH;
				case CElementBSDF::Type::ROUGHPLASTIC:
					plastic = other.plastic;
					break;
				case CElementBSDF::Type::COATING:
					IRR_FALLTHROUGH;
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

		bool addProperty(SPropertyElementData&& _property) override;
		bool onEndTag(asset::IAssetLoader::IAssetLoaderOverride* _override, CGlobalMitsubaMetadata* globalMetadata) override;
		IElement::Type getType() const override { return IElement::Type::BSDF; }
		std::string getLogName() const override { return "bsdf"; }

		bool processChildData(IElement* _child) override;

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
			//HanrahanKrueger	hk;
			//IrawanMarschner	irawan;
		};
};



}
}
}

#endif