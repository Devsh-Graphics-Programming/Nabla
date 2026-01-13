// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXT_MISTUBA_LOADER_C_ELEMENT_BSDF_H_INCLUDED_
#define _NBL_EXT_MISTUBA_LOADER_C_ELEMENT_BSDF_H_INCLUDED_


#include "nbl/ext/MitsubaLoader/CElementTexture.h"


namespace nbl::ext::MitsubaLoader
{

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
			NORMALMAP,
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
		static inline core::unordered_map<core::string,Type,core::CaseInsensitiveHash,core::CaseInsensitiveEquals> compStringToTypeMap()
		{
			return {
				{"diffuse",			CElementBSDF::Type::DIFFUSE},
				{"roughdiffuse",	CElementBSDF::Type::ROUGHDIFFUSE},
				{"dielectric",		CElementBSDF::Type::DIELECTRIC},
				{"thindielectric",	CElementBSDF::Type::THINDIELECTRIC},
				{"roughdielectric",	CElementBSDF::Type::ROUGHDIELECTRIC},
				{"conductor",		CElementBSDF::Type::CONDUCTOR},
				{"roughconductor",	CElementBSDF::Type::ROUGHCONDUCTOR},
				{"plastic",			CElementBSDF::Type::PLASTIC},
				{"roughplastic",	CElementBSDF::Type::ROUGHPLASTIC},
				{"coating",			CElementBSDF::Type::COATING},
				{"roughcoating",	CElementBSDF::Type::ROUGHCOATING},
				{"bumpmap",			CElementBSDF::Type::BUMPMAP},
				{"normalmap",		CElementBSDF::Type::NORMALMAP},
				{"phong",			CElementBSDF::Type::PHONG},
				{"ward",			CElementBSDF::Type::WARD},
				{"mixturebsdf",		CElementBSDF::Type::MIXTURE_BSDF},
				{"blendbsdf",		CElementBSDF::Type::BLEND_BSDF},
				{"mask",			CElementBSDF::Type::MASK},
				{"twosided",		CElementBSDF::Type::TWO_SIDED},
				{"difftrans",		CElementBSDF::Type::DIFFUSE_TRANSMITTER}//,
				//{"hk",				CElementBSDF::Type::HANRAHAN_KRUEGER},
				//{"irawan",			CElementBSDF::Type::IRAWAN_MARSCHNER}
			};
		}

		struct DiffuseTransmitter
		{
			constexpr static inline Type VariantType = Type::DIFFUSE_TRANSMITTER;

			inline DiffuseTransmitter() : transmittance(0.5f) {}

			inline DiffuseTransmitter& operator=(const DiffuseTransmitter& other)
			{
				transmittance = other.transmittance;
				return *this;
			}

			CElementTexture::SpectrumOrTexture transmittance;
		};
		struct AllDiffuse
		{
			inline AllDiffuse() : reflectance(0.5f), alpha(0.2f), useFastApprox(false) {}
			inline ~AllDiffuse() {}

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
		struct Diffuse : AllDiffuse
		{
			constexpr static inline Type VariantType = Type::DIFFUSE;
		};
		struct RoughDiffuse : AllDiffuse
		{
			constexpr static inline Type VariantType = Type::ROUGHDIFFUSE;
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

		inline RoughSpecularBase(float defaultAlpha) : distribution(GGX), specularReflectance(1.f),
			// union ignores ctors, and ctors are important to not try to free garbage strings
			alphaU(core::nan<float>()), alphaV(core::nan<float>())
		{
			alpha = defaultAlpha;
		}
		inline ~RoughSpecularBase() {}

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
		struct AllConductor : RoughSpecularBase
		{
			inline AllConductor() : AllConductor("cu",nullptr) {}
			inline AllConductor(const std::string_view material, system::logger_opt_ptr logger);
			inline AllConductor(SPropertyElementData&& _eta, SPropertyElementData&& _k, system::logger_opt_ptr logger) :
				RoughSpecularBase(0.1f), eta(_eta), k(_k), extEta(TransmissiveBase::findIOR("air",logger)) {}

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
		struct Conductor : AllConductor
		{
			constexpr static inline Type VariantType = Type::CONDUCTOR;
		};
		struct RoughConductor : AllConductor
		{
			constexpr static inline Type VariantType = Type::ROUGHCONDUCTOR;
		};
	struct TransmissiveBase
	{
		static float findIOR(const std::string_view name, system::logger_opt_ptr logger);

		inline TransmissiveBase(float _intIOR, float _extIOR) : intIOR(_intIOR), extIOR(_extIOR), specularTransmittance(1.f) {}
		inline TransmissiveBase(const std::string_view _intIOR, const std::string_view _extIOR, system::logger_opt_ptr logger) :
			TransmissiveBase(findIOR(_intIOR,logger), findIOR(_extIOR,logger)) {}

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
			inline AllDielectric() : RoughSpecularBase(0.1f), TransmissiveBase("bk7","air",nullptr) {}
			inline AllDielectric(float intIOR, float extIOR) : RoughSpecularBase(0.1f), TransmissiveBase(intIOR,extIOR) {}
			inline AllDielectric(const std::string_view intIOR, const std::string_view extIOR, system::logger_opt_ptr logger) :
				RoughSpecularBase(0.1f), TransmissiveBase(intIOR,extIOR,logger) {}

			inline AllDielectric& operator=(const AllDielectric& other)
			{
				RoughSpecularBase::operator=(other);
				TransmissiveBase::operator=(other);
				return *this;
			}
		};
		struct Dielectric : AllDielectric
		{
			constexpr static inline Type VariantType = Type::DIELECTRIC;
		};
		struct ThinDielectric : AllDielectric
		{
			constexpr static inline Type VariantType = Type::THINDIELECTRIC;
		};
		struct RoughDielectric : AllDielectric
		{
			constexpr static inline Type VariantType = Type::ROUGHDIELECTRIC;
		};
		struct AllPlastic : RoughSpecularBase, TransmissiveBase
		{
			inline AllPlastic() : RoughSpecularBase(0.1f), TransmissiveBase("polypropylene","air",nullptr), nonlinear(false) {}
			inline AllPlastic(float intIOR, float extIOR) :
				RoughSpecularBase(0.1f), TransmissiveBase(intIOR,extIOR), nonlinear(false) {}
			inline AllPlastic(const std::string_view intIOR, const std::string_view extIOR, system::logger_opt_ptr logger) :
				RoughSpecularBase(0.1f), TransmissiveBase(intIOR,extIOR,logger), nonlinear(false) {}

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
		};
		struct Plastic : AllPlastic
		{
			constexpr static inline Type VariantType = Type::PLASTIC;
		};
		struct RoughPlastic : AllPlastic
		{
			constexpr static inline Type VariantType = Type::ROUGHPLASTIC;
		};/*
		struct HanrahanKrueger
		{
			class CPhaseElement
			{
			};
			HanrahanKrueger(const std::string_view material);
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
		constexpr static inline size_t MaxChildCount = 32u;
		size_t childCount = 0u;
		CElementBSDF* bsdf[MaxChildCount] = { nullptr };
	};
		struct AllCoating : MetaBSDF, RoughSpecularBase, TransmissiveBase
		{
			constexpr static inline size_t MaxChildCount = 1u;

			inline AllCoating() final : MetaBSDF(), RoughSpecularBase(0.1f),
				TransmissiveBase("bk7","air",nullptr), thickness(1.f), sigmaA(0.f) {}

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
		struct Coating final : AllCoating
		{
			constexpr static inline Type VariantType = Type::COATING;
		};
		struct RoughCoating final : AllCoating
		{
			constexpr static inline Type VariantType = Type::ROUGHCOATING;
		};
		struct BumpMap final : MetaBSDF
		{
			constexpr static inline Type VariantType = Type::BUMPMAP;

			CElementTexture* texture = nullptr;
		};
		struct NormalMap final : MetaBSDF
		{
			constexpr static inline Type VariantType = Type::NORMALMAP;

			CElementTexture* texture = nullptr;
		};
		struct MixtureBSDF final : MetaBSDF
		{
			constexpr static inline Type VariantType = Type::MIXTURE_BSDF;

			uint32_t weightCount = 0u;
			float weights[MetaBSDF::MaxChildCount] = { 1.f };
		};
		struct BlendBSDF final : MetaBSDF
		{
			constexpr static inline Type VariantType = Type::BLEND_BSDF;
			constexpr static inline size_t MaxChildCount = 2u;

			inline BlendBSDF() : weight(0.5f) {}

			CElementTexture::SpectrumOrTexture weight;
		};
		struct Mask final : MetaBSDF
		{
			constexpr static inline Type VariantType = Type::MASK;
			constexpr static inline size_t MaxChildCount = 1u;

			inline Mask() : opacity(0.5f) {}

			CElementTexture::SpectrumOrTexture opacity;
		};
		struct TwoSided final : MetaBSDF
		{
			constexpr static inline Type VariantType = Type::TWO_SIDED;
			constexpr static inline size_t MaxChildCount = 1u;
		};
		// legacy and evil
		struct Phong
		{
			constexpr static inline Type VariantType = Type::PHONG;

			CElementTexture::FloatOrTexture exponent = 30.f;
			CElementTexture::SpectrumOrTexture specularReflectance = 0.2f;
			CElementTexture::SpectrumOrTexture diffuseReflectance = 0.5f;
		};
		struct Ward
		{
			constexpr static inline Type VariantType = Type::WARD;

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
		
		//
		using variant_list_t = core::type_list<
			Diffuse,
			RoughDiffuse,
			Dielectric,
			ThinDielectric,
			RoughDielectric,
			Conductor,
			RoughConductor,
			Plastic,
			RoughPlastic,
			Coating,
			RoughCoating,
			BumpMap,
			NormalMap,
			Phong,
			Ward,
			MixtureBSDF,
			BlendBSDF,
			Mask,
			TwoSided,
			DiffuseTransmitter/*,
			HanrahanKrueger,
			IrawanMarschner*/
		>;
		//
		static AddPropertyMap<CElementBSDF> compAddPropertyMap();

		//
		inline CElementBSDF(const char* id) : IElement(id), type(Type::INVALID)
		{
		}
		inline CElementBSDF(const CElementBSDF& other) : IElement(other)
		{
			operator=(other);
		}
		virtual ~CElementBSDF()
		{
		}

		template<typename Visitor>
		inline void visit(Visitor&& func)
		{
			switch (type)
			{
				case CElementBSDF::Type::DIFFUSE:
					[[fallthrough]];
				case CElementBSDF::Type::ROUGHDIFFUSE:
					func(diffuse);
					break;
				case CElementBSDF::Type::DIELECTRIC:
					[[fallthrough]];
				case CElementBSDF::Type::THINDIELECTRIC:
					[[fallthrough]];
				case CElementBSDF::Type::ROUGHDIELECTRIC:
					func(dielectric);
					break;
				case CElementBSDF::Type::CONDUCTOR:
					[[fallthrough]];
				case CElementBSDF::Type::ROUGHCONDUCTOR:
					func(conductor);
					break;
				case CElementBSDF::Type::PLASTIC:
					[[fallthrough]];
				case CElementBSDF::Type::ROUGHPLASTIC:
					func(plastic);
					break;
				case CElementBSDF::Type::COATING:
					[[fallthrough]];
				case CElementBSDF::Type::ROUGHCOATING:
					func(coating);
					break;
				case CElementBSDF::Type::BUMPMAP:
					func(bumpmap);
					break;
				case CElementBSDF::Type::NORMALMAP:
					func(normalmap);
					break;
				case CElementBSDF::Type::PHONG:
					func(phong);
					break;
				case CElementBSDF::Type::WARD:
					func(ward);
					break;
				case CElementBSDF::Type::MIXTURE_BSDF:
					func(mixturebsdf);
					break;
				case CElementBSDF::Type::BLEND_BSDF:
					func(blendbsdf);
					break;
				case CElementBSDF::Type::MASK:
					func(mask);
					break;
				case CElementBSDF::Type::TWO_SIDED:
					func(twosided);
					break;
				case CElementBSDF::Type::DIFFUSE_TRANSMITTER:
					func(difftrans);
					break;
				//case CElementBSDF::Type::HANRAHAN_KRUEGER:
					//func(hk);
					//break;
				//case CElementBSDF::Type::IRAWAN_MARSCHNER:
					//func(irwan);
					//break;
				default:
					break;
			}
		}
		template<typename Visitor>
		inline void visit(Visitor&& visitor) const
		{
			const_cast<CElementBSDF*>(this)->visit([&]<typename T>(T& var)->void
				{
					visitor(const_cast<const T&>(var));
				}
			);
		}

		inline CElementBSDF& operator=(const CElementBSDF& other)
		{
			IElement::operator=(other);
			type = other.type;
			IElement::copyVariant(this,&other);
			return *this;
		}

		bool onEndTag(CMitsubaMetadata* globalMetadata, system::logger_opt_ptr logger) override;

		constexpr static inline auto ElementType = IElement::Type::BSDF;
		inline IElement::Type getType() const override { return ElementType; }
		std::string getLogName() const override { return "bsdf"; }

		bool processChildData(IElement* _child, const std::string& name, system::logger_opt_ptr logger) override;

		inline bool isMeta() const
		{
			switch (type)
			{
				case COATING: [[fallthrough]];
				case ROUGHCOATING: [[fallthrough]];
				case TWO_SIDED: [[fallthrough]];
				case MASK: [[fallthrough]];
				case BLEND_BSDF: [[fallthrough]];
				case MIXTURE_BSDF: [[fallthrough]];
				case BUMPMAP: [[fallthrough]];
				case NORMALMAP:
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
			NormalMap			normalmap;
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
#endif