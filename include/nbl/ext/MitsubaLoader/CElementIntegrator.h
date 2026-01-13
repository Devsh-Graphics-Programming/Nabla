// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXT_MISTUBA_LOADER_C_ELEMENT_INTEGRATOR_H_INCLUDED_
#define _NBL_EXT_MISTUBA_LOADER_C_ELEMENT_INTEGRATOR_H_INCLUDED_


#include "nbl/ext/MitsubaLoader/IElement.h"


namespace nbl::ext::MitsubaLoader
{


class CElementIntegrator final : public IElement
{
	public:
		enum Type : uint8_t
		{
			AO,
			DIRECT,
			PATH,
			VOL_PATH_SIMPLE,
			VOL_PATH,
			BDPT,
			PHOTONMAPPER,
			PPM,
			SPPM,
			PSSMLT,
			MLT,
			ERPT,
			ADJ_P_TRACER,
			ADAPTIVE,
			VPL,
			IRR_CACHE,
			MULTI_CHANNEL,
			FIELD_EXTRACT,
			INVALID
		};
		static inline core::unordered_map<core::string,Type,core::CaseInsensitiveHash,core::CaseInsensitiveEquals> compStringToTypeMap()
		{
			return {
				{"ao",				Type::AO},
				{"direct",			Type::DIRECT},
				{"path",			Type::PATH},
				{"volpath_simple",	Type::VOL_PATH_SIMPLE},
				{"volpath",			Type::VOL_PATH},
				{"bdpt",			Type::BDPT},
				{"photonmapper",	Type::PHOTONMAPPER},
				{"ppm",				Type::PPM},
				{"sppm",			Type::SPPM},
				{"pssmlt",			Type::PSSMLT},
				{"mlt",				Type::MLT},
				{"erpt",			Type::ERPT},
				{"ptracer",			Type::ADJ_P_TRACER},
				{"adaptive",		Type::ADAPTIVE},
				{"vpl",				Type::VPL},
				{"irrcache",		Type::IRR_CACHE},
				{"multichannel",	Type::MULTI_CHANNEL},
				{"field",			Type::FIELD_EXTRACT}
			};
		}

		struct AmbientOcclusion final
		{
			constexpr static inline Type VariantType = Type::AO;

			int32_t shadingSamples = 1;
			float rayLength = -1.f;
		};
	struct EmitterHideableBase
	{
		bool hideEmitters = false;
		bool hideEnvironment = false;
	};
		struct DirectIllumination final : EmitterHideableBase
		{
			constexpr static inline Type VariantType = Type::DIRECT;

			int32_t emitterSamples = static_cast<int32_t>(0xdeadbeefu);
			int32_t bsdfSamples = static_cast<int32_t>(0xdeadbeefu);
			bool strictNormals = false;
		};
	struct MonteCarloTracingBase
	{
		int32_t maxPathDepth = -1; // -1 is infinite
		int32_t russianRouletteDepth = 5;
	};
		struct PathTracing : MonteCarloTracingBase,EmitterHideableBase
		{
			bool strictNormals = false;
		};
		struct UniDirectionalPathTracing final : PathTracing
		{
			constexpr static inline Type VariantType = Type::PATH;
		};
		struct SimpleVolumetricPathTracing : PathTracing
		{
			constexpr static inline Type VariantType = Type::VOL_PATH_SIMPLE;
		};
		struct ExtendedVolumetricPathTracing : SimpleVolumetricPathTracing
		{
			constexpr static inline Type VariantType = Type::VOL_PATH;
		};
		struct BiDirectionalPathTracing final : PathTracing
		{
			constexpr static inline Type VariantType = Type::BDPT;

			bool lightImage = true;
			bool sampleDirect = true;
		};
	struct PhotonMappingBase : MonteCarloTracingBase
	{
		int32_t granularity = 0;
	};
		struct PhotonMapping final : PhotonMappingBase, EmitterHideableBase
		{
			constexpr static inline Type VariantType = Type::PHOTONMAPPER;

			int32_t directSamples = 16;
			int32_t glossySamples = 32;
			int32_t globalPhotons = 250000;
			int32_t causticPhotons = 250000;
			int32_t volumePhotons = 250000;
			float globalLookupRadius = 0.05;
			float causticLookupRadius = 0.0125;
			int32_t lookupSize = 120;
		};
		struct ProgressivePhotonMapping : PhotonMappingBase
		{
			constexpr static inline Type VariantType = Type::PPM;

			int32_t photonCount = 250000;
			float initialRadius = 0.f;
			float alpha = 0.7f;
			int32_t maxPasses = -1;
		};
		struct StochasticProgressivePhotonMapping final : ProgressivePhotonMapping
		{
			constexpr static inline Type VariantType = Type::SPPM;
		};
	struct MetropolisLightTransportBase : MonteCarloTracingBase
	{
		int32_t directSamples = 16;
		int32_t luminanceSamples = 100000;
		bool twoStage = false;
	};
		struct PrimarySampleSpaceMetropolisLightTransport final : MetropolisLightTransportBase
		{
			constexpr static inline Type VariantType = Type::PSSMLT;

			bool bidirectional = true;
			float pLarge = 0.3f;
		};
	struct PerturbateableBase
	{
		bool lensPerturbation = true;
		bool multiChainPerturbation = true;
		bool causticPerturbation = true;
		bool manifoldPerturbation = false;
		float lambda = 50.f;
	};
		struct PathSpaceMetropolisLightTransport final : MetropolisLightTransportBase, PerturbateableBase
		{
			constexpr static inline Type VariantType = Type::MLT;

			bool bidirectionalMutation = true;
		};
		struct EnergyRedistributionPathTracing final : MonteCarloTracingBase, PerturbateableBase
		{
			constexpr static inline Type VariantType = Type::ERPT;

			float numChains = 1.f;
			float maxChains = 0.f;
			int32_t chainLength = 1;
			int32_t directSamples = 16;
		};
		struct AdjointParticleTracing final : MonteCarloTracingBase
		{
			constexpr static inline Type VariantType = Type::ADJ_P_TRACER;

			uint32_t granularity = 200000;
			bool bruteForce = false;
		};
		struct VirtualPointLights final
		{
			constexpr static inline Type VariantType = Type::VPL;

			int32_t maxPathDepth = 5;
			int32_t shadowMap = 512;
			float clamping = 0.1f;
		};
		struct FieldExtraction
		{
			constexpr static inline Type VariantType = Type::FIELD_EXTRACT;

			enum Type
			{
				INVALID,
				POSITION,
				RELATIVE_POSITION,
				DISTANCE,
				GEOMETRIC_NORMAL,
				SHADING_NORMAL,
				UV_COORD,
				ALBEDO,
				SHAPE_INDEX,
				PRIMITIVE_INDEX
			};

			FieldExtraction() : field(Type::INVALID)
			{
				undefined.type = SPropertyElementData::Type::INVALID;
			}

			Type field;
			// TODO: Specral properties really need a redo
			SPropertyElementData undefined;
		};
	struct MetaIntegrator
	{
		_NBL_STATIC_INLINE_CONSTEXPR size_t maxChildCount = 3u*(FieldExtraction::Type::PRIMITIVE_INDEX+Type::FIELD_EXTRACT);
		size_t childCount = 0u;
		CElementIntegrator* children[maxChildCount] = { nullptr };
	};
		struct AdaptiveIntegrator final : MetaIntegrator
		{
			constexpr static inline Type VariantType = Type::ADAPTIVE;

			float maxError = 0.05f;
			float pValue = 0.05f;
			int32_t maxSampleFactor = 32;
		};
		struct IrradianceCacheIntegrator final : MetaIntegrator
		{
			constexpr static inline Type VariantType = Type::IRR_CACHE;

			int32_t resolution = 14;
			float quality = 1.f;
			bool gradients = true;
			bool clampNeighbour = true;
			bool clampScreen = true;
			bool overture = true;
			float qualityAdjustment = 0.5f;
			bool indirectOnly = false;
			bool debug = false;
		};
		struct MultiChannelIntegrator final : MetaIntegrator
		{
			constexpr static inline Type VariantType = Type::MULTI_CHANNEL;
		};

		//
		using variant_list_t = core::type_list<
			AmbientOcclusion,
			DirectIllumination,
			UniDirectionalPathTracing,
			SimpleVolumetricPathTracing,
			ExtendedVolumetricPathTracing,
			BiDirectionalPathTracing,
			PhotonMapping,
			ProgressivePhotonMapping,
			StochasticProgressivePhotonMapping,
			PrimarySampleSpaceMetropolisLightTransport,
			PathSpaceMetropolisLightTransport,
			EnergyRedistributionPathTracing,
			AdjointParticleTracing,
			AdaptiveIntegrator,
			VirtualPointLights,
			IrradianceCacheIntegrator,
			MultiChannelIntegrator,
			FieldExtraction
		>;
		//
		static AddPropertyMap<CElementIntegrator> compAddPropertyMap();

		//
		inline CElementIntegrator(const char* id) : IElement(id), type(Type::INVALID)
		{
		}
		inline ~CElementIntegrator()
		{
		}

		template<typename Visitor>
		inline void visit(Visitor&& visitor)
		{
			switch (type)
			{
				case CElementIntegrator::Type::AO:
					visitor(ao);
					break;
				case CElementIntegrator::Type::DIRECT:
					visitor(direct);
					break;
				case CElementIntegrator::Type::PATH:
					visitor(path);
					break;
				case CElementIntegrator::Type::VOL_PATH_SIMPLE:
					visitor(volpath_simple);
					break;
				case CElementIntegrator::Type::VOL_PATH:
					visitor(volpath);
					break;
				case CElementIntegrator::Type::BDPT:
					visitor(bdpt);
					break;
				case CElementIntegrator::Type::PHOTONMAPPER:
					visitor(photonmapper);
					break;
				case CElementIntegrator::Type::PPM:
					visitor(ppm);
					break;
				case CElementIntegrator::Type::SPPM:
					visitor(sppm);
					break;
				case CElementIntegrator::Type::PSSMLT:
					visitor(pssmlt);
					break;
				case CElementIntegrator::Type::MLT:
					visitor(mlt);
					break;
				case CElementIntegrator::Type::ERPT:
					visitor(erpt);
					break;
				case CElementIntegrator::Type::ADJ_P_TRACER:
					visitor(ptracer);
					break;
				case CElementIntegrator::Type::ADAPTIVE:
					visitor(adaptive);
					break;
				case CElementIntegrator::Type::VPL:
					visitor(vpl);
					break;
				case CElementIntegrator::Type::IRR_CACHE:
					visitor(irrcache);
					break;
				case CElementIntegrator::Type::MULTI_CHANNEL:
					visitor(multichannel);
					break;
				case CElementIntegrator::Type::FIELD_EXTRACT:
					visitor(field);
					break;
				default:
					break;
			}
		}
		template<typename Visitor>
		inline void visit(Visitor&& visitor) const
		{
			const_cast<CElementIntegrator*>(this)->visit([&]<typename T>(T& var)->void
				{
					visitor(const_cast<const T&>(var));
				}
			);
		}

		inline CElementIntegrator& operator=(const CElementIntegrator& other)
		{
			IElement::operator=(other);
			type = other.type;
			IElement::copyVariant(this,&other);
			return *this;
		}

		bool onEndTag(CMitsubaMetadata* globalMetadata, system::logger_opt_ptr logger) override;

		constexpr static inline auto ElementType = IElement::Type::INTEGRATOR;
		inline IElement::Type getType() const override { return ElementType; }
		inline std::string getLogName() const override { return "integrator"; }

		inline bool processChildData(IElement* _child, const std::string& name, system::logger_opt_ptr logger) override
		{
			if (!_child)
				return true;

			switch (type)
			{
				case Type::IRR_CACHE:
					[[fallthrough]];
				case Type::MULTI_CHANNEL:
					if (_child->getType() == IElement::Type::INTEGRATOR)
						break;
					[[fallthrough]];
				default:
					logger.log("Only IrradianceCache or MultiChannel <integrator> can only have another <integrator> nested inside", system::ILogger::ELL_ERROR);
					return false;
			}
			switch (type)
			{
				case Type::IRR_CACHE:
					irrcache.children[0u] = static_cast<CElementIntegrator*>(_child);
					irrcache.childCount = 1u;
					return true;
					break;
				case Type::MULTI_CHANNEL:
					if (irrcache.childCount < MetaIntegrator::maxChildCount)
					{
						irrcache.children[irrcache.childCount++] = static_cast<CElementIntegrator*>(_child);
						return true;
					}
					break;
				default: // to make compiler shut up
					assert(false);
					break;
			}
			return false;
		}

		//
		Type type;
		union
		{
			AmbientOcclusion							ao;
			DirectIllumination							direct;
			PathTracing									path;
			SimpleVolumetricPathTracing					volpath_simple;
			ExtendedVolumetricPathTracing				volpath;
			BiDirectionalPathTracing					bdpt;
			PhotonMapping								photonmapper;
			ProgressivePhotonMapping					ppm;
			StochasticProgressivePhotonMapping			sppm;
			PrimarySampleSpaceMetropolisLightTransport	pssmlt;
			PathSpaceMetropolisLightTransport			mlt;
			EnergyRedistributionPathTracing				erpt;
			AdjointParticleTracing						ptracer;
			AdaptiveIntegrator							adaptive;
			VirtualPointLights							vpl;
			IrradianceCacheIntegrator					irrcache;
			MultiChannelIntegrator						multichannel;
			FieldExtraction								field;
		};
};


}
#endif