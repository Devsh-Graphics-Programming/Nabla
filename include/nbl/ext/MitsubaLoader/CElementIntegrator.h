// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __C_ELEMENT_INTEGRATOR_H_INCLUDED__
#define __C_ELEMENT_INTEGRATOR_H_INCLUDED__

#include "nbl/ext/MitsubaLoader/IElement.h"

namespace nbl
{
namespace ext
{
namespace MitsubaLoader
{


class NBL_API CElementIntegrator : public IElement
{
	public:
		enum Type
		{
			INVALID,
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
			FIELD_EXTRACT
		};
		struct AmbientOcclusion
		{
			int32_t shadingSamples = 1;
			float rayLength = -1.f;
		};
	struct EmitterHideableBase
	{
		bool hideEmitters = false;
	};
		struct DirectIllumination : EmitterHideableBase
		{
			int32_t emitterSamples = 0xdeadbeefu;
			int32_t bsdfSamples = 0xdeadbeefu;
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
		struct SimpleVolumetricPathTracing : PathTracing
		{
		};
		struct ExtendedVolumetricPathTracing : SimpleVolumetricPathTracing
		{
		};
		struct BiDirectionalPathTracing : MonteCarloTracingBase
		{
			bool lightImage = true;
			bool sampleDirect = true;
		};
	struct PhotonMappingBase : MonteCarloTracingBase
	{
		int32_t granularity = 0;
	};
		struct PhotonMapping : PhotonMappingBase, EmitterHideableBase
		{
			int32_t directSamples = 16;
			int32_t glossySamples = 32;
			int32_t globalPhotons = 250000;
			int32_t causticPhotons = 250000;
			int32_t volumePhotons = 250000;
			float globalLURadius = 0.05;
			float causticLURadius = 0.0125;
			int32_t LUSize = 120;
		};
		struct ProgressivePhotonMapping : PhotonMappingBase
		{
			int32_t photonCount = 250000;
			float initialRadius = 0.f;
			float alpha = 0.7f;
			int32_t maxPasses = -1;
		};
		struct StochasticProgressivePhotonMapping : ProgressivePhotonMapping
		{
		};
	struct MetropolisLightTransportBase : MonteCarloTracingBase
	{
		int32_t directSamples = 16;
		int32_t luminanceSamples = 100000;
		bool twoStage = false;
	};
		struct PrimarySampleSpaceMetropolisLightTransport : MetropolisLightTransportBase
		{
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
		struct PathSpaceMetropolisLightTransport : MetropolisLightTransportBase, PerturbateableBase
		{
			bool bidirectionalMutation = true;
		};
		struct EnergyRedistributionPathTracing : MonteCarloTracingBase, PerturbateableBase
		{
			float numChains = 1.f;
			float maxChains = 0.f;
			int32_t chainLength = 1;
			int32_t directSamples = 16;
		};
		struct AdjointParticleTracing : MonteCarloTracingBase
		{
			uint32_t granularity = 200000;
			bool bruteForce = false;
		};
		struct VirtualPointLights
		{
			int32_t maxPathDepth = 5;
			int32_t shadowMap = 512;
			float clamping = 0.1f;
		};
		struct FieldExtraction
		{
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
			SPropertyElementData undefined;
		};
	struct MetaIntegrator
	{
		_NBL_STATIC_INLINE_CONSTEXPR size_t maxChildCount = 3u*(FieldExtraction::Type::PRIMITIVE_INDEX+Type::FIELD_EXTRACT);
		size_t childCount = 0u;
		CElementIntegrator* children[maxChildCount] = { nullptr };
	};
		struct AdaptiveIntegrator : MetaIntegrator
		{
			float maxError = 0.05f;
			float pValue = 0.05f;
			int32_t maxSampleFactor = 32;
		};
		struct IrradianceCacheIntegrator : MetaIntegrator
		{
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
		struct MultiChannelIntegrator : MetaIntegrator
		{
		};

		CElementIntegrator(const char* id) : IElement(id), type(Type::INVALID)
		{
		}
		virtual ~CElementIntegrator()
		{
		}

		inline CElementIntegrator& operator=(const CElementIntegrator& other)
		{
			IElement::operator=(other);
			type = other.type;
			switch (type)
			{
				case CElementIntegrator::Type::AO:
					ao = other.ao;
					break;
				case CElementIntegrator::Type::DIRECT:
					direct = other.direct;
					break;
				case CElementIntegrator::Type::PATH:
					path = other.path;
					break;
				case CElementIntegrator::Type::VOL_PATH_SIMPLE:
					volpath_simple = other.volpath_simple;
					break;
				case CElementIntegrator::Type::VOL_PATH:
					volpath = other.volpath;
					break;
				case CElementIntegrator::Type::BDPT:
					bdpt = other.bdpt;
					break;
				case CElementIntegrator::Type::PHOTONMAPPER:
					photonmapper = other.photonmapper;
					break;
				case CElementIntegrator::Type::PPM:
					ppm = other.ppm;
					break;
				case CElementIntegrator::Type::SPPM:
					sppm = other.sppm;
					break;
				case CElementIntegrator::Type::PSSMLT:
					pssmlt = other.pssmlt;
					break;
				case CElementIntegrator::Type::MLT:
					mlt = other.mlt;
					break;
				case CElementIntegrator::Type::ERPT:
					erpt = other.erpt;
					break;
				case CElementIntegrator::Type::ADJ_P_TRACER:
					ptracer = other.ptracer;
					break;
				case CElementIntegrator::Type::ADAPTIVE:
					adaptive = other.adaptive;
					break;
				case CElementIntegrator::Type::VPL:
					vpl = other.vpl;
					break;
				case CElementIntegrator::Type::IRR_CACHE:
					irrcache = other.irrcache;
					break;
				case CElementIntegrator::Type::MULTI_CHANNEL:
					multichannel = other.multichannel;
					break;
				case CElementIntegrator::Type::FIELD_EXTRACT:
					field = other.field;
					break;
				default:
					break;
			}
			return *this;
		}

		bool addProperty(SNamedPropertyElement&& _property) override;
		bool onEndTag(asset::IAssetLoader::IAssetLoaderOverride* _override, CMitsubaMetadata* globalMetadata) override;
		IElement::Type getType() const override { return IElement::Type::INTEGRATOR; }
		std::string getLogName() const override { return "integrator"; }

		bool processChildData(IElement* _child, const std::string& name) override
		{
			if (!_child)
				return true;

			switch (type)
			{
				case Type::IRR_CACHE:
					[[fallthrough]];
				case Type::MULTI_CHANNEL:
					if (_child->getType() != IElement::Type::INTEGRATOR)
						return false;
					break;
				default:
					break;
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
				default:
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
			SimpleVolumetricPathTracing					volpath_simple;
			ExtendedVolumetricPathTracing				volpath;
			PathTracing									path;
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
}
}

#endif