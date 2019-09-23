#include "../../ext/MitsubaLoader/ParserUtil.h"

#include "../../ext/MitsubaLoader/CElementFactory.h"

#include <functional>

namespace irr
{
namespace ext
{
namespace MitsubaLoader
{


template<>
IElement* CElementFactory::createElement<CElementFilm>(const char** _atts, ParserManager* _util)
{
	if (IElement::invalidAttributeCount(_atts, 2u))
		return nullptr;
	if (core::strcmpi(_atts[0], "type"))
		return nullptr;

	static const core::unordered_map<std::string, CElementFilm::Type, core::CaseInsensitiveHash, core::CaseInsensitiveEquals> StringToType =
	{
		{"hdrfilm",		CElementFilm::Type::HDR_FILM},
		{"tiledhdrfilm",CElementFilm::Type::TILED_HDR},
		{"ldrfilm",		CElementFilm::Type::LDR_FILM},
		{"mfilm",		CElementFilm::Type::MFILM}
	};

	auto found = StringToType.find(_atts[1]);
	if (found==StringToType.end())
	{
		ParserLog::invalidXMLFileStructure("unknown type");
		_IRR_DEBUG_BREAK_IF(false);
		return nullptr;
	}

	CElementIntegrator* obj = _util->objects.construct<CElementIntegrator>();
	if (!obj)
		return nullptr;

	obj->type = found->second;
	// defaults
	switch (obj->type)
	{
		case CElementFilm::Type::TILED_HDR:
			obj-> = ;
			break;
		case CElementFilm::Type::LDR_FILM:
			obj->ldrfilm = CElementFilm::LDR();
			break;
		case CElementFilm::Type::MFILM:
			obj->m = CElementFilm::M();
			break;
		default:
			break;
	}
	return obj;
}


bool CElementFilm::addProperty(SPropertyElementData&& _property)
{
	bool error = false;/*
	auto dispatch = [&](auto func) -> void
	{
		switch (type)
		{
		case CElementIntegrator::Type::AO:
			func(ao);
			break;
		case CElementIntegrator::Type::DIRECT:
			func(direct);
			break;
		case CElementIntegrator::Type::PATH:
			func(path);
			break;
		case CElementIntegrator::Type::VOL_PATH_SIMPLE:
			func(volpath_simple);
			break;
		case CElementIntegrator::Type::VOL_PATH:
			func(volpath);
			break;
		case CElementIntegrator::Type::BDPT:
			func(bdpt);
			break;
		case CElementIntegrator::Type::PHOTONMAPPER:
			func(photonmapper);
			break;
		case CElementIntegrator::Type::PPM:
			func(ppm);
			break;
		case CElementIntegrator::Type::SPPM:
			func(sppm);
			break;
		case CElementIntegrator::Type::PSSMLT:
			func(pssmlt);
			break;
		case CElementIntegrator::Type::MLT:
			func(mlt);
			break;
		case CElementIntegrator::Type::ERPT:
			func(erpt);
			break;
		case CElementIntegrator::Type::ADJ_P_TRACER:
			func(ptracer);
			break;
		case CElementIntegrator::Type::ADAPTIVE:
			func(adaptive);
			break;
		case CElementIntegrator::Type::VPL:
			func(vpl);
			break;
		case CElementIntegrator::Type::IRR_CACHE:
			func(irrcache);
			break;
		case CElementIntegrator::Type::MULTI_CHANNEL:
			func(multichannel);
			break;
		case CElementIntegrator::Type::FIELD_EXTRACT:
			func(field);
			break;
		default:
			error = true;
			break;
		}
	};

#define SET_PROPERTY_TEMPLATE(MEMBER,PROPERTY_TYPE, ... )		[&]() -> void { \
		dispatch([&](auto& state) -> void { \
			IRR_PSEUDO_IF_CONSTEXPR_BEGIN(is_any_of<std::remove_reference<decltype(state)>::type,__VA_ARGS__>::value) \
			{ \
				if (_property.type!=PROPERTY_TYPE) { \
					error = true; \
					return; \
				} \
				state. ## MEMBER = _property.getProperty<PROPERTY_TYPE>(); \
			} \
			IRR_PSEUDO_IF_CONSTEXPR_END \
		}); \
	}

	auto processRayLength = SET_PROPERTY_TEMPLATE(rayLength, SPropertyElementData::Type::FLOAT, AmbientOcclusion);
	auto processEmitterSamples = SET_PROPERTY_TEMPLATE(emitterSamples, SPropertyElementData::Type::INTEGER, DirectIllumination);
	auto processBSDFSamples = SET_PROPERTY_TEMPLATE(bsdfSamples, SPropertyElementData::Type::INTEGER, DirectIllumination);
	auto processShadingSamples = [&]() -> void
	{
		dispatch([&](auto& state) -> void {
			using state_type = std::remove_reference<decltype(state)>::type;

			IRR_PSEUDO_IF_CONSTEXPR_BEGIN(std::is_same<state_type, AmbientOcclusion>::value)
			{
				if (_property.type != SPropertyElementData::Type::INTEGER)
				{
					error = true;
					return;
				}
				state.shadingSamples = _property.getProperty<SPropertyElementData::Type::INTEGER>();
			}
			IRR_PSEUDO_ELSE_CONSTEXPR
			{
				IRR_PSEUDO_IF_CONSTEXPR_BEGIN(std::is_same<state_type,DirectIllumination>::value)
				{
					processEmitterSamples();
					processBSDFSamples();
				}
				IRR_PSEUDO_IF_CONSTEXPR_END
			}
				IRR_PSEUDO_IF_CONSTEXPR_END
			});
	};
	auto processStrictNormals = SET_PROPERTY_TEMPLATE(strictNormals, SPropertyElementData::Type::BOOLEAN, DirectIllumination, PathTracing);
	auto processHideEmitters = SET_PROPERTY_TEMPLATE(hideEmitters, SPropertyElementData::Type::BOOLEAN, DirectIllumination, PathTracing, PhotonMapping);
#define ALL_PHOTONMAPPING_TYPES PhotonMapping,ProgressivePhotonMapping,StochasticProgressivePhotonMapping
#define ALL_MLT_TYPES			PrimarySampleSpaceMetropolisLightTransport,PathSpaceMetropolisLightTransport
#define ALL_MC_TYPES			PathTracing,SimpleVolumetricPathTracing,ExtendedVolumetricPathTracing,BiDirectionalPathTracing, \
								ALL_PHOTONMAPPING_TYPES,ALL_MLT_TYPES,EnergyRedistributionPathTracing,AdjointParticleTracing
	auto processMaxDepth = SET_PROPERTY_TEMPLATE(maxPathDepth, SPropertyElementData::Type::INTEGER, ALL_MC_TYPES, VirtualPointLights);
	auto processRRDepth = SET_PROPERTY_TEMPLATE(russianRouletteDepth, SPropertyElementData::Type::INTEGER, ALL_MC_TYPES);
#undef ALL_MC_TYPES
	auto processLightImage = SET_PROPERTY_TEMPLATE(lightImage, SPropertyElementData::Type::BOOLEAN, BiDirectionalPathTracing);
	auto processSampleDirect = SET_PROPERTY_TEMPLATE(sampleDirect, SPropertyElementData::Type::BOOLEAN, BiDirectionalPathTracing);
	auto processGranularity = SET_PROPERTY_TEMPLATE(granularity, SPropertyElementData::Type::INTEGER, ALL_PHOTONMAPPING_TYPES, AdjointParticleTracing);
#undef ALL_PHOTONMAPPING_TYPES
	auto processDirectSamples = SET_PROPERTY_TEMPLATE(directSamples, SPropertyElementData::Type::INTEGER, PhotonMapping, ALL_MLT_TYPES, EnergyRedistributionPathTracing);
	auto processGlossySamples = SET_PROPERTY_TEMPLATE(glossySamples, SPropertyElementData::Type::INTEGER, PhotonMapping);
	auto processGlobalPhotons = SET_PROPERTY_TEMPLATE(globalPhotons, SPropertyElementData::Type::INTEGER, PhotonMapping);
	auto processCausticPhotons = SET_PROPERTY_TEMPLATE(causticPhotons, SPropertyElementData::Type::INTEGER, PhotonMapping);
	auto processVolumePhotons = SET_PROPERTY_TEMPLATE(volumePhotons, SPropertyElementData::Type::INTEGER, PhotonMapping);
	auto processGlobalLookupRadius = SET_PROPERTY_TEMPLATE(globalLURadius, SPropertyElementData::Type::FLOAT, PhotonMapping);
	auto processCausticLookupRadius = SET_PROPERTY_TEMPLATE(causticLURadius, SPropertyElementData::Type::FLOAT, PhotonMapping);
	auto processLookupSize = SET_PROPERTY_TEMPLATE(LUSize, SPropertyElementData::Type::INTEGER, PhotonMapping);
	auto processPhotonCount = SET_PROPERTY_TEMPLATE(photonCount, SPropertyElementData::Type::INTEGER, ProgressivePhotonMapping, StochasticProgressivePhotonMapping);
	auto processInitialRadius = SET_PROPERTY_TEMPLATE(initialRadius, SPropertyElementData::Type::FLOAT, ProgressivePhotonMapping, StochasticProgressivePhotonMapping);
	auto processAlpha = SET_PROPERTY_TEMPLATE(alpha, SPropertyElementData::Type::FLOAT, ProgressivePhotonMapping, StochasticProgressivePhotonMapping);
	auto processMaxPasses = SET_PROPERTY_TEMPLATE(maxPasses, SPropertyElementData::Type::INTEGER, ProgressivePhotonMapping, StochasticProgressivePhotonMapping);
	auto processLuminanceSamples = SET_PROPERTY_TEMPLATE(luminanceSamples, SPropertyElementData::Type::INTEGER, ALL_MLT_TYPES);
	auto processTwoStage = SET_PROPERTY_TEMPLATE(twoStage, SPropertyElementData::Type::BOOLEAN, ALL_MLT_TYPES);
#undef ALL_MLT_TYPES
	auto processBidirectional = SET_PROPERTY_TEMPLATE(bidirectional, SPropertyElementData::Type::BOOLEAN, PrimarySampleSpaceMetropolisLightTransport);
	auto processPLarge = SET_PROPERTY_TEMPLATE(pLarge, SPropertyElementData::Type::FLOAT, PrimarySampleSpaceMetropolisLightTransport);
	auto processLensPerturbation = SET_PROPERTY_TEMPLATE(lensPerturbation, SPropertyElementData::Type::BOOLEAN, PathSpaceMetropolisLightTransport, EnergyRedistributionPathTracing);
	auto processMultiChainPerturbation = SET_PROPERTY_TEMPLATE(multiChainPerturbation, SPropertyElementData::Type::BOOLEAN, PathSpaceMetropolisLightTransport, EnergyRedistributionPathTracing);
	auto processCausticPerturbation = SET_PROPERTY_TEMPLATE(causticPerturbation, SPropertyElementData::Type::BOOLEAN, PathSpaceMetropolisLightTransport, EnergyRedistributionPathTracing);
	auto processManifoldPerturbation = SET_PROPERTY_TEMPLATE(manifoldPerturbation, SPropertyElementData::Type::BOOLEAN, PathSpaceMetropolisLightTransport, EnergyRedistributionPathTracing);
	auto processLambda = SET_PROPERTY_TEMPLATE(lambda, SPropertyElementData::Type::FLOAT, PathSpaceMetropolisLightTransport, EnergyRedistributionPathTracing);
	auto processBidirectionalMutation = SET_PROPERTY_TEMPLATE(bidirectionalMutation, SPropertyElementData::Type::BOOLEAN, PathSpaceMetropolisLightTransport);
	auto processNumChains = SET_PROPERTY_TEMPLATE(numChains, SPropertyElementData::Type::FLOAT, EnergyRedistributionPathTracing);
	auto processMaxChains = SET_PROPERTY_TEMPLATE(maxChains, SPropertyElementData::Type::FLOAT, EnergyRedistributionPathTracing);
	auto processChainLength = SET_PROPERTY_TEMPLATE(chainLength, SPropertyElementData::Type::INTEGER, EnergyRedistributionPathTracing);
	auto processBruteForce = SET_PROPERTY_TEMPLATE(bruteForce, SPropertyElementData::Type::BOOLEAN, AdjointParticleTracing);
	auto processShadowMap = SET_PROPERTY_TEMPLATE(shadowMap, SPropertyElementData::Type::INTEGER, VirtualPointLights);
	auto processClamping = SET_PROPERTY_TEMPLATE(clamping, SPropertyElementData::Type::FLOAT, VirtualPointLights);
	auto processField = [&]() -> void
	{
		dispatch([&](auto& state) -> void
			{
				using state_type = std::remove_reference<decltype(state)>::type;
				IRR_PSEUDO_IF_CONSTEXPR_BEGIN(std::is_same<state_type, FieldExtraction>::value)
				{
					if (_property.type != SPropertyElementData::Type::STRING)
					{
						error = true;
						return;
					}
					static const core::unordered_map<std::string, FieldExtraction::Type, core::CaseInsensitiveHash, core::CaseInsensitiveEquals> StringToType =
					{
						{"position",FieldExtraction::Type::POSITION},
						{"relPosition",FieldExtraction::Type::RELATIVE_POSITION},
						{"distance",FieldExtraction::Type::DISTANCE},
						{"geoNormal",FieldExtraction::Type::GEOMETRIC_NORMAL},
						{"shNormal",FieldExtraction::Type::SHADING_NORMAL},
						{"uv",FieldExtraction::Type::UV_COORD},
						{"albedo",FieldExtraction::Type::ALBEDO},
						{"shapeIndex",FieldExtraction::Type::SHAPE_INDEX},
						{"primIndex",FieldExtraction::Type::PRIMITIVE_INDEX}
					};
					auto found = StringToType.find(_property.svalue);
					if (found != StringToType.end())
						state.field = found->second;
					else
						state.field = FieldExtraction::Type::INVALID;
				}
				IRR_PSEUDO_IF_CONSTEXPR_END
			});
	};
	auto processUndefined = [&]() -> void
	{
		dispatch([&](auto& state) -> void {
			using state_type = std::remove_reference<decltype(state)>::type;

			IRR_PSEUDO_IF_CONSTEXPR_BEGIN(std::is_same<state_type, FieldExtraction>::value)
			{
				if (_property.type != SPropertyElementData::Type::FLOAT && _property.type != SPropertyElementData::Type::SPECTRUM)
				{
					error = true;
					return;
				}
				state.undefined = _property; // TODO: redo
			}
			IRR_PSEUDO_IF_CONSTEXPR_END
			});
	};
	auto processMaxError = SET_PROPERTY_TEMPLATE(maxError, SPropertyElementData::Type::FLOAT, AdaptiveIntegrator);
	auto processPValue = SET_PROPERTY_TEMPLATE(pValue, SPropertyElementData::Type::FLOAT, AdaptiveIntegrator);
	auto processMaxSampleFactor = SET_PROPERTY_TEMPLATE(maxSampleFactor, SPropertyElementData::Type::INTEGER, AdaptiveIntegrator);
	auto processResolution = SET_PROPERTY_TEMPLATE(resolution, SPropertyElementData::Type::INTEGER, IrradianceCacheIntegrator);
	auto processQuality = SET_PROPERTY_TEMPLATE(quality, SPropertyElementData::Type::FLOAT, IrradianceCacheIntegrator);
	auto processGradients = SET_PROPERTY_TEMPLATE(gradients, SPropertyElementData::Type::BOOLEAN, IrradianceCacheIntegrator);
	auto processClampNeighbour = SET_PROPERTY_TEMPLATE(clampNeighbour, SPropertyElementData::Type::BOOLEAN, IrradianceCacheIntegrator);
	auto processClampScreen = SET_PROPERTY_TEMPLATE(clampScreen, SPropertyElementData::Type::BOOLEAN, IrradianceCacheIntegrator);
	auto processOverture = SET_PROPERTY_TEMPLATE(overture, SPropertyElementData::Type::BOOLEAN, IrradianceCacheIntegrator);
	auto processQualityAdjustment = SET_PROPERTY_TEMPLATE(qualityAdjustment, SPropertyElementData::Type::FLOAT, IrradianceCacheIntegrator);
	auto processIndirecOnly = SET_PROPERTY_TEMPLATE(indirectOnly, SPropertyElementData::Type::BOOLEAN, IrradianceCacheIntegrator);
	auto processDebug = SET_PROPERTY_TEMPLATE(debug, SPropertyElementData::Type::BOOLEAN, IrradianceCacheIntegrator);
	*/
	static const core::unordered_map<std::string, std::function<void()>, core::CaseInsensitiveHash, core::CaseInsensitiveEquals> SetPropertyMap =
	{
		//{"shadingSamples",processShadingSamples},
		//{"shadingSamples",processShadingSamples}
	};

	auto found = SetPropertyMap.find(_property.name);
	if (found != SetPropertyMap.end())
	{
		_IRR_DEBUG_BREAK_IF(true);
		ParserLog::invalidXMLFileStructure("No Film can have such property set with name: " + _property.name);
		return false;
	}

	found->second();
	return !error;
}

bool CElementFilm::onEndTag(asset::IAssetLoader::IAssetLoaderOverride* _override, CGlobalMitsubaMetadata* globalMetadata)
{
	switch (type)
	{
		case Type::HDR_FILM:
			break;
		default:
			ParserLog::invalidXMLFileStructure(getLogName() + ": type not specified");
			_IRR_DEBUG_BREAK_IF(true);
			return false;
	}

	return true;
}


}
}
}