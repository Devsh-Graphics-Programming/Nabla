#include "irr/ext/MitsubaLoader/ParserUtil.h"
#include "irr/ext/MitsubaLoader/CElementFactory.h"

#include "irr/ext/MitsubaLoader/CGlobalMitsubaMetadata.h"

#include <functional>

namespace irr
{
namespace ext
{
namespace MitsubaLoader
{


template<>
CElementFactory::return_type CElementFactory::createElement<CElementIntegrator>(const char** _atts, ParserManager* _util)
{
	const char* type;
	const char* id;
	std::string name;
	if (!IElement::getTypeIDAndNameStrings(type, id, name, _atts))
		return CElementFactory::return_type(nullptr,"");

	static const core::unordered_map<std::string, CElementIntegrator::Type, core::CaseInsensitiveHash, core::CaseInsensitiveEquals> StringToType =
	{
		{"ao",				CElementIntegrator::Type::AO},
		{"direct",			CElementIntegrator::Type::DIRECT},
		{"path",			CElementIntegrator::Type::PATH},
		{"volpath_simple",	CElementIntegrator::Type::VOL_PATH_SIMPLE},
		{"volpath",			CElementIntegrator::Type::VOL_PATH},
		{"bdpt",			CElementIntegrator::Type::BDPT},
		{"photonmapper",	CElementIntegrator::Type::PHOTONMAPPER},
		{"ppm",				CElementIntegrator::Type::PPM},
		{"sppm",			CElementIntegrator::Type::SPPM},
		{"pssmlt",			CElementIntegrator::Type::PSSMLT},
		{"mlt",				CElementIntegrator::Type::MLT},
		{"erpt",			CElementIntegrator::Type::ERPT},
		{"ptracer",			CElementIntegrator::Type::ADJ_P_TRACER},
		{"adaptive",		CElementIntegrator::Type::ADAPTIVE},
		{"vpl",				CElementIntegrator::Type::VPL},
		{"irrcache",		CElementIntegrator::Type::IRR_CACHE},
		{"multichannel",	CElementIntegrator::Type::MULTI_CHANNEL},
		{"field",			CElementIntegrator::Type::FIELD_EXTRACT}
	};

	auto found = StringToType.find(type);
	if (found==StringToType.end())
	{
		ParserLog::invalidXMLFileStructure("unknown type");
		_IRR_DEBUG_BREAK_IF(false);
		return CElementFactory::return_type(nullptr, "");
	}

	CElementIntegrator* obj = _util->objects.construct<CElementIntegrator>(id);
	if (!obj)
		return CElementFactory::return_type(nullptr, "");

	obj->type = found->second;
	// defaults
	switch (obj->type)
	{
		case CElementIntegrator::Type::AO:
			obj->ao = CElementIntegrator::AmbientOcclusion();
			break;
		case CElementIntegrator::Type::DIRECT:
			obj->direct = CElementIntegrator::DirectIllumination();
			break;
		case CElementIntegrator::Type::PATH:
			obj->path = CElementIntegrator::PathTracing();
			break;
		case CElementIntegrator::Type::VOL_PATH_SIMPLE:
			obj->volpath_simple = CElementIntegrator::SimpleVolumetricPathTracing();
			break;
		case CElementIntegrator::Type::VOL_PATH:
			obj->volpath = CElementIntegrator::ExtendedVolumetricPathTracing();
			break;
		case CElementIntegrator::Type::BDPT:
			obj->bdpt = CElementIntegrator::BiDirectionalPathTracing();
			break;
		case CElementIntegrator::Type::PHOTONMAPPER:
			obj->photonmapper = CElementIntegrator::PhotonMapping();
			break;
		case CElementIntegrator::Type::PPM:
			obj->ppm = CElementIntegrator::ProgressivePhotonMapping();
			break;
		case CElementIntegrator::Type::SPPM:
			obj->sppm = CElementIntegrator::StochasticProgressivePhotonMapping();
			break;
		case CElementIntegrator::Type::PSSMLT:
			obj->pssmlt = CElementIntegrator::PrimarySampleSpaceMetropolisLightTransport();
			break;
		case CElementIntegrator::Type::MLT:
			obj->mlt = CElementIntegrator::PathSpaceMetropolisLightTransport();
			break;
		case CElementIntegrator::Type::ERPT:
			obj->erpt = CElementIntegrator::EnergyRedistributionPathTracing();
			break;
		case CElementIntegrator::Type::ADJ_P_TRACER:
			obj->ptracer = CElementIntegrator::AdjointParticleTracing();
			break;
		case CElementIntegrator::Type::ADAPTIVE:
			obj->adaptive = CElementIntegrator::AdaptiveIntegrator();
			break;
		case CElementIntegrator::Type::VPL:
			obj->vpl = CElementIntegrator::VirtualPointLights();
			break;
		case CElementIntegrator::Type::IRR_CACHE:
			obj->irrcache = CElementIntegrator::IrradianceCacheIntegrator();
			break;
		case CElementIntegrator::Type::MULTI_CHANNEL:
			obj->multichannel = CElementIntegrator::MultiChannelIntegrator();
			break;
		case CElementIntegrator::Type::FIELD_EXTRACT:
			obj->field = CElementIntegrator::FieldExtraction();
			break;
		default:
			break;
	}
	return CElementFactory::return_type(obj, std::move(name));
}

bool CElementIntegrator::addProperty(SNamedPropertyElement&& _property)
{
	bool error = false;
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
			if constexpr (is_any_of<std::remove_reference<decltype(state)>::type,__VA_ARGS__>::value) \
			{ \
				if (_property.type!=PROPERTY_TYPE) { \
					error = true; \
					return; \
				} \
				state. ## MEMBER = _property.getProperty<PROPERTY_TYPE>(); \
			} \
		}); \
	}

	auto processRayLength = SET_PROPERTY_TEMPLATE(rayLength,SNamedPropertyElement::Type::FLOAT,AmbientOcclusion);
	auto processEmitterSamples = SET_PROPERTY_TEMPLATE(emitterSamples,SNamedPropertyElement::Type::INTEGER,DirectIllumination);
	auto processBSDFSamples = SET_PROPERTY_TEMPLATE(bsdfSamples,SNamedPropertyElement::Type::INTEGER,DirectIllumination);
	auto processShadingSamples = [&]() -> void
	{ 
		dispatch([&](auto& state) -> void {
			using state_type = std::remove_reference<decltype(state)>::type;

			if constexpr (std::is_same<state_type,AmbientOcclusion>::value)
			{
				if (_property.type!=SNamedPropertyElement::Type::INTEGER)
				{
					error = true;
					return;
				}
				state.shadingSamples = _property.getProperty<SNamedPropertyElement::Type::INTEGER>();
			}
			else
			{
				if constexpr (std::is_same<state_type,DirectIllumination>::value)
				{
					processEmitterSamples();
					processBSDFSamples();
				}
			}
		});
	};
	auto processStrictNormals = SET_PROPERTY_TEMPLATE(strictNormals,SNamedPropertyElement::Type::BOOLEAN,DirectIllumination,PathTracing);
	auto processHideEmitters = SET_PROPERTY_TEMPLATE(hideEmitters,SNamedPropertyElement::Type::BOOLEAN,DirectIllumination,PathTracing,PhotonMapping);
#define ALL_PHOTONMAPPING_TYPES PhotonMapping,ProgressivePhotonMapping,StochasticProgressivePhotonMapping
#define ALL_MLT_TYPES			PrimarySampleSpaceMetropolisLightTransport,PathSpaceMetropolisLightTransport
#define ALL_MC_TYPES			PathTracing,SimpleVolumetricPathTracing,ExtendedVolumetricPathTracing,BiDirectionalPathTracing, \
								ALL_PHOTONMAPPING_TYPES,ALL_MLT_TYPES,EnergyRedistributionPathTracing,AdjointParticleTracing
	auto processMaxDepth = SET_PROPERTY_TEMPLATE(maxPathDepth,SNamedPropertyElement::Type::INTEGER,ALL_MC_TYPES, VirtualPointLights);
	auto processRRDepth = SET_PROPERTY_TEMPLATE(russianRouletteDepth,SNamedPropertyElement::Type::INTEGER,ALL_MC_TYPES);
#undef ALL_MC_TYPES
	auto processLightImage = SET_PROPERTY_TEMPLATE(lightImage,SNamedPropertyElement::Type::BOOLEAN,BiDirectionalPathTracing);
	auto processSampleDirect = SET_PROPERTY_TEMPLATE(sampleDirect,SNamedPropertyElement::Type::BOOLEAN,BiDirectionalPathTracing);
	auto processGranularity = SET_PROPERTY_TEMPLATE(granularity,SNamedPropertyElement::Type::INTEGER,ALL_PHOTONMAPPING_TYPES,AdjointParticleTracing);
#undef ALL_PHOTONMAPPING_TYPES
	auto processDirectSamples = SET_PROPERTY_TEMPLATE(directSamples,SNamedPropertyElement::Type::INTEGER,PhotonMapping,ALL_MLT_TYPES,EnergyRedistributionPathTracing);
	auto processGlossySamples = SET_PROPERTY_TEMPLATE(glossySamples,SNamedPropertyElement::Type::INTEGER,PhotonMapping);
	auto processGlobalPhotons = SET_PROPERTY_TEMPLATE(globalPhotons,SNamedPropertyElement::Type::INTEGER,PhotonMapping);
	auto processCausticPhotons = SET_PROPERTY_TEMPLATE(causticPhotons,SNamedPropertyElement::Type::INTEGER,PhotonMapping);
	auto processVolumePhotons = SET_PROPERTY_TEMPLATE(volumePhotons,SNamedPropertyElement::Type::INTEGER,PhotonMapping);
	auto processGlobalLookupRadius = SET_PROPERTY_TEMPLATE(globalLURadius,SNamedPropertyElement::Type::FLOAT,PhotonMapping);
	auto processCausticLookupRadius = SET_PROPERTY_TEMPLATE(causticLURadius,SNamedPropertyElement::Type::FLOAT,PhotonMapping);
	auto processLookupSize = SET_PROPERTY_TEMPLATE(LUSize,SNamedPropertyElement::Type::INTEGER,PhotonMapping);
	auto processPhotonCount = SET_PROPERTY_TEMPLATE(photonCount,SNamedPropertyElement::Type::INTEGER,ProgressivePhotonMapping,StochasticProgressivePhotonMapping);
	auto processInitialRadius = SET_PROPERTY_TEMPLATE(initialRadius,SNamedPropertyElement::Type::FLOAT,ProgressivePhotonMapping,StochasticProgressivePhotonMapping);
	auto processAlpha = SET_PROPERTY_TEMPLATE(alpha,SNamedPropertyElement::Type::FLOAT,ProgressivePhotonMapping,StochasticProgressivePhotonMapping);
	auto processMaxPasses = SET_PROPERTY_TEMPLATE(maxPasses,SNamedPropertyElement::Type::INTEGER,ProgressivePhotonMapping,StochasticProgressivePhotonMapping);
	auto processLuminanceSamples = SET_PROPERTY_TEMPLATE(luminanceSamples,SNamedPropertyElement::Type::INTEGER,ALL_MLT_TYPES);
	auto processTwoStage = SET_PROPERTY_TEMPLATE(twoStage,SNamedPropertyElement::Type::BOOLEAN,ALL_MLT_TYPES);
#undef ALL_MLT_TYPES
	auto processBidirectional = SET_PROPERTY_TEMPLATE(bidirectional,SNamedPropertyElement::Type::BOOLEAN,PrimarySampleSpaceMetropolisLightTransport);
	auto processPLarge = SET_PROPERTY_TEMPLATE(pLarge,SNamedPropertyElement::Type::FLOAT,PrimarySampleSpaceMetropolisLightTransport);
	auto processLensPerturbation = SET_PROPERTY_TEMPLATE(lensPerturbation,SNamedPropertyElement::Type::BOOLEAN,PathSpaceMetropolisLightTransport,EnergyRedistributionPathTracing);
	auto processMultiChainPerturbation = SET_PROPERTY_TEMPLATE(multiChainPerturbation,SNamedPropertyElement::Type::BOOLEAN,PathSpaceMetropolisLightTransport,EnergyRedistributionPathTracing);
	auto processCausticPerturbation = SET_PROPERTY_TEMPLATE(causticPerturbation,SNamedPropertyElement::Type::BOOLEAN,PathSpaceMetropolisLightTransport,EnergyRedistributionPathTracing);
	auto processManifoldPerturbation = SET_PROPERTY_TEMPLATE(manifoldPerturbation,SNamedPropertyElement::Type::BOOLEAN,PathSpaceMetropolisLightTransport,EnergyRedistributionPathTracing);
	auto processLambda = SET_PROPERTY_TEMPLATE(lambda,SNamedPropertyElement::Type::FLOAT,PathSpaceMetropolisLightTransport,EnergyRedistributionPathTracing);
	auto processBidirectionalMutation = SET_PROPERTY_TEMPLATE(bidirectionalMutation,SNamedPropertyElement::Type::BOOLEAN,PathSpaceMetropolisLightTransport);
	auto processNumChains = SET_PROPERTY_TEMPLATE(numChains,SNamedPropertyElement::Type::FLOAT,EnergyRedistributionPathTracing);
	auto processMaxChains = SET_PROPERTY_TEMPLATE(maxChains,SNamedPropertyElement::Type::FLOAT,EnergyRedistributionPathTracing);
	auto processChainLength = SET_PROPERTY_TEMPLATE(chainLength,SNamedPropertyElement::Type::INTEGER,EnergyRedistributionPathTracing);
	auto processBruteForce = SET_PROPERTY_TEMPLATE(bruteForce,SNamedPropertyElement::Type::BOOLEAN,AdjointParticleTracing);
	auto processShadowMap = SET_PROPERTY_TEMPLATE(shadowMap,SNamedPropertyElement::Type::INTEGER,VirtualPointLights);
	auto processClamping = SET_PROPERTY_TEMPLATE(clamping,SNamedPropertyElement::Type::FLOAT,VirtualPointLights);
	auto processField = [&]() -> void
	{
		dispatch([&](auto& state) -> void
		{
			using state_type = std::remove_reference<decltype(state)>::type;
			if constexpr (std::is_same<state_type,FieldExtraction>::value)
			{
				if (_property.type != SNamedPropertyElement::Type::STRING)
				{
					error = true;
					return;
				}
				static const core::unordered_map<std::string,FieldExtraction::Type,core::CaseInsensitiveHash,core::CaseInsensitiveEquals> StringToType =
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
				if (found!=StringToType.end())
					state.field = found->second;
				else
					state.field = FieldExtraction::Type::INVALID;
			}
		});
	};
	auto processUndefined = [&]() -> void
	{
		dispatch([&](auto& state) -> void {
			using state_type = std::remove_reference<decltype(state)>::type;

			if constexpr (std::is_same<state_type, FieldExtraction>::value)
			{
				if (_property.type != SNamedPropertyElement::Type::FLOAT && _property.type != SNamedPropertyElement::Type::SPECTRUM)
				{
					error = true;
					return;
				}
				state.undefined = _property; // TODO: redo
			}
		});
	};
	auto processMaxError = SET_PROPERTY_TEMPLATE(maxError,SNamedPropertyElement::Type::FLOAT,AdaptiveIntegrator);
	auto processPValue = SET_PROPERTY_TEMPLATE(pValue,SNamedPropertyElement::Type::FLOAT,AdaptiveIntegrator);
	auto processMaxSampleFactor = SET_PROPERTY_TEMPLATE(maxSampleFactor,SNamedPropertyElement::Type::INTEGER,AdaptiveIntegrator);
	auto processResolution = SET_PROPERTY_TEMPLATE(resolution,SNamedPropertyElement::Type::INTEGER,IrradianceCacheIntegrator);
	auto processQuality = SET_PROPERTY_TEMPLATE(quality,SNamedPropertyElement::Type::FLOAT,IrradianceCacheIntegrator);
	auto processGradients = SET_PROPERTY_TEMPLATE(gradients,SNamedPropertyElement::Type::BOOLEAN,IrradianceCacheIntegrator);
	auto processClampNeighbour = SET_PROPERTY_TEMPLATE(clampNeighbour,SNamedPropertyElement::Type::BOOLEAN,IrradianceCacheIntegrator);
	auto processClampScreen = SET_PROPERTY_TEMPLATE(clampScreen,SNamedPropertyElement::Type::BOOLEAN,IrradianceCacheIntegrator);
	auto processOverture = SET_PROPERTY_TEMPLATE(overture,SNamedPropertyElement::Type::BOOLEAN,IrradianceCacheIntegrator);
	auto processQualityAdjustment = SET_PROPERTY_TEMPLATE(qualityAdjustment,SNamedPropertyElement::Type::FLOAT,IrradianceCacheIntegrator);
	auto processIndirecOnly = SET_PROPERTY_TEMPLATE(indirectOnly,SNamedPropertyElement::Type::BOOLEAN,IrradianceCacheIntegrator);
	auto processDebug = SET_PROPERTY_TEMPLATE(debug,SNamedPropertyElement::Type::BOOLEAN,IrradianceCacheIntegrator);

	const core::unordered_map<std::string, std::function<void()>, core::CaseInsensitiveHash, core::CaseInsensitiveEquals> SetPropertyMap =
	{
		{"shadingSamples",processShadingSamples},
		{"rayLength",processRayLength},
		{"emitterSamples",processEmitterSamples},
		{"bsdfSamples",processBSDFSamples},
		{"strictNormals",processStrictNormals},
		{"hideEmitters",processHideEmitters},
		{"maxDepth",processMaxDepth},
		{"rrDepth",processRRDepth},
		{"lightImage",processLightImage},
		{"sampleDirect",processSampleDirect},
		{"granularity",processGranularity},
		{"directSamples",processDirectSamples},
		{"glossySamples",processGlossySamples},
		{"globalPhotons",processGlobalPhotons},
		{"causticPhotons",processCausticPhotons},
		{"volumePhotons",processVolumePhotons},
		{"globalLookupRadius",processGlobalLookupRadius},
		{"causticLookupRadius",processCausticLookupRadius},
		{"lookupSize",processLookupSize},
		{"photonCount",processPhotonCount},
		{"initialRadius",processInitialRadius},
		{"alpha",processAlpha},
		{"maxPasses",processMaxPasses},
		{"luminanceSamples",processLuminanceSamples},
		{"twoStage",processTwoStage},
		{"bidirectional",processBidirectional},
		{"pLarge",processPLarge},
		{"lensPerturbation",processLensPerturbation},
		{"multiChainPerturbation",processMultiChainPerturbation},
		{"causticPerturbation",processCausticPerturbation},
		{"manifoldPerturbation",processManifoldPerturbation},
		{"lambda",processLambda},
		{"bidirectionalMutation",processBidirectionalMutation},
		{"numChains",processNumChains},
		{"maxChains",processMaxChains},
		{"chainLength",processChainLength},
		{"bruteForce",processBruteForce},
		{"shadowMap",processShadowMap},
		{"clamping",processClamping},
		{"field",processField},
		{"undefined",processUndefined},
		{"maxError",processMaxError},
		{"pValue",processPValue},
		{"maxSampleFactor",processMaxSampleFactor},
		{"resolution",processResolution},
		{"quality",processQuality},
		{"gradients",processGradients},
		{"clampNeighbour",processClampNeighbour},
		{"clampScreen",processClampScreen},
		{"overture",processOverture},
		{"qualityAdjustment",processQualityAdjustment},
		{"indirectOnly",processIndirecOnly},
		{"debug",processDebug},
	};
	

	auto found = SetPropertyMap.find(_property.name);
	if (found==SetPropertyMap.end())
	{
		_IRR_DEBUG_BREAK_IF(true);
		ParserLog::invalidXMLFileStructure("No Integrator can have such property set with name: "+_property.name);
		return false;
	}

	found->second();
	return !error;
}

bool CElementIntegrator::onEndTag(asset::IAssetLoader::IAssetLoaderOverride* _override, CGlobalMitsubaMetadata* globalMetadata)
{
	if (type == Type::INVALID)
	{
		ParserLog::invalidXMLFileStructure(getLogName() + ": type not specified");
		_IRR_DEBUG_BREAK_IF(true);
		return true;
	}
	
	// TODO: Validation
	{
	}

	if (globalMetadata->integrator.type!=Type::INVALID)
	{
		ParserLog::invalidXMLFileStructure(getLogName() + ": already specified an integrator");
		_IRR_DEBUG_BREAK_IF(true);
		return true;
	}
	globalMetadata->integrator = *this;

	return true;
}

}
}
}