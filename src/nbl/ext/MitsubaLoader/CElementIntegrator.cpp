// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/ext/MitsubaLoader/CElementIntegrator.h"
#include "nbl/ext/MitsubaLoader/CMitsubaMetadata.h"

#include "nbl/ext/MitsubaLoader/ElementMacros.h"

#include "nbl/type_traits.h" // legacy stuff for `is_any_of`
#include <functional>


namespace nbl::ext::MitsubaLoader
{

auto CElementFilm::compAddPropertyMap() -> AddPropertyMap<CElementFilm>
{
	using this_t = CElementFilm;
	AddPropertyMap<CElementFilm> retval;

	return retval;
}

bool CElementIntegrator::addProperty(SNamedPropertyElement&& _property, system::logger_opt_ptr logger)
{
	if (type>=Type::INVALID)
		return false;
	bool error = false;
#if 0
#define SET_PROPERTY_TEMPLATE(MEMBER,PROPERTY_TYPE, ... )		[&]() -> void { \
		visit([&](auto& state) -> void { \
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
		visit([&](auto& state) -> void {
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
	auto processHideEnvironment = SET_PROPERTY_TEMPLATE(hideEnvironment,SNamedPropertyElement::Type::BOOLEAN,DirectIllumination,PathTracing,PhotonMapping);
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
		visit([&](auto& state) -> void
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
		visit([&](auto& state) -> void {
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
		{"hideEnvironment",processHideEnvironment},
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
		invalidXMLFileStructure(logger,"No Integrator can have such property set with name: "+_property.name);
		return false;
	}

	found->second();
#endif
	return !error;
}

bool CElementIntegrator::onEndTag(CMitsubaMetadata* metadata, system::logger_opt_ptr logger)
{
	NBL_EXT_MITSUBA_LOADER_ELEMENT_INVALID_TYPE_CHECK(true);
	
	// TODO: Validation
	{
	}

	if (metadata->m_global.m_integrator.type!=Type::INVALID)
	{
		invalidXMLFileStructure(logger,"already specified an integrator, NOT overwriting.");
		return true;
	}
	metadata->m_global.m_integrator = *this;

	return true;
}

}