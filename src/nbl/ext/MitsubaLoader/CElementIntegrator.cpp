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

namespace impl
{
template<typename T>
struct has_strictNormals
{
	constexpr static bool value = std::is_same_v<T,CElementIntegrator::DirectIllumination> ||
		std::is_base_of_v<CElementIntegrator::PathTracing,T>;
};
template<typename T>
struct has_granularity
{
	constexpr static bool value = std::is_base_of_v<CElementIntegrator::PhotonMappingBase,T> ||
		std::is_same_v<T,CElementIntegrator::AdjointParticleTracing>;
};
template<typename T>
struct has_directSamples
{
	constexpr static bool value = std::is_same_v<T,CElementIntegrator::PhotonMapping> ||
		std::is_base_of_v<CElementIntegrator::MetropolisLightTransportBase,T> ||
		std::is_same_v<T,CElementIntegrator::EnergyRedistributionPathTracing>;
};
}

auto CElementIntegrator::compAddPropertyMap() -> AddPropertyMap<CElementIntegrator>
{
	using this_t = CElementIntegrator;
	AddPropertyMap<CElementIntegrator> retval;

	// common
	// this one has really funny legacy behaviour which Mitsuba allowed contrary to its PDF docs
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_PROPERTY_CONSTRAINED("shadingSamples",INTEGER,is_any_of,AmbientOcclusion,DirectIllumination)
		{
			if (_this->type == Type::AO)
				_this->ao.shadingSamples = _property.ivalue;
			else
				_this->direct.emitterSamples = _this->direct.bsdfSamples = _property.ivalue;
			return true;
		}
	);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(strictNormals,BOOLEAN,impl::has_strictNormals);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(granularity,INTEGER,impl::has_granularity);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(directSamples,INTEGER,impl::has_directSamples);

	// ambient
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(rayLength,FLOAT,std::is_same,AmbientOcclusion);

	// emitter hideables
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(hideEmitters,BOOLEAN,derived_from,DirectIllumination);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(hideEnvironment,BOOLEAN,derived_from,DirectIllumination);

	// direct
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(emitterSamples,INTEGER,std::is_same,DirectIllumination);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(bsdfSamples,INTEGER,std::is_same,DirectIllumination);
	// COMMON: strictNormals

	// monte carlo base
	// Not using `NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED` because members have different names than XML names
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_PROPERTY_CONSTRAINED("maxDepth",INTEGER,derived_from,MonteCarloTracingBase)
		{
			_this->path.maxPathDepth = _property.ivalue;
			return true;
		}
	);
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_PROPERTY_CONSTRAINED("rrDepth",INTEGER,derived_from,MonteCarloTracingBase)
		{
			_this->path.russianRouletteDepth = _property.ivalue;
			return true;
		}
	);

	// path tracing
	// COMMON: strictNormals

	// bidirectional path tracing
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(lightImage,BOOLEAN,std::is_same,BiDirectionalPathTracing);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(sampleDirect,BOOLEAN,std::is_same,BiDirectionalPathTracing);

	// photon mapping base
	// COMMON: granularity

	// photon mapping
	// COMMON: directSamples
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(glossySamples,INTEGER,std::is_same,PhotonMapping);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(globalPhotons,INTEGER,std::is_same,PhotonMapping);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(causticPhotons,INTEGER,std::is_same,PhotonMapping);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(volumePhotons,INTEGER,std::is_same,PhotonMapping);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(globalLookupRadius,FLOAT,std::is_same,PhotonMapping);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(causticLookupRadius,FLOAT,std::is_same,PhotonMapping);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(lookupSize,INTEGER,std::is_same,PhotonMapping);

	// progressive
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(photonCount,INTEGER,derived_from,ProgressivePhotonMapping);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(initialRadius,FLOAT,derived_from,ProgressivePhotonMapping);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(alpha,FLOAT,derived_from,ProgressivePhotonMapping);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(maxPasses,INTEGER,derived_from,ProgressivePhotonMapping);

	// metropolis base
	// COMMON: directSamples
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(luminanceSamples,INTEGER,derived_from,MetropolisLightTransportBase);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(twoStage,BOOLEAN,derived_from,MetropolisLightTransportBase);

	// primary sample space
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(bidirectional,BOOLEAN,std::is_same,PrimarySampleSpaceMetropolisLightTransport);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(pLarge,FLOAT,std::is_same,PrimarySampleSpaceMetropolisLightTransport);

	// permutable base
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(lensPerturbation,BOOLEAN,derived_from,PerturbateableBase);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(multiChainPerturbation,BOOLEAN,derived_from,PerturbateableBase);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(causticPerturbation,BOOLEAN,derived_from,PerturbateableBase);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(manifoldPerturbation,BOOLEAN,derived_from,PerturbateableBase);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(lambda,FLOAT,derived_from,PerturbateableBase);

	// path space metropolis
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(bidirectionalMutation,BOOLEAN,std::is_same,PathSpaceMetropolisLightTransport);

	// energy redistribution
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(numChains,FLOAT,std::is_same,EnergyRedistributionPathTracing);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(maxChains,FLOAT,std::is_same,EnergyRedistributionPathTracing);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(chainLength,INTEGER,std::is_same,EnergyRedistributionPathTracing);
	// COMMON: directSamples

	// adjoint
	// COMMON: granularity
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(bruteForce,BOOLEAN,std::is_same,AdjointParticleTracing);

	// vpl
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(maxPathDepth,INTEGER,std::is_same,VirtualPointLights);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(shadowMap,INTEGER,std::is_same,VirtualPointLights);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(clamping,FLOAT,std::is_same,VirtualPointLights);

	// field extraction
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_PROPERTY_CONSTRAINED("field",STRING,std::is_same,FieldExtraction)
		{
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
				_this->field.field = found->second;
			else
				_this->field.field = FieldExtraction::Type::INVALID;
			return true;
		}
	);
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_PROPERTY_CONSTRAINED("undefined",FLOAT,std::is_same,FieldExtraction)
		{
			_this->field.undefined = _property;
			return true;
		}
	);
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_PROPERTY_CONSTRAINED("undefined",SPECTRUM,std::is_same,FieldExtraction)
		{
			_this->field.undefined = _property;
			return true;
		}
	);

	// Now for the compound/nested integrators
	// meta integrator has no members settable via properties

	// adaptive integrator
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(maxError,FLOAT,std::is_same,AdaptiveIntegrator);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(pValue,FLOAT,std::is_same,AdaptiveIntegrator);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(maxSampleFactor,INTEGER,std::is_same,AdaptiveIntegrator);

	// irradiance cache
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(resolution,INTEGER,std::is_same,IrradianceCacheIntegrator);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(quality,FLOAT,std::is_same,IrradianceCacheIntegrator);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(gradients,BOOLEAN,std::is_same,IrradianceCacheIntegrator);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(clampNeighbour,BOOLEAN,std::is_same,IrradianceCacheIntegrator);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(clampScreen,BOOLEAN,std::is_same,IrradianceCacheIntegrator);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(overture,BOOLEAN,std::is_same,IrradianceCacheIntegrator);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(qualityAdjustment,FLOAT,std::is_same,IrradianceCacheIntegrator);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(indirectOnly,BOOLEAN,std::is_same,IrradianceCacheIntegrator);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(debug,BOOLEAN,std::is_same,IrradianceCacheIntegrator);

	// multi channel no extra members

	
	return retval;
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