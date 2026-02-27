// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_PATH_TRACING_UNIDIRECTIONAL_INCLUDED_
#define _NBL_BUILTIN_HLSL_PATH_TRACING_UNIDIRECTIONAL_INCLUDED_

#include <nbl/builtin/hlsl/colorspace/EOTF.hlsl>
#include <nbl/builtin/hlsl/colorspace/encodeCIEXYZ.hlsl>
#include <nbl/builtin/hlsl/math/functions.hlsl>
#include <nbl/builtin/hlsl/sampling/basic.hlsl>
#include <nbl/builtin/hlsl/bxdf/bxdf_traits.hlsl>
#include <nbl/builtin/hlsl/vector_utils/vector_traits.hlsl>
#include <nbl/builtin/hlsl/path_tracing/concepts.hlsl>

namespace nbl
{
namespace hlsl
{
namespace path_tracing
{

template<class RandGen, class Ray, class Intersector, class MaterialSystem, /* class PathGuider, */ class NextEventEstimator, class Accumulator, class Scene
NBL_PRIMARY_REQUIRES(concepts::RandGenerator<RandGen> && concepts::Ray<Ray> &&
    concepts::Intersector<Intersector> && concepts::MaterialSystem<MaterialSystem> &&
    concepts::NextEventEstimator<NextEventEstimator> && concepts::Accumulator<Accumulator> &&
    concepts::Scene<Scene>)
struct Unidirectional
{
    using this_t = Unidirectional<RandGen, Ray, Intersector, MaterialSystem, NextEventEstimator, Accumulator, Scene>;
    using randgen_type = RandGen;
    using intersector_type = Intersector;
    using material_system_type = MaterialSystem;
    using nee_type = NextEventEstimator;
    using scene_type = Scene;

    using scalar_type = typename MaterialSystem::scalar_type;
    using vector3_type = vector<scalar_type, 3>;
    using monochrome_type = vector<scalar_type, 1>;
    using measure_type = typename MaterialSystem::measure_type;
    using sample_type = typename NextEventEstimator::sample_type;
    using ray_dir_info_type = typename sample_type::ray_dir_info_type;
    using ray_type = Ray;
    using object_handle_type = typename Intersector::object_handle_type;
    using closest_hit_type = typename Intersector::closest_hit_type;
    using bxdfnode_type = typename MaterialSystem::bxdfnode_type;
    using anisotropic_interaction_type = typename MaterialSystem::anisotropic_interaction_type;
    using isotropic_interaction_type = typename anisotropic_interaction_type::isotropic_interaction_type;
    using anisocache_type = typename MaterialSystem::anisocache_type;
    using quotient_pdf_type = typename NextEventEstimator::quotient_pdf_type;
    using tolerance_method_type = typename NextEventEstimator::tolerance_method_type;

    scalar_type getLuma(NBL_CONST_REF_ARG(vector3_type) col)
    {
        return hlsl::dot(spectralTypeToLumaCoeffs, col);
    }

    bool closestHitProgram(uint16_t depth, uint32_t _sample, NBL_REF_ARG(ray_type) ray, NBL_CONST_REF_ARG(closest_hit_type) intersectData)
    {
        anisotropic_interaction_type interaction = intersectData.getInteraction();

        // emissive
        typename scene_type::mat_light_id_type matLightID = scene.getMatLightIDs(intersectData.getObjectID());
        const typename material_system_type::material_id_type matID = matLightID.getMaterialID();
        if (materialSystem.hasEmission(matID))
        {
            measure_type emissive = materialSystem.getEmission(matID, interaction);

            const typename nee_type::light_id_type lightID = matLightID.getLightID();
            if (ray.shouldDoMIS() && matLightID.isLight())
            {
                const scalar_type pdf = nee.deferred_pdf(scene, lightID, ray);
                assert(!hlsl::isinf(pdf));
                emissive *= ray.foundEmissiveMIS(pdf * pdf);
            }
            ray.addPayloadContribution(emissive);
        }

        if (!matLightID.canContinuePath())
            return false;

        bxdfnode_type bxdf = materialSystem.getBxDFNode(matID, interaction);

        vector3_type eps0 = randGen(depth * 2u, _sample);
        vector3_type eps1 = randGen(depth * 2u + 1u, _sample);

        const vector3_type intersectP = intersectData.getPosition();
        measure_type throughputCIE_Y = interaction.getLuminosityContributionHint();

        // sample lights
        const scalar_type neeProbability = bxdf.getNEEProb();
        scalar_type rcpChoiceProb;
        sampling::PartitionRandVariable<scalar_type> partitionRandVariable;
        partitionRandVariable.leftProb = neeProbability;
        assert(neeProbability >= 0.0 && neeProbability <= 1.0);
        if (!partitionRandVariable(eps0.z, rcpChoiceProb))
        {
            typename nee_type::sample_quotient_return_type ret = nee.template generate_and_quotient_and_pdf<material_system_type>(
                scene, materialSystem, intersectP, interaction,
                eps0, depth
            );
            scalar_type t = ret.getT();
            sample_type nee_sample = ret.getSample();
            quotient_pdf_type neeContrib = ret.getQuotientPdf();

            // We don't allow non watertight transmitters in this renderer
            // but if we allowed non-watertight transmitters (single water surface), it would make sense just to apply this line by itself
            const scalar_type monochromeEta = materialSystem.setMonochromeEta(matID, throughputCIE_Y);
            bxdf::fresnel::OrientedEtas<monochrome_type> orientedEta = bxdf::fresnel::OrientedEtas<monochrome_type>::create(interaction.getNdotV(), hlsl::promote<monochrome_type>(monochromeEta));
            anisocache_type _cache = anisocache_type::template create<anisotropic_interaction_type, sample_type>(interaction, nee_sample, orientedEta);

            // While NEE or other generators are not supposed to pick up Delta lobes by accident, we need the MIS weights to add up to 1 for the non-delta lobes.
            // So we need to weigh the Delta lobes as if the MIS weight is always 1, but other areas regularly.
            // Meaning that eval's pdf should equal quotient's pdf , this way even the diffuse contributions coming from within a specular lobe get a MIS weight near 0 for NEE.
            // This stops a discrepancy in MIS weights and NEE mistakenly trying to add non-delta lobe contributions with a MIS weight > 0 and creating energy from thin air.
            if (neeContrib.pdf > scalar_type(0.0))
            {
                // we'll need an `eval_and_mis_weight` and `quotient_and_mis_weight`
                const scalar_type bsdf_pdf = materialSystem.pdf(matID, nee_sample, interaction, _cache);
                neeContrib.quotient *= materialSystem.eval(matID, nee_sample, interaction, _cache) * rcpChoiceProb;
                const scalar_type otherGenOverLightAndChoice = bsdf_pdf * rcpChoiceProb / neeContrib.pdf;
                neeContrib.quotient /= 1.f + otherGenOverLightAndChoice * otherGenOverLightAndChoice;   // balance heuristic

                const vector3_type origin = intersectP;
                const vector3_type direction = nee_sample.getL().getDirection();
                ray_type nee_ray;
                nee_ray.init(origin, direction);
                nee_ray.template setInteraction<anisotropic_interaction_type>(interaction);
                nee_ray.setT(t);
                tolerance_method_type::template adjust<ray_type>(nee_ray, intersectData.getGeometricNormal(), depth);
                if (getLuma(neeContrib.quotient) > lumaContributionThreshold)
                    ray.addPayloadContribution(neeContrib.quotient * intersector_type::traceShadowRay(scene, nee_ray, ret.getLightObjectID()));
            }
        }

        // sample BSDF
        scalar_type bxdfPdf;
        vector3_type bxdfSample;
        vector3_type throughput = ray.getPayloadThroughput();
        {
            anisocache_type _cache;
            sample_type bsdf_sample = materialSystem.generate(matID, interaction, eps1, _cache);

            if (!bsdf_sample.isValid())
                return false;

            // the value of the bsdf divided by the probability of the sample being generated
            quotient_pdf_type bsdf_quotient_pdf = materialSystem.quotient_and_pdf(matID, bsdf_sample, interaction, _cache);
            throughput *= bsdf_quotient_pdf.quotient;
            bxdfPdf = bsdf_quotient_pdf.pdf;
            bxdfSample = bsdf_sample.getL().getDirection();
        }

        // additional threshold
        const float lumaThroughputThreshold = lumaContributionThreshold;
        if (bxdfPdf > bxdfPdfThreshold && getLuma(throughput) > lumaThroughputThreshold)
        {
            scalar_type otherTechniqueHeuristic = neeProbability / bxdfPdf; // numerically stable, don't touch
            assert(!hlsl::isinf(otherTechniqueHeuristic));
            ray.updateThroughputAndMISWeights(throughput, otherTechniqueHeuristic * otherTechniqueHeuristic);

            // trace new ray
            vector3_type origin = intersectP;
            vector3_type direction = bxdfSample;
            ray.init(origin, direction);
            ray.template setInteraction<anisotropic_interaction_type>(interaction);
            tolerance_method_type::template adjust<ray_type>(ray, intersectData.getGeometricNormal(), depth);

            return true;
        }

        return false;
    }

    void missProgram(NBL_REF_ARG(ray_type) ray)
    {
        vector3_type finalContribution = nee.get_environment_radiance(ray);
        typename nee_type::light_id_type env_light_id;
        env_light_id.id = 0u;
        const scalar_type pdf = nee.deferred_pdf(scene, env_light_id, ray);
        finalContribution *= ray.foundEmissiveMIS(pdf * pdf);
        ray.addPayloadContribution(finalContribution);
    }

    // Li
    void sampleMeasure(NBL_REF_ARG(ray_type) ray, uint32_t sampleIndex, uint32_t maxDepth, NBL_REF_ARG(Accumulator) accumulator)
    {
        // bounces
        // note do 1-based indexing because we expect first dimension was consumed to generate the ray
        bool continuePath = true;
        for (uint16_t d = 1; (d <= maxDepth) && continuePath; d++)
        {
            ray.setT(numeric_limits<scalar_type>::max);
            closest_hit_type intersection = intersector_type::traceClosestHit(scene, ray);

            continuePath = intersection.foundHit();
            if (continuePath)
                continuePath = closestHitProgram(d, sampleIndex, ray, intersection);
        }
        if (!continuePath)
            missProgram(ray);

        const uint32_t sampleCount = sampleIndex + 1;
        accumulator.addSample(sampleCount, ray.getPayloadAccumulatiion());

        // TODO: visualize high variance

        // TODO: russian roulette early exit?
    }

    randgen_type randGen;
    material_system_type materialSystem;
    nee_type nee;
    scene_type scene;

    scalar_type bxdfPdfThreshold;
    scalar_type lumaContributionThreshold; // OETF smallest perceptible value
    measure_type spectralTypeToLumaCoeffs;
};

}
}
}

#endif
