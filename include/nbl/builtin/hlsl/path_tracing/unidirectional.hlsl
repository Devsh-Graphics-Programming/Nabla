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

// TODO: unsure what to do with this, awaiting refactor or turning into concept
template<typename T>
struct Tolerance
{
    NBL_CONSTEXPR_STATIC_INLINE T INTERSECTION_ERROR_BOUND_LOG2 = -8.0;

    static T __common(uint32_t depth)
    {
        T depthRcp = 1.0 / T(depth);
        return INTERSECTION_ERROR_BOUND_LOG2;
    }

    static T getStart(uint32_t depth)
    {
        return nbl::hlsl::exp2(__common(depth));
    }

    static T getEnd(uint32_t depth)
    {
        return 1.0 - nbl::hlsl::exp2(__common(depth) + 1.0);
    }
};

template<class RandGen, class RayGen, class Intersector, class MaterialSystem, /* class PathGuider, */ class NextEventEstimator, class Accumulator, class Scene
NBL_PRIMARY_REQUIRES(concepts::RandGenerator<RandGen> && concepts::RayGenerator<RayGen> &&
    concepts::Intersector<Intersector> && concepts::MaterialSystem<MaterialSystem> &&
    concepts::NextEventEstimator<NextEventEstimator> && concepts::Accumulator<Accumulator> &&
    concepts::Scene<Scene>)
struct Unidirectional
{
    using this_t = Unidirectional<RandGen, RayGen, Intersector, MaterialSystem, NextEventEstimator, Accumulator, Scene>;
    using randgen_type = RandGen;
    using raygen_type = RayGen;
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
    using ray_type = typename RayGen::ray_type;
    using object_handle_type = typename Intersector::object_handle_type;
    using intersect_data_type = typename Intersector::intersect_data_type;
    using bxdfnode_type = typename MaterialSystem::bxdfnode_type;
    using anisotropic_interaction_type = typename MaterialSystem::anisotropic_interaction_type;
    using isotropic_interaction_type = typename anisotropic_interaction_type::isotropic_interaction_type;
    using anisocache_type = typename MaterialSystem::anisocache_type;
    using quotient_pdf_type = typename NextEventEstimator::quotient_pdf_type;

    scalar_type getLuma(NBL_CONST_REF_ARG(vector3_type) col)
    {
        return hlsl::dot(spectralTypeToLumaCoeffs, col);
    }

    // TODO: will only work with isotropic surfaces, need to do aniso
    bool closestHitProgram(uint32_t depth, uint32_t _sample, NBL_REF_ARG(ray_type) ray, NBL_CONST_REF_ARG(intersect_data_type) intersectData)
    {
        const vector3_type intersection = intersectData.intersection;
        vector3_type throughput = ray.payload.throughput;
        const vector3_type throughputCIE_Y = hlsl::normalize(spectralTypeToLumaCoeffs * throughput);

        isotropic_interaction_type iso_interaction = intersectData.iso_interaction;
        anisotropic_interaction_type interaction = intersectData.aniso_interaction;
        iso_interaction.luminosityContributionHint = throughputCIE_Y;
        interaction.isotropic.luminosityContributionHint = throughputCIE_Y;

        // emissive
        typename scene_type::mat_light_id_type matLightID = scene.getMatLightIDs(ray.objectID);
        const uint32_t matID = matLightID.matID;
        const bool isEmissive = materialSystem.hasEmission(matID);
        if (isEmissive)
        {
            measure_type emissive = materialSystem.getEmission(matID, interaction.isotropic);

            const uint32_t lightID = matLightID.lightID;
            if (matLightID.isLight())
            {
                const scalar_type pdf = nee.deferred_pdf(lightID, ray, scene);
                scalar_type pdfSq = hlsl::mix(pdf, pdf * pdf, pdf < numeric_limits<scalar_type>::max);
                emissive *= ray.foundEmissiveMIS(pdfSq);
            }
            ray.addPayloadContribution(emissive);
        }

        if (!matLightID.isBxDF() || isEmissive)
            return false;

        bxdfnode_type bxdf = materialSystem.bxdfs[matID];
        const bool isBSDF = materialSystem.isBSDF(matID);

        vector3_type eps0 = randGen(depth * 2u, _sample, 0u);
        vector3_type eps1 = randGen(depth * 2u + 1u, _sample, 1u);

        // thresholds
        const measure_type eta = bxdf.params.ior1 / bxdf.params.ior0;
        const scalar_type monochromeEta = hlsl::dot<vector3_type>(throughputCIE_Y, eta) / (throughputCIE_Y.r + throughputCIE_Y.g + throughputCIE_Y.b);  // TODO: imaginary eta?

        // sample lights
        const scalar_type neeProbability = bxdf.getNEEProb();
        scalar_type rcpChoiceProb;
        sampling::PartitionRandVariable<scalar_type> partitionRandVariable;
        partitionRandVariable.leftProb = neeProbability;
        assert(neeProbability >= 0.0 && neeProbability <= 1.0)
        if (!partitionRandVariable(eps0.z, rcpChoiceProb))
        {
            typename nee_type::sample_quotient_return_type ret = nee.template generate_and_quotient_and_pdf<material_system_type>(
                materialSystem, scene, intersection, interaction,
                isBSDF, eps0, depth
            );
            scalar_type t = ret.newRayMaxT;
            sample_type nee_sample = ret.sample_;
            quotient_pdf_type neeContrib = ret.quotient_pdf;

            // We don't allow non watertight transmitters in this renderer
            // but if we allowed non-watertight transmitters (single water surface), it would make sense just to apply this line by itself
            bxdf::fresnel::OrientedEtas<monochrome_type> orientedEta = bxdf::fresnel::OrientedEtas<monochrome_type>::create(interaction.getNdotV(), hlsl::promote<monochrome_type>(monochromeEta));
            anisocache_type _cache = anisocache_type::template create<anisotropic_interaction_type, sample_type>(interaction, nee_sample, orientedEta);
            materialSystem.bxdfs[matID].params.eta = monochromeEta;

            if (neeContrib.pdf > scalar_type(0.0))
            {
                // example only uses isotropic bxdfs
                quotient_pdf_type bsdf_quotient_pdf = materialSystem.quotient_and_pdf(matID, nee_sample, interaction.isotropic, _cache.iso_cache);
                neeContrib.quotient *= materialSystem.eval(matID, nee_sample, interaction.isotropic, _cache.iso_cache) * rcpChoiceProb;
                const scalar_type otherGenOverLightAndChoice = bsdf_quotient_pdf.pdf * rcpChoiceProb / neeContrib.pdf;
                neeContrib.quotient /= 1.f + otherGenOverLightAndChoice * otherGenOverLightAndChoice;   // balance heuristic

                ray_type nee_ray;
                nee_ray.origin = intersection + nee_sample.getL().getDirection() * t * Tolerance<scalar_type>::getStart(depth);
                nee_ray.direction = nee_sample.getL().getDirection();
                nee_ray.intersectionT = t;
                if (getLuma(neeContrib.quotient) > lumaContributionThreshold)
                    ray.addPayloadContribution(neeContrib.quotient * intersector_type::traceShadowRay(nee_ray, scene, ret.lightObjectID));
            }
        }

        // sample BSDF
        scalar_type bxdfPdf;
        vector3_type bxdfSample;
        {
            anisocache_type _cache;
            sample_type bsdf_sample = materialSystem.generate(matID, interaction, eps1, _cache);

            if (!bsdf_sample.isValid())
                return false;

            // example only uses isotropic bxdfs
            // the value of the bsdf divided by the probability of the sample being generated
            quotient_pdf_type bsdf_quotient_pdf = materialSystem.quotient_and_pdf(matID, bsdf_sample, interaction.isotropic, _cache.iso_cache);
            throughput *= bsdf_quotient_pdf.quotient;
            bxdfPdf = bsdf_quotient_pdf.pdf;
            bxdfSample = bsdf_sample.getL().getDirection();
        }

        // additional threshold
        const float lumaThroughputThreshold = lumaContributionThreshold;
        if (bxdfPdf > bxdfPdfThreshold && getLuma(throughput) > lumaThroughputThreshold)
        {
            scalar_type otherTechniqueHeuristic = neeProbability / bxdfPdf; // numerically stable, don't touch
            ray.setPayloadMISWeights(throughput, otherTechniqueHeuristic * otherTechniqueHeuristic);

            // trace new ray
            vector3_type origin = intersection + bxdfSample * (1.0/*kSceneSize*/) * Tolerance<scalar_type>::getStart(depth);
            vector3_type direction = bxdfSample;

            ray.initData(origin, direction, interaction.getN(), isBSDF);
            return true;
        }

        return false;
    }

    void missProgram(NBL_REF_ARG(ray_type) ray)
    {
        vector3_type finalContribution = ray.payload.throughput;
        finalContribution *= nee.get_environment_radiance(ray);
        ray.addPayloadContribution(finalContribution);
    }

    // Li
    void sampleMeasure(uint32_t sampleIndex, uint32_t maxDepth, NBL_REF_ARG(Accumulator) accumulator)
    {
        //scalar_type meanLumaSq = 0.0;
        vector3_type uvw = randGen(0u, sampleIndex, 0u);
        ray_type ray = rayGen.generate(uvw);
        ray.initPayload();

        // bounces
        bool hit = true;
        bool rayAlive = true;
        for (int d = 1; (d <= maxDepth) && hit && rayAlive; d++)
        {
            ray.intersectionT = numeric_limits<scalar_type>::max;
            intersect_data_type intersection = intersector_type::traceRay(ray, scene);

            hit = intersection.foundHit;
            if (hit)
                rayAlive = closestHitProgram(d, sampleIndex, ray, intersection);
        }
        if (!hit)
            missProgram(ray);

        const uint32_t sampleCount = sampleIndex + 1;
        accumulator.addSample(sampleCount, ray.payload.accumulation);

        // TODO: visualize high variance

        // TODO: russian roulette early exit?
    }

    NBL_CONSTEXPR_STATIC_INLINE uint32_t MaxDepthLog2 = 4u;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t MaxSamplesLog2 = 10u;

    randgen_type randGen;
    raygen_type rayGen;
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
