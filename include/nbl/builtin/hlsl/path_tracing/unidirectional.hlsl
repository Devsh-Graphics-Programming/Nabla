#ifndef _NBL_BUILTIN_HLSL_PATH_TRACING_UNIDIRECTIONAL_INCLUDED_
#define _NBL_BUILTIN_HLSL_PATH_TRACING_UNIDIRECTIONAL_INCLUDED_

#include <nbl/builtin/hlsl/colorspace/EOTF.hlsl>
#include <nbl/builtin/hlsl/colorspace/encodeCIEXYZ.hlsl>
#include <nbl/builtin/hlsl/math/functions.hlsl>
#include <nbl/builtin/hlsl/sampling/basic.hlsl>
#include <nbl/builtin/hlsl/bxdf/bxdf_traits.hlsl>
#include <nbl/builtin/hlsl/sampling/quantized_sequence.hlsl>
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
    using output_storage_type = typename Accumulator::output_storage_type; // ?
    using sample_type = typename NextEventEstimator::sample_type;
    using ray_dir_info_type = typename sample_type::ray_dir_info_type;
    using ray_type = typename RayGen::ray_type;
    using object_handle_type = typename Intersector::object_handle_type;
    using light_type = typename NextEventEstimator::light_type;
    using bxdfnode_type = typename MaterialSystem::bxdfnode_type;
    using anisotropic_interaction_type = typename MaterialSystem::anisotropic_interaction_type;
    using isotropic_interaction_type = typename anisotropic_interaction_type::isotropic_interaction_type;
    using anisocache_type = typename MaterialSystem::anisocache_type;
    using isocache_type = typename anisocache_type::isocache_type;
    using quotient_pdf_type = typename NextEventEstimator::quotient_pdf_type;

    using diffuse_op_type = typename MaterialSystem::diffuse_op_type;
    using conductor_op_type = typename MaterialSystem::conductor_op_type;
    using dielectric_op_type = typename MaterialSystem::dielectric_op_type;

    vector3_type rand3d(uint32_t protoDimension, uint32_t _sample, uint32_t i)
    {
        using sequence_type = sampling::QuantizedSequence<uint32_t2,3>;
        uint32_t address = glsl::bitfieldInsert<uint32_t>(protoDimension, _sample, MAX_DEPTH_LOG2, MAX_SAMPLES_LOG2);
        sequence_type tmpSeq = vk::RawBufferLoad<sequence_type>(pSampleBuffer + (address + i) * sizeof(sequence_type));
        return tmpSeq.template decode<float32_t>(randGen());
    }

    scalar_type getLuma(NBL_CONST_REF_ARG(vector3_type) col)
    {
        return hlsl::dot<vector3_type>(colorspace::scRGBtoXYZ[1], col);
    }

    // TODO: probably will only work with isotropic surfaces, need to do aniso
    bool closestHitProgram(uint32_t depth, uint32_t _sample, NBL_REF_ARG(ray_type) ray, NBL_CONST_REF_ARG(scene_type) scene)
    {
        const object_handle_type objectID = ray.objectID;
        const vector3_type intersection = ray.origin + ray.direction * ray.intersectionT;

        uint32_t bsdfLightIDs = scene.getBsdfLightIDs(objectID);
        vector3_type N = scene.getNormal(objectID, intersection);
        N = nbl::hlsl::normalize(N);
        ray_dir_info_type V;
        V.setDirection(-ray.direction);
        isotropic_interaction_type iso_interaction = isotropic_interaction_type::create(V, N);
        iso_interaction.luminosityContributionHint = colorspace::scRGBtoXYZ[1];
        anisotropic_interaction_type interaction = anisotropic_interaction_type::create(iso_interaction);

        vector3_type throughput = ray.payload.throughput;

        // emissive
        const uint32_t lightID = glsl::bitfieldExtract(bsdfLightIDs, 16, 16);
        if (lightID != light_type::INVALID_ID)
        {
            scalar_type _pdf;
            measure_type emissive = nee.deferredEvalAndPdf(_pdf, scene, lightID, ray) * throughput;
            scalar_type _pdfSq = hlsl::mix(_pdf, _pdf * _pdf, _pdf < numeric_limits<scalar_type>::max);
            emissive /= (1.0 + _pdfSq * ray.payload.otherTechniqueHeuristic);
            ray.payload.accumulation += emissive;
        }

        const uint32_t bsdfID = glsl::bitfieldExtract(bsdfLightIDs, 0, 16);
        if (bsdfID == bxdfnode_type::INVALID_ID)
            return false;

        bxdfnode_type bxdf = materialSystem.bxdfs[bsdfID];

        const bool isBSDF = materialSystem.isBSDF(bsdfID);

        vector3_type eps0 = rand3d(depth, _sample, 0u);
        vector3_type eps1 = rand3d(depth, _sample, 1u);

        // thresholds
        const scalar_type bxdfPdfThreshold = 0.0001;
        const scalar_type lumaContributionThreshold = getLuma(colorspace::eotf::sRGB<vector3_type>((vector3_type)1.0 / 255.0)); // OETF smallest perceptible value
        const vector3_type throughputCIE_Y = colorspace::sRGBtoXYZ[1] * throughput;    // TODO: this only works if spectral_type is dim 3
        const measure_type eta = bxdf.params.ior1 / bxdf.params.ior0;
        const scalar_type monochromeEta = hlsl::dot<vector3_type>(throughputCIE_Y, eta) / (throughputCIE_Y.r + throughputCIE_Y.g + throughputCIE_Y.b);  // TODO: imaginary eta?

        // sample lights
        const scalar_type neeProbability = bxdf.getNEEProb();
        scalar_type rcpChoiceProb;
        sampling::PartitionRandVariable<scalar_type> partitionRandVariable;
        partitionRandVariable.leftProb = neeProbability;
        if (!partitionRandVariable(eps0.z, rcpChoiceProb) && depth < 2u)
        {
            uint32_t randLightID = uint32_t(float32_t(randGen.rng()) / numeric_limits<uint32_t>::max) * nee.lightCount;
            quotient_pdf_type neeContrib_pdf;
            scalar_type t;
            sample_type nee_sample = nee.generate_and_quotient_and_pdf(
                neeContrib_pdf, t,
                scene, randLightID, intersection, interaction,
                isBSDF, eps0, depth
            );

            // We don't allow non watertight transmitters in this renderer
            bool validPath = nee_sample.getNdotL() > numeric_limits<scalar_type>::min && nee_sample.isValid();
            // but if we allowed non-watertight transmitters (single water surface), it would make sense just to apply this line by itself
            bxdf::fresnel::OrientedEtas<monochrome_type> orientedEta = bxdf::fresnel::OrientedEtas<monochrome_type>::create(interaction.getNdotV(), hlsl::promote<monochrome_type>(monochromeEta));
            anisocache_type _cache = anisocache_type::template create<anisotropic_interaction_type, sample_type>(interaction, nee_sample, orientedEta);
            validPath = validPath && _cache.getAbsNdotH() >= 0.0;
            materialSystem.bxdfs[bsdfID].params.eta = monochromeEta;

            if (neeContrib_pdf.pdf < numeric_limits<scalar_type>::max)
            {
                if (validPath)
                {
                    // example only uses isotropic bxdfs
                    quotient_pdf_type bsdf_quotient_pdf = materialSystem.quotient_and_pdf(bsdfID, nee_sample, interaction.isotropic, _cache.iso_cache);
                    neeContrib_pdf.quotient *= bxdf.albedo * throughput * bsdf_quotient_pdf.quotient;
                    const scalar_type otherGenOverChoice = bsdf_quotient_pdf.pdf * rcpChoiceProb;
                    const scalar_type otherGenOverLightAndChoice = otherGenOverChoice / bsdf_quotient_pdf.pdf;
                    neeContrib_pdf.quotient *= otherGenOverChoice / (1.f + otherGenOverLightAndChoice * otherGenOverLightAndChoice);   // balance heuristic

                    ray_type nee_ray;
                    nee_ray.origin = intersection + nee_sample.getL().getDirection() * t * Tolerance<scalar_type>::getStart(depth);
                    nee_ray.direction = nee_sample.getL().getDirection();
                    nee_ray.intersectionT = t;
                    if (bsdf_quotient_pdf.pdf < numeric_limits<scalar_type>::max && getLuma(neeContrib_pdf.quotient) > lumaContributionThreshold && intersector_type::traceRay(nee_ray, scene).id == -1)
                        ray.payload.accumulation += neeContrib_pdf.quotient;
                }
            }
        }

        // sample BSDF
        scalar_type bxdfPdf;
        vector3_type bxdfSample;
        {
            anisocache_type _cache;
            sample_type bsdf_sample = materialSystem.generate(bsdfID, interaction, eps1, _cache);

            if (!bsdf_sample.isValid())
                return false;

            // example only uses isotropic bxdfs
            // the value of the bsdf divided by the probability of the sample being generated
            quotient_pdf_type bsdf_quotient_pdf = materialSystem.quotient_and_pdf(bsdfID, bsdf_sample, interaction.isotropic, _cache.iso_cache);
            throughput *= bxdf.albedo * bsdf_quotient_pdf.quotient;
            bxdfPdf = bsdf_quotient_pdf.pdf;
            bxdfSample = bsdf_sample.getL().getDirection();
        }

        // additional threshold
        const float lumaThroughputThreshold = lumaContributionThreshold;
        if (bxdfPdf > bxdfPdfThreshold && getLuma(throughput) > lumaThroughputThreshold)
        {
            ray.payload.throughput = throughput;
            scalar_type otherTechniqueHeuristic = neeProbability / bxdfPdf; // numerically stable, don't touch
            ray.payload.otherTechniqueHeuristic = otherTechniqueHeuristic * otherTechniqueHeuristic;

            // trace new ray
            ray.origin = intersection + bxdfSample * (1.0/*kSceneSize*/) * Tolerance<scalar_type>::getStart(depth);
            ray.direction = bxdfSample;
            NBL_IF_CONSTEXPR (nee_type::IsPolygonMethodProjectedSolidAngle)
            {
                ray.normalAtOrigin = interaction.getN();
                ray.wasBSDFAtOrigin = isBSDF;
            }
            return true;
        }

        return false;
    }

    void missProgram(NBL_REF_ARG(ray_type) ray)
    {
        vector3_type finalContribution = ray.payload.throughput;
        // #ifdef USE_ENVMAP
        //     vec2 uv = SampleSphericalMap(ray.direction);
        //     finalContribution *= textureLod(envMap, uv, 0.0).rgb;
        // #else
        const vector3_type kConstantEnvLightRadiance = vector3_type(0.15, 0.21, 0.3);   // TODO: match spectral_type
        finalContribution *= kConstantEnvLightRadiance;
        ray.payload.accumulation += finalContribution;
        // #endif
    }

    // Li
    void sampleMeasure(uint32_t sampleIndex, uint32_t maxDepth, NBL_CONST_REF_ARG(scene_type) scene, NBL_REF_ARG(Accumulator) accumulator)
    {
        //scalar_type meanLumaSq = 0.0;
        vector3_type uvw = rand3d(0u, sampleIndex, 0u);
        ray_type ray = rayGen.generate(uvw);
        ray.initPayload();

        NBL_IF_CONSTEXPR (nee_type::IsPolygonMethodProjectedSolidAngle)
        {
            ray.normalAtOrigin = hlsl::promote<vector3_type>(0.0);
            ray.wasBSDFAtOrigin = false;
        }

        // bounces
        bool hit = true;
        bool rayAlive = true;
        for (int d = 1; (d <= maxDepth) && hit && rayAlive; d += 2)
        {
            ray.intersectionT = numeric_limits<scalar_type>::max;
            ray.objectID = intersector_type::traceRay(ray, scene);

            hit = ray.objectID.id != -1;
            if (hit)
                rayAlive = closestHitProgram(1, sampleIndex, ray, scene);
        }
        if (!hit)
            missProgram(ray);

        const uint32_t sampleCount = sampleIndex + 1;
        accumulator.addSample(sampleCount, ray.payload.accumulation);

        // TODO: visualize high variance

        // TODO: russian roulette early exit?
    }

    NBL_CONSTEXPR_STATIC_INLINE uint32_t MAX_DEPTH_LOG2 = 4u;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t MAX_SAMPLES_LOG2 = 10u;

    randgen_type randGen;
    raygen_type rayGen;
    material_system_type materialSystem;
    nee_type nee;

    uint64_t pSampleBuffer;
};

}
}
}

#endif
