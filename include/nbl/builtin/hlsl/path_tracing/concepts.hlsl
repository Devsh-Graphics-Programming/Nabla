#ifndef _NBL_BUILTIN_HLSL_PATH_TRACING_CONCEPTS_INCLUDED_
#define _NBL_BUILTIN_HLSL_PATH_TRACING_CONCEPTS_INCLUDED_

#include <nbl/builtin/hlsl/concepts.hlsl>

namespace nbl
{
namespace hlsl
{
namespace path_tracing
{
namespace concepts
{

#define NBL_CONCEPT_NAME RandGenerator
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (rand, T)
NBL_CONCEPT_BEGIN(1)
#define rand NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::rng_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::return_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((rand()), ::nbl::hlsl::is_same_v, typename T::return_type))
);
#undef rand
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

#define NBL_CONCEPT_NAME RayGenerator
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (raygen, T)
#define NBL_CONCEPT_PARAM_1 (randVec, typename T::vector3_type)
NBL_CONCEPT_BEGIN(2)
#define raygen NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define randVec NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::vector3_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::ray_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((raygen.generate(randVec)), ::nbl::hlsl::is_same_v, typename T::ray_type))
);
#undef randVec
#undef raygen
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

#define NBL_CONCEPT_NAME Intersector
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (intersect, T)
#define NBL_CONCEPT_PARAM_1 (ray, typename T::ray_type)
#define NBL_CONCEPT_PARAM_2 (scene, typename T::scene_type)
NBL_CONCEPT_BEGIN(3)
#define intersect NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define ray NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define scene NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::scene_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::ray_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::object_handle_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((intersect.traceRay(ray, scene)), ::nbl::hlsl::is_same_v, typename T::object_handle_type))
);
#undef scene
#undef ray
#undef intersect
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

#define NBL_CONCEPT_NAME MaterialSystem
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (matsys, T)
#define NBL_CONCEPT_PARAM_1 (_sample, typename T::sample_type)
#define NBL_CONCEPT_PARAM_2 (matid, typename T::material_id_type)
#define NBL_CONCEPT_PARAM_3 (aniso_inter, typename T::anisotropic_interaction_type)
#define NBL_CONCEPT_PARAM_4 (iso_inter, typename T::isotropic_interaction_type)
#define NBL_CONCEPT_PARAM_5 (aniso_cache, typename T::anisocache_type)
#define NBL_CONCEPT_PARAM_6 (iso_cache, typename T::isocache_type)
#define NBL_CONCEPT_PARAM_7 (params, typename T::create_params_t)
#define NBL_CONCEPT_PARAM_8 (u, typename T::vector3_type)
NBL_CONCEPT_BEGIN(9)
#define matsys NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define _sample NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define matid NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
#define aniso_inter NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_3
#define iso_inter NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_4
#define aniso_cache NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_5
#define iso_cache NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_6
#define params NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_7
#define u NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_8
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::vector3_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::material_id_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::sample_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::quotient_pdf_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::measure_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::anisotropic_interaction_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::isotropic_interaction_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::anisocache_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::isocache_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::create_params_t))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((matsys.eval(matid, params, _sample, iso_inter, iso_cache)), ::nbl::hlsl::is_same_v, typename T::measure_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((matsys.generate(matid, params, aniso_inter, u, aniso_cache)), ::nbl::hlsl::is_same_v, typename T::sample_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((matsys.quotient_and_pdf(matid, params, _sample, iso_inter, iso_cache)), ::nbl::hlsl::is_same_v, typename T::quotient_pdf_type))
);
#undef u
#undef params
#undef iso_cache
#undef aniso_cache
#undef iso_inter
#undef aniso_inter
#undef matid
#undef _sample
#undef matsys
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

#define NBL_CONCEPT_NAME NextEventEstimator
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (nee, T)
#define NBL_CONCEPT_PARAM_1 (ray, typename T::ray_type)
#define NBL_CONCEPT_PARAM_2 (scene, typename T::scene_type)
#define NBL_CONCEPT_PARAM_3 (id, uint32_t)
#define NBL_CONCEPT_PARAM_4 (pdf, typename T::scalar_type)
#define NBL_CONCEPT_PARAM_5 (quo_pdf, typename T::quotient_pdf_type)
#define NBL_CONCEPT_PARAM_6 (v, typename T::vector3_type)
#define NBL_CONCEPT_PARAM_7 (interaction, typename T::interaction_type)
#define NBL_CONCEPT_PARAM_8 (b, bool)
NBL_CONCEPT_BEGIN(9)
#define nee NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define ray NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define scene NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
#define id NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_3
#define pdf NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_4
#define quo_pdf NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_5
#define v NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_6
#define interaction NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_7
#define b NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_8
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::scalar_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::vector3_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::scene_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::light_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::ray_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::spectral_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::sample_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::quotient_pdf_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::interaction_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((nee.deferredEvalAndPdf(pdf, scene, id, ray)), ::nbl::hlsl::is_same_v, typename T::spectral_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((nee.generate_and_quotient_and_pdf(quo_pdf, pdf, scene, id, v, interaction, b, v, id)), ::nbl::hlsl::is_same_v, typename T::sample_type))
);
#undef b
#undef interaction
#undef v
#undef quo_pdf
#undef pdf
#undef id
#undef scene
#undef ray
#undef nee
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

#define NBL_CONCEPT_NAME Accumulator
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (acc, T)
#define NBL_CONCEPT_PARAM_1 (sampleCount, uint32_t)
#define NBL_CONCEPT_PARAM_2 (_sample, typename T::input_sample_type)
NBL_CONCEPT_BEGIN(3)
#define acc NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define sampleCount NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define _sample NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::input_sample_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((acc.addSample(sampleCount, _sample)), ::nbl::hlsl::is_same_v, void))
);
#undef _sample
#undef sampleCount
#undef acc
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

#define NBL_CONCEPT_NAME Scene
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (scene, T)
#define NBL_CONCEPT_PARAM_1 (intersectP, typename T::vector3_type)
#define NBL_CONCEPT_PARAM_2 (id, typename T::object_handle_type)
NBL_CONCEPT_BEGIN(3)
#define scene NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define intersectP NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define id NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::vector3_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::object_handle_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((scene.getBsdfLightIDs(id)), ::nbl::hlsl::is_same_v, uint32_t))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((scene.getNormal(id, intersectP)), ::nbl::hlsl::is_same_v, typename T::vector3_type))
);
#undef id
#undef intersectP
#undef scene
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

}
}
}
}

#endif
