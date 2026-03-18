#ifndef _NBL_BUILTIN_HLSL_SAMPLING_CONCEPTS_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_CONCEPTS_INCLUDED_

#include <nbl/builtin/hlsl/concepts.hlsl>

namespace nbl
{
namespace hlsl
{
namespace sampling
{
namespace concepts
{

// ============================================================================
// BasicSampler
//
// The simplest sampler: maps domain -> codomain.
//
// Required types:
//   domain_type     - the input space (e.g. float for 1D, float2 for 2D)
//   codomain_type   - the output space (e.g. float3 for directions)
//
// Required methods:
//   codomain_type generate(domain_type u)
// ============================================================================

// clang-format off
#define NBL_CONCEPT_NAME BasicSampler
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (_sampler, T)
#define NBL_CONCEPT_PARAM_1 (u, typename T::domain_type)
NBL_CONCEPT_BEGIN(2)
#define _sampler NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define u NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
NBL_CONCEPT_END(
	((NBL_CONCEPT_REQ_TYPE)(T::domain_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::codomain_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE) ((_sampler.generate(u)), ::nbl::hlsl::is_same_v, typename T::codomain_type)));
#undef u
#undef _sampler
#include <nbl/builtin/hlsl/concepts/__end.hlsl>
// clang-format on

// ============================================================================
// TractableSampler
//
// A sampler whose density can be computed analytically in the forward
// (sampling) direction. generate returns a codomain_type value and writes
// intermediates to a cache_type out-param for later pdf evaluation.
//
// The cache_type out-param stores the input domain_type and/or values
// derived from it during generate, for reuse by forwardPdf without
// redundant recomputation. If there is no common computation between
// generate and backwardPdf, the cache simply stores the codomain value.
//
// For constant-pdf samplers, forwardPdf(cache) == __pdf() (cache ignored).
// For complex samplers (e.g. Cook-Torrance), cache carries intermediate
// values derived from the domain input (e.g. DG1/Fresnel) and forwardPdf
// computes the pdf from those stored intermediates.
//
// Required types:
//   domain_type, codomain_type, density_type, cache_type
//
// Required methods:
//   codomain_type generate(domain_type u, out cache_type cache)
//   density_type  forwardPdf(cache_type cache)
// ============================================================================

// clang-format off
#define NBL_CONCEPT_NAME TractableSampler
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (_sampler, T)
#define NBL_CONCEPT_PARAM_1 (u, typename T::domain_type)
#define NBL_CONCEPT_PARAM_2 (cache, typename T::cache_type)
NBL_CONCEPT_BEGIN(3)
#define _sampler NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define u NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define cache NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
NBL_CONCEPT_END(
	((NBL_CONCEPT_REQ_TYPE)(T::domain_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::codomain_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::density_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::cache_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE) ((_sampler.generate(u, cache)), ::nbl::hlsl::is_same_v, typename T::codomain_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE) ((_sampler.forwardPdf(cache)), ::nbl::hlsl::is_same_v, typename T::density_type)));
#undef cache
#undef u
#undef _sampler
#include <nbl/builtin/hlsl/concepts/__end.hlsl>
// clang-format on

// ============================================================================
// ResamplableSampler
//
// A sampler with forward and backward importance weights, enabling use in
// Multiple Importance Sampling (MIS) and Resampled Importance Sampling (RIS).
//
// Note: resampling does not require tractability - the weights need not be
// normalized probability densities, so this concept is satisfied by
// intractable samplers as well.
//
// Required types:
//   domain_type  - the input space
//   codomain_type - the output space
//   cache_type   - stores intermediates from generate for forward weight reuse
//   weight_type  - the type of the importance weight
//
// Required methods:
//   codomain_type generate(domain_type u, out cache_type cache)
//   weight_type   forwardWeight(cache_type cache)  - forward weight for MIS
//   weight_type   backwardWeight(codomain_type v)  - backward weight for RIS
// ============================================================================

// clang-format off
#define NBL_CONCEPT_NAME ResamplableSampler
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (_sampler, T)
#define NBL_CONCEPT_PARAM_1 (u, typename T::domain_type)
#define NBL_CONCEPT_PARAM_2 (v, typename T::codomain_type)
#define NBL_CONCEPT_PARAM_3 (cache, typename T::cache_type)
NBL_CONCEPT_BEGIN(4)
#define _sampler NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define u NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define v NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
#define cache NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_3
NBL_CONCEPT_END(
	((NBL_CONCEPT_REQ_TYPE)(T::domain_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::codomain_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::cache_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::weight_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((_sampler.generate(u, cache)), ::nbl::hlsl::is_same_v, typename T::codomain_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((_sampler.forwardWeight(cache)), ::nbl::hlsl::is_same_v, typename T::weight_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((_sampler.backwardWeight(v)), ::nbl::hlsl::is_same_v, typename T::weight_type)));
#undef cache
#undef v
#undef u
#undef _sampler
#include <nbl/builtin/hlsl/concepts/__end.hlsl>
// clang-format on

// ============================================================================
// InvertibleSampler
//
// Extends TractableSampler with the ability to evaluate the PDF given
// a codomain value (i.e. without knowing the original domain input).
// The mapping need not be injective — multiple domain elements may map
// to the same codomain output, and backwardPdf accounts for this
// correctly (e.g. by summing contributions).
//
// Also exposes forward and backward importance weights for use in MIS
// and RIS.
//
// Required types (in addition to TractableSampler):
//   weight_type  - the type of the importance weight
//
// Required methods (in addition to TractableSampler):
//   density_type backwardPdf(codomain_type v)   - evaluate pdf at codomain value v
//   weight_type  forwardWeight(cache_type cache) - weight for MIS, reuses generate cache
//   weight_type  backwardWeight(codomain_type v) - weight for RIS, evaluated at v
// ============================================================================

// clang-format off
#define NBL_CONCEPT_NAME InvertibleSampler
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (_sampler, T)
#define NBL_CONCEPT_PARAM_1 (u, typename T::domain_type)
#define NBL_CONCEPT_PARAM_2 (v, typename T::codomain_type)
#define NBL_CONCEPT_PARAM_3 (cache, typename T::cache_type)
NBL_CONCEPT_BEGIN(4)
#define _sampler NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define u NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define v NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
#define cache NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_3
NBL_CONCEPT_END(
	((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(TractableSampler, T))
    ((NBL_CONCEPT_REQ_TYPE)(T::weight_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((_sampler.backwardPdf(v)), ::nbl::hlsl::is_same_v, typename T::density_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((_sampler.forwardWeight(cache)), ::nbl::hlsl::is_same_v, typename T::weight_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((_sampler.backwardWeight(v)), ::nbl::hlsl::is_same_v, typename T::weight_type)));
#undef cache
#undef v
#undef u
#undef _sampler
#include <nbl/builtin/hlsl/concepts/__end.hlsl>
// clang-format on

// ============================================================================
// BijectiveSampler
//
// The mapping domain <-> codomain is bijective (1:1), so it can be
// inverted. Extends InvertibleSampler with generateInverse.
//
// Because the mapping is bijective, the absolute value of the determinant
// of the Jacobian matrix of the inverse equals the reciprocal of the
// absolute value of the determinant of the Jacobian matrix of the forward
// mapping (the Jacobian is an NxM matrix, not a scalar):
//   backwardPdf(v) == 1.0 / forwardPdf(cache)  (where v == generate(u, cache))
//
// Required methods (in addition to InvertibleSampler):
//   domain_type generateInverse(codomain_type v)
// ============================================================================

// clang-format off
#define NBL_CONCEPT_NAME BijectiveSampler
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (_sampler, T)
#define NBL_CONCEPT_PARAM_1 (v, typename T::codomain_type)
NBL_CONCEPT_BEGIN(2)
#define _sampler NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define v NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
NBL_CONCEPT_END(
	((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(InvertibleSampler, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((_sampler.generateInverse(v)), ::nbl::hlsl::is_same_v, typename T::domain_type)));
#undef v
#undef _sampler
#include <nbl/builtin/hlsl/concepts/__end.hlsl>
// clang-format on

} // namespace concepts
} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif // _NBL_BUILTIN_HLSL_SAMPLING_CONCEPTS_INCLUDED_
