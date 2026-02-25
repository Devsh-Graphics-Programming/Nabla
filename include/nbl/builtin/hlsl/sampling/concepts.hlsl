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
// SampleWithPDF
//
// Checks that a sample type bundles a value with its PDF.
//
// Required members/methods:
//   value  - the sampled value (member or method)
//   pdf    - the probability density
//
// Satisfied by: codomain_and_pdf, domain_and_pdf, quotient_and_pdf
// ============================================================================

// clang-format off
#define NBL_CONCEPT_NAME SampleWithPDF
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (s, T)
NBL_CONCEPT_BEGIN(1)
#define s NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
NBL_CONCEPT_END(
	((NBL_CONCEPT_REQ_EXPR)(s.pdf))
    ((NBL_CONCEPT_REQ_EXPR)(s.value)));
#undef s
#include <nbl/builtin/hlsl/concepts/__end.hlsl>
// clang-format on

// ============================================================================
// SampleWithRcpPDF
//
// Checks that a sample type bundles a value with its reciprocal PDF.
//
// Required members/methods:
//   value  - the sampled value (member or method)
//   rcpPdf - the reciprocal probability density
//
// Satisfied by: codomain_and_rcpPdf, domain_and_rcpPdf
// ============================================================================

// clang-format off
#define NBL_CONCEPT_NAME SampleWithRcpPDF
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (s, T)
NBL_CONCEPT_BEGIN(1)
#define s NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
NBL_CONCEPT_END(
	((NBL_CONCEPT_REQ_EXPR)(s.rcpPdf))
    ((NBL_CONCEPT_REQ_EXPR)(s.value)));
#undef s
#include <nbl/builtin/hlsl/concepts/__end.hlsl>
// clang-format on

// ============================================================================
// SampleWithDensity
//
// A sample type that bundles a value with either its PDF or reciprocal PDF.
// This is the disjunction of SampleWithPDF and SampleWithRcpPDF.
// ============================================================================
template<typename T>
NBL_BOOL_CONCEPT SampleWithDensity = SampleWithPDF<T> || SampleWithRcpPDF<T>;

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
// A _sampler whose density can be computed analytically in the forward
// (sampling) direction. The generate method returns the sample bundled
// with its density to avoid redundant computation.
//
// Required types:
//   domain_type     - the input space
//   codomain_type   - the output space
//   density_type    - the density type
//   sample_type     - bundled return of generate, must satisfy
//                     SampleWithDensity (i.e. SampleWithPDF or SampleWithRcpPDF)
//
// Required methods:
//   sample_type  generate(domain_type u)    - sample + density
//   density_type forwardPdf(domain_type u)  - density only
// ============================================================================

// clang-format off
#define NBL_CONCEPT_NAME TractableSampler
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
    ((NBL_CONCEPT_REQ_TYPE)(T::density_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::sample_type))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(SampleWithDensity, typename T::sample_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE) ((_sampler.generate(u)), ::nbl::hlsl::is_same_v, typename T::sample_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE) ((_sampler.forwardPdf(u)), ::nbl::hlsl::is_same_v, typename T::density_type)));
#undef u
#undef _sampler
#include <nbl/builtin/hlsl/concepts/__end.hlsl>
// clang-format on

// ============================================================================
// ResamplableSampler
//
// Extends TractableSampler with the ability to evaluate the PDF given
// a codomain value (i.e. without knowing the original domain input).
//
// Required methods (in addition to TractableSampler):
//   density_type backwardPdf(codomain_type v)
// ============================================================================

// clang-format off
#define NBL_CONCEPT_NAME ResamplableSampler
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (_sampler, T)
#define NBL_CONCEPT_PARAM_1 (v, typename T::codomain_type)
NBL_CONCEPT_BEGIN(2)
#define _sampler NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define v NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
NBL_CONCEPT_END(
	((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(TractableSampler, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((_sampler.backwardPdf(v)), ::nbl::hlsl::is_same_v, typename T::density_type)));
#undef v
#undef _sampler
#include <nbl/builtin/hlsl/concepts/__end.hlsl>
// clang-format on

// ============================================================================
// BijectiveSampler
//
// The mapping domain <-> codomain is bijective (1:1), so it can be
// inverted. Extends ResamplableSampler with invertGenerate.
//
// Because the mapping is bijective, the Jacobian of the inverse is
// the reciprocal of the Jacobian of the forward mapping:
//   backwardPdf(v) == 1.0 / forwardPdf(invertGenerate(v).value)
//
// Required types (in addition to ResamplableSampler):
//   inverse_sample_type - bundled return of invertGenerate, should be
//                         one of:
//                         domain_and_rcpPdf<domain_type, density_type> (preferred)
//                         domain_and_pdf<domain_type, density_type>
//
// Required methods (in addition to ResamplableSampler):
//   inverse_sample_type invertGenerate(codomain_type v)
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
	((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(ResamplableSampler, T))
    ((NBL_CONCEPT_REQ_TYPE)(T::inverse_sample_type))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(SampleWithDensity, typename T::inverse_sample_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((_sampler.invertGenerate(v)), ::nbl::hlsl::is_same_v, typename T::inverse_sample_type)));
#undef v
#undef _sampler
#include <nbl/builtin/hlsl/concepts/__end.hlsl>
// clang-format on

} // namespace concepts
} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif // _NBL_BUILTIN_HLSL_SAMPLING_CONCEPTS_INCLUDED_
