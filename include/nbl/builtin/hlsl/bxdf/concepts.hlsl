#ifndef _NBL_BUILTIN_HLSL_BXDF_CONCEPTS_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_CONCEPTS_INCLUDED_

#include <nbl/builtin/hlsl/concepts.hlsl>

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace concepts
{

// ============================================================================
// SampleWithPDF
//
// Checks that a sample type bundles a value with its PDF.
//
// Required methods:
//   value()  - the sampled value
//   pdf()    - the probability density
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
	((NBL_CONCEPT_REQ_EXPR)(s.pdf()))
    ((NBL_CONCEPT_REQ_EXPR)(s.value())));
#undef s
#include <nbl/builtin/hlsl/concepts/__end.hlsl>
// clang-format on

// ============================================================================
// SampleWithRcpPDF
//
// Checks that a sample type bundles a value with its reciprocal PDF.
//
// Required methods:
//   value()   - the sampled value
//   rcpPdf()  - the reciprocal probability density
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
	((NBL_CONCEPT_REQ_EXPR)(s.rcpPdf()))
    ((NBL_CONCEPT_REQ_EXPR)(s.value())));
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

} // namespace concepts
} // namespace bxdf
} // namespace hlsl
} // namespace nbl

#endif // _NBL_BUILTIN_HLSL_BXDF_CONCEPTS_INCLUDED_
