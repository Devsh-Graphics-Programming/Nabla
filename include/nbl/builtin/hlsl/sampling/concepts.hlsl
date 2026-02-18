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

#define NBL_CONCEPT_NAME BasicSampler
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (sampler, T)
#define NBL_CONCEPT_PARAM_1 (u, typename T::domain_type)
                NBL_CONCEPT_BEGIN(2)
#define sampler NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define u NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
                NBL_CONCEPT_END(
                    ((NBL_CONCEPT_REQ_TYPE)(T::domain_type))
                    ((NBL_CONCEPT_REQ_TYPE)(T::codomain_type))
                    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((sampler.generate(u)), ::nbl::hlsl::is_same_v, typename T::codomain_type))
                );
#undef u
#undef sampler
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

                // ============================================================================
                // TractableSampler
                //
                // A sampler whose density can be computed analytically in the forward
                // (sampling) direction. The generate method returns the sample bundled
                // with its density to avoid redundant computation.
                //
                // Required types:
                //   domain_type     - the input space
                //   codomain_type   - the output space
                //   density_type    - the density type (typically scalar)
                //   sample_type     - bundled return of generate, should be one of:
                //                     codomain_and_rcpPdf<codomain_type, density_type> (preferred)
                //                     codomain_and_pdf<codomain_type, density_type>
                //
                // Required methods:
                //   sample_type  generate(domain_type u)    - sample + density
                //   density_type forwardPdf(domain_type u)  - density only
                // ============================================================================

#define NBL_CONCEPT_NAME TractableSampler
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (sampler, T)
#define NBL_CONCEPT_PARAM_1 (u, typename T::domain_type)
                NBL_CONCEPT_BEGIN(2)
#define sampler NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define u NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
                NBL_CONCEPT_END(
                    ((NBL_CONCEPT_REQ_TYPE)(T::domain_type))
                    ((NBL_CONCEPT_REQ_TYPE)(T::codomain_type))
                    ((NBL_CONCEPT_REQ_TYPE)(T::density_type))
                    ((NBL_CONCEPT_REQ_TYPE)(T::sample_type))
                    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((sampler.generate(u)), ::nbl::hlsl::is_same_v, typename T::sample_type))
                    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((sampler.forwardPdf(u)), ::nbl::hlsl::is_same_v, typename T::density_type))
                );
#undef u
#undef sampler
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

                // ============================================================================
                // BackwardDensitySampler
                //
                // Extends TractableSampler with the ability to evaluate the PDF given
                // a codomain value (i.e. without knowing the original domain input).
                //
                // Required methods (in addition to TractableSampler):
                //   density_type backwardPdf(codomain_type v)
                // ============================================================================

#define NBL_CONCEPT_NAME BackwardDensitySampler
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (sampler, T)
#define NBL_CONCEPT_PARAM_1 (u, typename T::domain_type)
#define NBL_CONCEPT_PARAM_2 (v, typename T::codomain_type)
                NBL_CONCEPT_BEGIN(3)
#define sampler NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define u NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define v NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
                NBL_CONCEPT_END(
                    ((NBL_CONCEPT_REQ_TYPE)(T::domain_type))
                    ((NBL_CONCEPT_REQ_TYPE)(T::codomain_type))
                    ((NBL_CONCEPT_REQ_TYPE)(T::density_type))
                    ((NBL_CONCEPT_REQ_TYPE)(T::sample_type))
                    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((sampler.generate(u)), ::nbl::hlsl::is_same_v, typename T::sample_type))
                    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((sampler.forwardPdf(u)), ::nbl::hlsl::is_same_v, typename T::density_type))
                    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((sampler.backwardPdf(v)), ::nbl::hlsl::is_same_v, typename T::density_type))
                );
#undef v
#undef u
#undef sampler
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

                // ============================================================================
                // BijectiveSampler
                //
                // The mapping domain <-> codomain is bijective (1:1), so it can be
                // inverted. Extends BackwardDensitySampler with invertGenerate.
                //
                // Because the mapping is bijective, the Jacobian of the inverse is
                // the reciprocal of the Jacobian of the forward mapping:
                //   backwardPdf(v) == 1.0 / forwardPdf(invertGenerate(v).value)
                //
                // Required types (in addition to BackwardDensitySampler):
                //   inverse_sample_type - bundled return of invertGenerate, should be
                //                         one of:
                //                         domain_and_rcpPdf<domain_type, density_type> (preferred)
                //                         domain_and_pdf<domain_type, density_type>
                //
                // Required methods (in addition to BackwardDensitySampler):
                //   inverse_sample_type invertGenerate(codomain_type v)
                // ============================================================================

#define NBL_CONCEPT_NAME BijectiveSampler
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (sampler, T)
#define NBL_CONCEPT_PARAM_1 (u, typename T::domain_type)
#define NBL_CONCEPT_PARAM_2 (v, typename T::codomain_type)
                NBL_CONCEPT_BEGIN(3)
#define sampler NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define u NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define v NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
                NBL_CONCEPT_END(
                    ((NBL_CONCEPT_REQ_TYPE)(T::domain_type))
                    ((NBL_CONCEPT_REQ_TYPE)(T::codomain_type))
                    ((NBL_CONCEPT_REQ_TYPE)(T::density_type))
                    ((NBL_CONCEPT_REQ_TYPE)(T::sample_type))
                    ((NBL_CONCEPT_REQ_TYPE)(T::inverse_sample_type))
                    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((sampler.generate(u)), ::nbl::hlsl::is_same_v, typename T::sample_type))
                    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((sampler.forwardPdf(u)), ::nbl::hlsl::is_same_v, typename T::density_type))
                    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((sampler.backwardPdf(v)), ::nbl::hlsl::is_same_v, typename T::density_type))
                    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((sampler.invertGenerate(v)), ::nbl::hlsl::is_same_v, typename T::inverse_sample_type))
                );
#undef v
#undef u
#undef sampler
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

            } // namespace concepts
        } // namespace sampling
    } // namespace hlsl
} // namespace nbl

#endif // _NBL_BUILTIN_HLSL_SAMPLING_CONCEPTS_INCLUDED_
