// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_QUOTIENT_AND_PDF_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_QUOTIENT_AND_PDF_INCLUDED_

#include "nbl/builtin/hlsl/concepts/vector.hlsl"

namespace nbl
{
namespace hlsl
{
namespace sampling
{

template<typename T, typename F>
NBL_BOOL_CONCEPT spectral_of = concepts::Vectorial<T> || concepts::Scalar<T> || concepts::Scalar<F>;

// finally fixed the semantic F-up, value/pdf = quotient not remainder
template<typename Q, typename P NBL_PRIMARY_REQUIRES(spectral_of<Q,P> && is_floating_point_v<P>)
struct quotient_and_pdf
{
    using this_t = quotient_and_pdf<Q, P>;
    static this_t create(NBL_CONST_REF_ARG(Q) _quotient, NBL_CONST_REF_ARG(P) _pdf)
    {
        this_t retval;
        retval.quotient = _quotient;
        retval.pdf = _pdf;
        return retval;
    }

    Q value()
    {
        return quotient*pdf;
    }

    Q quotient;
    P pdf;
};

typedef quotient_and_pdf<float32_t, float32_t> quotient_and_pdf_scalar;
typedef quotient_and_pdf<vector<float32_t, 3>, float32_t> quotient_and_pdf_rgb;

}
}
}

#endif
