// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_QUOTIENT_AND_PDF_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_QUOTIENT_AND_PDF_INCLUDED_

#include "nbl/builtin/hlsl/concepts/vector.hlsl"
#include "nbl/builtin/hlsl/vector_utils/vector_traits.hlsl"

namespace nbl
{
namespace hlsl
{
namespace sampling
{

// finally fixed the semantic F-up, value/pdf = quotient not remainder
template<typename Q, typename P NBL_PRIMARY_REQUIRES(concepts::FloatingPointLikeVectorial<Q> && concepts::FloatingPointLikeScalar<P>)
struct quotient_and_pdf
{
    using this_t = quotient_and_pdf<Q, P>;
    using scalar_q = typename vector_traits<Q>::scalar_type;

    static this_t create(const Q _quotient, const P _pdf)
    {
        this_t retval;
        retval._quotient = _quotient;
        retval._pdf = _pdf;
        return retval;
    }

    static this_t create(const scalar_q _quotient, const P _pdf)
    {
        this_t retval;
        retval._quotient = hlsl::promote<Q>(_quotient);
        retval._pdf = _pdf;
        return retval;
    }

    Q quotient() NBL_CONST_MEMBER_FUNC { return _quotient; }
    P pdf() NBL_CONST_MEMBER_FUNC { return _pdf; }

    Q value() NBL_CONST_MEMBER_FUNC
    {
        return _quotient * _pdf;
    }

    Q _quotient;
    P _pdf;
};

}
}
}

#endif
