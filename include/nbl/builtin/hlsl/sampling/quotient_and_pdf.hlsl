// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SAMPLING_QUOTIENT_AND_PDF_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_QUOTIENT_AND_PDF_INCLUDED_


#include "nbl/builtin/hlsl/sampling/quotient_and_weight.hlsl"


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
    using this_t = quotient_and_pdf<Q,P>;
    using base_t = quotient_and_weight<Q,P>;
    using scalar_q = typename base_t::scalar_q;

    static this_t create(const Q _quotient, const P _pdf)
    {
        this_t retval;
        retval._base._quotient = _quotient;
        retval._base._weight = _pdf;
        return retval;
    }

    static this_t create(const scalar_q _quotient, const P _pdf)
    {
        this_t retval;
        retval._base._quotient = hlsl::promote<Q>(_quotient);
        retval._base._weight = _pdf;
        return retval;
    }

    Q quotient() NBL_CONST_MEMBER_FUNC { return _base.quotient(); }
    P pdf() NBL_CONST_MEMBER_FUNC { return _base._weight; }

    Q value() NBL_CONST_MEMBER_FUNC
    {
        return quotient() * pdf();
    }

    base_t _base;
};

}
}
}

#endif
