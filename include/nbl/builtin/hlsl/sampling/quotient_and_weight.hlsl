// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_QUOTIENT_AND_WEIGHT_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_QUOTIENT_AND_WEIGHT_INCLUDED_

#include "nbl/builtin/hlsl/concepts/vector.hlsl"
#include "nbl/builtin/hlsl/vector_utils/vector_traits.hlsl"

namespace nbl
{
namespace hlsl
{
namespace sampling
{

// finally fixed the semantic mixup, value/pdf = quotient not remainder
template<typename Q, typename W NBL_PRIMARY_REQUIRES(concepts::FloatingPointLikeVectorial<Q> && concepts::FloatingPointLikeScalar<W>)
struct quotient_and_weight
{
    using this_t = quotient_and_weight<Q,W>;
    using scalar_q = typename vector_traits<Q>::scalar_type;

    static this_t create(const Q _quotient, const W _weight)
    {
        this_t retval;
        retval._quotient = _quotient;
        retval._weight = _weight;
        return retval;
    }

    static this_t create(const scalar_q _quotient, const W _weight)
    {
        this_t retval;
        retval._quotient = hlsl::promote<Q>(_quotient);
        retval._weight = _weight;
        return retval;
    }

    Q quotient() NBL_CONST_MEMBER_FUNC {return _quotient;}
    W weight() NBL_CONST_MEMBER_FUNC {return _weight;}

    Q _quotient;
    W _weight;
};

}
}
}

#endif
