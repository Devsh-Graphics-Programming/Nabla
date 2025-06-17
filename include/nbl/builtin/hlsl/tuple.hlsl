// Copyright (C) 2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_TUPLE_INCLUDED_
#define _NBL_BUILTIN_HLSL_TUPLE_INCLUDED_

#include "nbl/builtin/hlsl/type_traits.hlsl"

namespace nbl
{
namespace hlsl
{

template<typename T0, typename T1=void, typename T2=void> // TODO: in the future use BOOST_PP to make this
struct tuple
{
    T0 t0;
    T1 t1;
    T2 t2;
};

template<uint32_t N, typename Tuple>
struct tuple_element;

template<typename T0>
struct tuple<T0,void,void>
{
   T0 t0;
};

template<typename T0, typename T1>
struct tuple<T0,T1,void>
{
   T0 t0;
   T1 t1;
};
// specializations for less and less void elements

// base case
template<typename Head, typename T1, typename T2>
struct tuple_element<0,tuple<Head,T1,T2> >
{
   using type = Head;
};

template<typename T0, typename Head, typename T2>
struct tuple_element<1,tuple<T0,Head,T2> >
{
   using type = Head;
};

template<typename T0, typename T1, typename Head>
struct tuple_element<2,tuple<T0,T1,Head> >
{
   using type = Head;
};

}
}

#endif
