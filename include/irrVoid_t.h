// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h
#ifndef __IRR_VOID_T_H_INCLUDED__
#define __IRR_VOID_T_H_INCLUDED__

namespace irr
{
#if __cplusplus >= 201703L
template<typename... Ts>
using void_t = std::void_t<Ts...>;
#else
namespace impl
{
    template<class...>
    struct voider{using type=void;};
}

template<class... T0toN>
using void_t = typename impl::voider<T0toN...>::type;
#endif // C
}

#endif
