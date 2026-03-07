// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_CONCENTRIC_MAPPING_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_CONCENTRIC_MAPPING_INCLUDED_

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/tgmath.hlsl"
#include "nbl/builtin/hlsl/math/functions.hlsl"

namespace nbl
{
namespace hlsl
{
namespace sampling
{

template<typename T>
vector<T, 2> concentricMapping(const vector<T, 2> _u)
{
	//map [0;1]^2 to [-1;1]^2
	vector<T, 2> u = 2.0f * _u - hlsl::promote<vector<T, 2> >(1.0);

	vector<T, 2> p;
	if (hlsl::all<vector<bool, 2> >(glsl::equal(u, hlsl::promote<vector<T, 2> >(0.0))))
		p = hlsl::promote<vector<T, 2> >(0.0);
	else
	{
		T r;
		T theta;
		if (abs<T>(u.x) > abs<T>(u.y))
		{
			r = u.x;
			theta = 0.25 * numbers::pi<T> * (u.y / u.x);
		}
		else
		{
			r = u.y;
			theta = 0.5 * numbers::pi<T> - 0.25 * numbers::pi<T> * (u.x / u.y);
		}

		p = r * vector<T, 2>(cos<T>(theta), sin<T>(theta));
	}

	return p;
}

template<typename T>
vector<T, 2> invertConcentricMapping(const vector<T, 2> p)
{
	T theta = hlsl::atan2(p.y, p.x);  // -pi -> pi
    T r = hlsl::sqrt(p.x * p.x + p.y * p.y);
	const T PiOver4 = T(0.25) * numbers::pi<T>;

	vector<T, 2> u;
	// TODO: should reduce branching somehow?
	if (hlsl::abs(theta) < PiOver4 || hlsl::abs(theta) > 3 * PiOver4)
	{
		r = ieee754::copySign(r, p.x);
        u.x = r;
        if (p.x < 0) {
            if (p.y < 0) {
                u.y = (numbers::pi<T> + theta) * r / PiOver4;
            } else {
                u.y = (theta - numbers::pi<T>) * r / PiOver4;
            }
        } else {
            u.y = (theta * r) / PiOver4;
        }
    }
	else
	{
		r = ieee754::copySign(r, p.y);
        u.y = r;
        if (p.y < 0) {
            u.x = -(T(0.5) * numbers::pi<T> + theta) * r / PiOver4;
        } else {
            u.x = (T(0.5) * numbers::pi<T> - theta) * r / PiOver4;
        }
    }

    return (u + hlsl::promote<vector<T, 2> >(1.0)) * T(0.5);
}

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
