// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_EQUATIONS_QUARTIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_EQUATIONS_QUARTIC_INCLUDED_

#include <nbl/builtin/hlsl/equations/cubic.hlsl>

// TODO: Later include from correct hlsl header
#ifndef nbl_hlsl_FLT_EPSILON
#define	nbl_hlsl_FLT_EPSILON 5.96046447754e-08
#endif

namespace nbl
{
namespace hlsl
{
namespace equations
{
    
    // TODO: Figure out why the other quadratic equation wont work here
    template<typename float_t>
	vector<float_t, 2> SolveQuadratic(float_t c[3])
	{
		float_t p, q, D;

		/* normal form: x^2 + px + q = 0 */

		p = c[1] / (2 * c[2]);
		q = c[0] / c[2];

		D = p * p - q;

		if (D == 0.0)
			return vector<float_t, 2>(-p, NBL_NOT_A_NUMBER());
		else if (D < 0)
			return vector<float_t, 2>(NBL_NOT_A_NUMBER(), NBL_NOT_A_NUMBER());
		else /* if (D > 0) */
		{
			float_t sqrt_D = sqrt(D);
			return vector<float_t, 2>(sqrt_D - p, -sqrt_D - p);
		}
	}
    

    template<typename float_t>
    struct Quartic
    {
        using float2_t = vector<float_t, 2>;
        using float3_t = vector<float_t, 3>;
        using float4_t = vector<float_t, 4>;

        float_t c[5];

        static Quartic construct(float_t A, float_t B, float_t C, float_t D, float_t E)
        {
            Quartic ret;
            ret.c[0] = A;
            ret.c[1] = B;
            ret.c[2] = C;
            ret.c[3] = D;
            ret.c[4] = E;
            return ret;
        }

    

    float4_t computeRoots() {
		double  coeffs[4];
		double  z, u, v, sub;
		double  A, B, C, D;
		double  sq_A, p, q, r;
		int     i;
        float4_t s = float4_t(NBL_NOT_A_NUMBER(), NBL_NOT_A_NUMBER(), NBL_NOT_A_NUMBER(), NBL_NOT_A_NUMBER());
        uint rootCount = 0;

		/* normal form: x^4 + Ax^3 + Bx^2 + Cx + D = 0 */

		A = c[3] / c[4];
		B = c[2] / c[4];
		C = c[1] / c[4];
		D = c[0] / c[4];

		/*  substitute x = y - A/4 to eliminate cubic term:
		x^4 + px^2 + qx + r = 0 */

		sq_A = A * A;
		p = -3.0 / 8 * sq_A + B;
		q = 1.0 / 8 * sq_A * A - 1.0 / 2 * A * B + C;
		r = -3.0 / 256 * sq_A * sq_A + 1.0 / 16 * sq_A * B - 1.0 / 4 * A * C + D;

		if (r == 0.0)
		{
			/* no absolute term: y(y^3 + py + q) = 0 */

			float3_t cubic = nbl::hlsl::equations::Cubic<float_t>::construct(q, p, 0, 1).computeRoots();
            s[rootCount ++] = cubic[0];
            s[rootCount ++] = cubic[1];
            s[rootCount ++] = cubic[2];
		}
		else
		{
			/* solve the resolvent cubic ... */
			float3_t cubic = nbl::hlsl::equations::Cubic<float_t>::construct( 1.0 / 2 * r * p - 1.0 / 8 * q * q, -r, -1.0 / 2 * p, 1).computeRoots();
			/* ... and take the one real solution ... */
            for (uint i = 0; i < 3; i ++)
				if (!isnan(cubic[i]))
				{
					z = cubic[i];
					break;
				}

			/* ... to build two quadratic equations */

			u = z * z - r;
			v = 2 * z - p;

			if (u == 0.0)
				u = 0;
			else if (u > 0)
				u = sqrt(u);
			else
				return s; // (empty)

			if (v == 0.0)
				v = 0;
			else if (v > 0)
				v = sqrt(v);
			else
				return s; // (empty)

			coeffs[0] = z - u;
			coeffs[1] = q < 0 ? -v : v;
			coeffs[2] = 1;

			float2_t quadric1 = SolveQuadratic(coeffs);

			coeffs[0] = z + u;
			coeffs[1] = q < 0 ? v : -v;
			coeffs[2] = 1;

			float2_t quadric2 = SolveQuadratic(coeffs);
            
            for (uint i = 0; i < 2; i ++)
                if (!isinf(quadric1[i]) && !isnan(quadric1[i]))
                    s[rootCount ++] = quadric1[i];

            for (uint i = 0; i < 2; i ++)
                if (!isinf(quadric2[i]) && !isnan(quadric2[i]))
                    s[rootCount ++] = quadric2[i];
		}

		/* resubstitute */

		sub = 1.0 / 4 * A;

		for (i = 0; i < rootCount; ++i)
			s[i] -= sub;

		return s;
        }


    };
}
}
}

#endif