// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_EQUATIONS_HATCH_INCLUDED_
#define _NBL_BUILTIN_HLSL_EQUATIONS_HATCH_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

namespace nbl
{
namespace hlsl
{
namespace equations
{
namespace hatch
{
	NBL_CONSTEXPR double PI_DOUBLE = 3.14159265358979323846;
	// TODO: Find way to do this that is HLSL & C++ compatible
#ifdef __cplusplus
#define NOT_A_NUMBER() nbl::core::nan<float_t>()
#else
// https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-float-rules#honored-ieee-754-rules
#define NOT_A_NUMBER() 0.0/0.0
#endif

	// From https://github.com/erich666/GraphicsGems/blob/master/gems/Roots3And4.c
	// These are not fully safe on NaNs or when handling precision, as pointed out by devsh in the Discord
    template<typename float_t>
	vector<float_t, 2> SolveQuadratic(float_t c[3])
	{
		float_t p, q, D;

		/* normal form: x^2 + px + q = 0 */

		p = c[1] / (2 * c[2]);
		q = c[0] / c[2];

		D = p * p - q;

		if (D == 0.0)
			return vector<float_t, 2>(-p, NOT_A_NUMBER());
		else if (D < 0)
			return vector<float_t, 2>(NOT_A_NUMBER(), NOT_A_NUMBER());
		else /* if (D > 0) */
		{
			float_t sqrt_D = sqrt(D);
			return vector<float_t, 2>(sqrt_D - p, -sqrt_D - p);
		}
	}
    
    template<typename float_t>
	vector<float_t, 3> SolveCubic(float_t c[4])
	{
		int     i;
		double  sub;
		double  A, B, C;
		double  sq_A, p, q;
		double  cb_p, D;
        vector<float_t, 3> s = vector<float_t, 3>(NOT_A_NUMBER(), NOT_A_NUMBER(), NOT_A_NUMBER());
        uint rootCount = 0;

		/* normal form: x^3 + Ax^2 + Bx + C = 0 */

		A = c[2] / c[3];
		B = c[1] / c[3];
		C = c[0] / c[3];

		/*  substitute x = y - A/3 to eliminate quadric term:
		x^3 +px + q = 0 */

		sq_A = A * A;
		p = 1.0 / 3 * (-1.0 / 3 * sq_A + B);
		q = 1.0 / 2 * (2.0 / 27 * A * sq_A - 1.0 / 3 * A * B + C);

		/* use Cardano's formula */

		cb_p = p * p * p;
		D = q * q + cb_p;

		if (D == 0.0)
		{
			if (q == 0.0) /* one triple solution */
			{
				s[rootCount++] = 0.0;
			}
			else /* one single and one double solution */
			{
				double u = cbrt(-q);
				s[rootCount++] = 2 * u;
				s[rootCount++] = -u;
			}
		}
		else if (D < 0) /* Casus irreducibilis: three real solutions */
		{
			double phi = 1.0 / 3 * acos(-q / sqrt(-cb_p));
			double t = 2 * sqrt(-p);

            s[rootCount++] = t * cos(phi);
            s[rootCount++] = -t * cos(phi + PI_DOUBLE / 3);
            s[rootCount++] = -t * cos(phi - PI_DOUBLE / 3);
		}
		else /* one real solution */
		{
			double sqrt_D = sqrt(D);
			double u = cbrt(sqrt_D - q);
			double v = -cbrt(sqrt_D + q);

            s[rootCount++] = u;
            s[rootCount++] = v;
		}

		/* resubstitute */

		sub = 1.0 / 3 * A;

		for (i = 0; i < rootCount; ++i)
			s[i] -= sub;

		return s;
	}

    
    template<typename float_t>
	vector<float_t, 4> SolveQuartic(float_t c[5])
    {
		double  coeffs[4];
		double  z, u, v, sub;
		double  A, B, C, D;
		double  sq_A, p, q, r;
		int     i;
        vector<float_t, 4> s = vector<float_t, 4>(NOT_A_NUMBER(), NOT_A_NUMBER(), NOT_A_NUMBER(), NOT_A_NUMBER());
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

			coeffs[0] = q;
			coeffs[1] = p;
			coeffs[2] = 0;
			coeffs[3] = 1;

			vector<float_t, 3> cubic = SolveCubic(coeffs);
            s[rootCount ++] = cubic[0];
            s[rootCount ++] = cubic[1];
            s[rootCount ++] = cubic[2];
		}
		else
		{
			/* solve the resolvent cubic ... */

			coeffs[0] = 1.0 / 2 * r * p - 1.0 / 8 * q * q;
			coeffs[1] = -r;
			coeffs[2] = -1.0 / 2 * p;
			coeffs[3] = 1;

			/* ... and take the one real solution ... */

			vector<float_t, 3> cubic = SolveCubic(coeffs);
            for (uint i = 0; i < 3; i ++)
                if (!isnan(cubic[i]) && !isnan(cubic[i]))
			        z = cubic[i];

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

			vector<float_t, 2> quadric1 = SolveQuadratic(coeffs);

			coeffs[0] = z + u;
			coeffs[1] = q < 0 ? v : -v;
			coeffs[2] = 1;

			vector<float_t, 2> quadric2 = SolveQuadratic(coeffs);
            
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


} // hatch
} // equations
} // hlsl
} // nbl

#endif
