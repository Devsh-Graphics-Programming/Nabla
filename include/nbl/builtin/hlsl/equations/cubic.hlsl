// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_EQUATIONS_CUBIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_EQUATIONS_CUBIC_INCLUDED_

// TODO: Later include from correct hlsl header
#ifndef nbl_hlsl_FLT_EPSILON
#define	nbl_hlsl_FLT_EPSILON 5.96046447754e-08
#endif

#ifndef NBL_NOT_A_NUMBER
#ifdef __cplusplus
#define NBL_NOT_A_NUMBER() nbl::core::nan<float_t>()
#else
// https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-float-rules#honored-ieee-754-rules
#define NBL_NOT_A_NUMBER() 0.0/0.0
#endif
#endif //NBL_NOT_A_NUMBER

namespace nbl
{
namespace hlsl
{
namespace equations
{
    //TODO: use numeric_limits<float_t>::PI
	NBL_CONSTEXPR double PI_DOUBLE = 3.14159265358979323846;
    
    template<typename float_t>
    struct Cubic
    {
        using float2_t = vector<float_t, 2>;
        using float3_t = vector<float_t, 3>;

        float_t c[4];

        static Cubic construct(float_t a, float_t b, float_t c, float_t d)
        {
            Cubic ret;
            ret.c[0] = d;
            ret.c[1] = c;
            ret.c[2] = b;
            ret.c[3] = a;
            return ret;
        }

        // Originally from: https://github.com/erich666/GraphicsGems/blob/master/gems/Roots3And4.c
        float3_t computeRoots() {
            int     i;
            double  sub;
            double  A, B, C;
            double  sq_A, p, q;
            double  cb_p, D;
            float3_t s = float3_t(NBL_NOT_A_NUMBER(), NBL_NOT_A_NUMBER(), NBL_NOT_A_NUMBER());
            uint32_t rootCount = 0;

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

                s[rootCount++] = u + v;
            }

            /* resubstitute */

            sub = 1.0 / 3 * A;

            for (i = 0; i < rootCount; ++i)
                s[i] -= sub;

            return s;
        }


    };
}
}
}

#endif