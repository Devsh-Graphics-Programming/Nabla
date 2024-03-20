// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_MATH_EQUATIONS_CUBIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_EQUATIONS_CUBIC_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

// TODO: Later include from correct hlsl header
#ifndef nbl_hlsl_FLT_EPSILON
#define	nbl_hlsl_FLT_EPSILON 5.96046447754e-08
#endif

namespace nbl
{
namespace hlsl
{
namespace math
{
namespace equations
{
    
    template<typename float_t>
    struct Cubic
    {
        using float_t2 = vector<float_t, 2>;
        using float_t3 = vector<float_t, 3>;

        //TODO: use numeric_limits<float_t>::PI
        NBL_CONSTEXPR_STATIC_INLINE float_t PI_DOUBLE = 3.14159265358979323846;

        // TODO: a, b, c, d instead for better understanding
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
        float_t3 computeRoots() NBL_CONST_MEMBER_FUNC
        {
            int     i;
            float_t  sub;
            float_t  A, B, C;
            float_t  sq_A, p, q;
            float_t  cb_p, D;
            const float_t NaN = bit_cast<float_t>(numeric_limits<float_t>::quiet_NaN);
            float_t3 s = float_t3(NaN, NaN, NaN);
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
                    float_t u = cbrt(-q);
                    s[rootCount++] = 2 * u;
                    s[rootCount++] = -u;
                }
            }
            else if (D < 0) /* Casus irreducibilis: three real solutions */
            {
                float_t phi = 1.0 / 3 * acos(-q / sqrt(-cb_p));
                float_t t = 2 * sqrt(-p);

                s[rootCount++] = t * cos(phi);
                s[rootCount++] = -t * cos(phi + PI_DOUBLE / 3);
                s[rootCount++] = -t * cos(phi - PI_DOUBLE / 3);
            }
            else /* one real solution */
            {
                float_t sqrt_D = sqrt(D);
                float_t u = cbrt(sqrt_D - q);
                float_t v = -cbrt(sqrt_D + q);

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
}

#endif