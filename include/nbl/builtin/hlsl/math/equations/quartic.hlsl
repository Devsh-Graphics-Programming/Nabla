// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_MATH_EQUATIONS_QUARTIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_EQUATIONS_QUARTIC_INCLUDED_

#include <nbl/builtin/hlsl/math/equations/quadratic.hlsl>
#include <nbl/builtin/hlsl/math/equations/cubic.hlsl>

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
    struct Quartic
    {
        using float_t2 = vector<float_t, 2>;
        using float_t3 = vector<float_t, 3>;
        using float_t4 = vector<float_t, 4>;

        // form: ax^4 + bx^3 + cx^2 + dx + e
        float_t a;
        float_t b;
        float_t c; 
        float_t d;
        float_t e;

        static Quartic construct(float_t a, float_t b, float_t c, float_t d, float_t e)
        {
            Quartic ret;
            ret.a = a;
            ret.b = b;
            ret.c = c;
            ret.d = d;
            ret.e = e;
            return ret;
        }

        // Originally from: https://github.com/erich666/GraphicsGems/blob/master/gems/Roots3And4.c
        float_t4 computeRoots() NBL_CONST_MEMBER_FUNC
        {
            float_t  coeffs[4];
            float_t  z, u, v, sub;
            float_t  A, B, C, D;
            float_t  sq_A, p, q, r;
            int     i;
            const float_t NaN = bit_cast<float_t>(numeric_limits<float_t>::quiet_NaN);
            float_t4 s = float_t4(NaN, NaN, NaN, NaN);
            uint32_t rootCount = 0;

            /* normal form: x^4 + Ax^3 + Bx^2 + Cx + D = 0 */

            A = b / a;
            B = c / a;
            C = d / a;
            D = e / a;

            /*  substitute x = y - A/4 to eliminate cubic term:
            x^4 + px^2 + qx + r = 0 */

            sq_A = A * A;
            p = -3.0 / 8 * sq_A + B;
            q = 1.0 / 8 * sq_A * A - 1.0 / 2 * A * B + C;
            r = -3.0 / 256 * sq_A * sq_A + 1.0 / 16 * sq_A * B - 1.0 / 4 * A * C + D;

            if (r == 0.0)
            {
                /* no absolute term: y(y^3 + py + q) = 0 */

                float_t3 cubic = Cubic<float_t>::construct(1, 0, p, q).computeRoots();
                s[rootCount ++] = cubic[0];
                s[rootCount ++] = cubic[1];
                s[rootCount ++] = cubic[2];
            }
            else
            {
                /* solve the resolvent cubic ... */
                float_t3 cubic = Cubic<float_t>::construct(1, -1.0 / 2 * p, -r, 1.0 / 2 * r * p - 1.0 / 8 * q * q).computeRoots();
                /* ... and take the one real solution ... */
                for (uint32_t i = 0; i < 3; i ++)
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

                float_t2 quadric1 = Quadratic<float_t>::construct(1, q < 0 ? -v : v, z - u).computeRoots();
                float_t2 quadric2 = Quadratic<float_t>::construct(1, q < 0 ? v : -v, z + u).computeRoots();

                for (uint32_t i = 0; i < 2; i ++)
                    if (!isinf(quadric1[i]) && !isnan(quadric1[i]))
                        s[rootCount ++] = quadric1[i];

                for (uint32_t i = 0; i < 2; i ++)
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
}

#endif