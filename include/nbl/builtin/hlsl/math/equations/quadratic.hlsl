// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_MATH_EQUATIONS_QUADRATIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_EQUATIONS_QUADRATIC_INCLUDED_

// TODO: Later include from correct hlsl header
#ifndef nbl_hlsl_FLT_EPSILON
#define	nbl_hlsl_FLT_EPSILON 5.96046447754e-08
#endif

#define SHADER_CRASHING_ASSERT(expr) \
    do { \
        [branch] if (!(expr)) \
          vk::RawBufferStore<uint32_t>(0xdeadbeefBADC0FFbull,0x45u,4u); \
    } while(true)

namespace nbl
{
namespace hlsl
{
namespace math
{
namespace equations
{
    template<typename float_t>
    struct Quadratic
    {
        using float_t2 = vector<float_t, 2>;
        using float_t3 = vector<float_t, 3>;

        float_t a;
        float_t b;
        float_t c;

        static Quadratic construct(float_t a, float_t b, float_t c)
        {
            Quadratic ret = { a, b, c };
            return ret;
        }

        float_t evaluate(float_t t)
        {
            return t * (a * t + b) + c;
        }

        float_t2 computeRoots()
        {
            float_t2 ret;

            const float_t det = b * b - 4.0 * a * c;
            const float_t detSqrt = sqrt(det);
            const float_t rcp = 0.5 / a;
            const float_t bOver2A = b * rcp;

            float_t t0 = 0.0, t1 = 0.0;
            if (b >= 0)
            {
                ret[0] = -detSqrt * rcp - bOver2A;
                ret[1] = 2 * c / (-b - detSqrt);
            }
            else
            {
                ret[0] = 2 * c / (-b + detSqrt);
                ret[1] = +detSqrt * rcp - bOver2A;
            }

            return ret;
        }


    };
}
}
}
}

#endif