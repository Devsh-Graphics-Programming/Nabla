// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_EQUATIONS_QUADRATIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_EQUATIONS_QUADRATIC_INCLUDED_

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

#define SHADER_CRASHING_ASSERT(expr) \
    do { \
        [branch] if (!(expr)) \
          vk::RawBufferStore<uint32_t>(0xdeadbeefBADC0FFbull,0x45u,4u); \
    } while(true)

namespace nbl
{
namespace hlsl
{
namespace equations
{
    template<typename float_t>
    struct Quadratic
    {
        using float2_t = vector<float_t, 2>;
        using float3_t = vector<float_t, 3>;

        float_t A;
        float_t B;
        float_t C;

        static Quadratic construct(float_t A, float_t B, float_t C)
        {
            Quadratic ret = { A, B, C };
            return ret;
        }

        float_t evaluate(float_t t)
        {
            return t * (A * t + B) + C;
        }

        float2_t computeRoots()
        {
            float2_t ret;

            const float_t det = B * B - 4.0 * A *C;
            const float_t detSqrt = sqrt(det);
            const float_t rcp = 0.5 / A;
            const float_t bOver2A = B * rcp;

            if (isinf(rcp)) return float2_t(-C / B, NBL_NOT_A_NUMBER());

            float_t t0 = 0.0, t1 = 0.0;
            if (B >= 0)
            {
                ret[0] = -detSqrt * rcp - bOver2A;
                ret[1] = 2 * C / (-B - detSqrt);
            }
            else
            {
                ret[0] = 2 * C / (-B + detSqrt);
                ret[1] = +detSqrt * rcp - bOver2A;
            }

            return ret;
        }

        // SolveQuadratic:
        // det = b*b-4.f*a*c;
        // rcp = 0.5f/a;
        // detSqrt = sqrt(det)*rcp;
        // tmp = b*rcp;
        // res = float2(-detSqrt,detSqrt)-tmp;
        //
        // Collapsed version:
        // detrcp2 = det * rcp * rcp
        // brcp = b * rcp
        //
        // (computeRoots())
        // detSqrt = sqrt(detrcp2)
        // res = float2(-detSqrt,detSqrt)-bRcp;
        struct PrecomputedRootFinder 
        {
            float_t detRcp2;
            float_t brcp;

            float2_t computeRoots() {
                float_t detSqrt = sqrt(detRcp2);
                float2_t roots = float2_t(-detSqrt,detSqrt)-brcp;
                // assert(roots.x == roots.y);
                // assert(!isnan(roots.x));
                // if a = 0, brcp is inf
                // then we return detRcp2, which was set below to -c / b
                // that works as a solution to the linear equation bx+c=0
                return isinf(brcp) ? detRcp2 : roots;
            }

            static PrecomputedRootFinder construct(float_t detRcp2, float_t brcp)
            {
                PrecomputedRootFinder result;
                result.detRcp2 = detRcp2;
                result.brcp = brcp;
                return result;
            }

            static PrecomputedRootFinder construct(nbl::hlsl::equations::Quadratic<float_t> quadratic)
            {
                float_t a = quadratic.A;
                float_t b = quadratic.B;
                float_t c = quadratic.C;
                
                float_t det = b*b-4.f*a*c;
                float_t rcp = 0.5f/a;
                
                PrecomputedRootFinder result;
                result.brcp = b * rcp;
                result.detRcp2 = isinf(result.brcp) ? -c / b : det * rcp * rcp;
                return result;
            }
        };


    };
}
}
}

#endif