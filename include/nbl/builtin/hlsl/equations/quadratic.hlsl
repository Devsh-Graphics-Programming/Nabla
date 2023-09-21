// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_EQUATIONS_QUADRATIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_EQUATIONS_QUADRATIC_INCLUDED_

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
                return roots;
            }

            static PrecomputedRootFinder construct(float_t detRcp2, float_t brcp)
            {
                PrecomputedRootFinder result;
                result.detRcp2 = detRcp2;
                result.brcp = brcp;
                return result;
            }

            static PrecomputedRootFinder construct(nbl::hlsl::equations::Quadratic<float_t> quadratic, float_t minorAxisNdc)
            {
                float_t a = quadratic.A;
                float_t b = quadratic.B;
                float_t c = quadratic.C - minorAxisNdc;
                
                float_t det = b*b-4.f*a*c;
                float_t rcp = 0.5f/a;
                
                PrecomputedRootFinder result;
                result.detRcp2 = det * rcp * rcp;
                result.brcp = b * rcp;
                return result;
            }
        };


    };
}
}
}

#endif