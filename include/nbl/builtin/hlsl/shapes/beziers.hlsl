// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SHAPES_BEZIERS_INCLUDED_
#define _NBL_BUILTIN_HLSL_SHAPES_BEZIERS_INCLUDED_

// TODO: Later include from correct hlsl header
#ifndef nbl_hlsl_FLT_EPSILON
#define	nbl_hlsl_FLT_EPSILON 5.96046447754e-08
#endif

namespace nbl
{
namespace hlsl
{
namespace shapes
{
    // Modified from http://tog.acm.org/resources/GraphicsGems/gems/Roots3And4.c
    // GH link https://github.com/erich666/GraphicsGems/blob/master/gems/Roots3And4.c
    // Credits to Doublefresh for hinting there
    // returns the roots, and number of filled in real values under numRealValues
    float2 SolveQuadratic(float3 C, out int numRealValues)
    {
        // bhaskara: x = (-b ± √(b² – 4ac)) / (2a)
        float b = C.y / (2 * C.z);
        float q = C.x / C.z;
        float delta = b * b - q;

        if (delta == 0.0) // Δ = 0
        {
            numRealValues = 1;
            return float2(-b, 0.0);
        }
        if (delta < 0) // Δ < 0 (no real values)
        {
            numRealValues = 0;
            return 0.0;
        }

        // Δ > 0 (two distinct real values)
        float sqrtD = sqrt(delta);
        numRealValues = 2;
        return float2(sqrtD - b, sqrtD + b);
    }
    
    template<typename float_t>
    struct QuadraticBezier
    {
        using float2_t = vector<float_t, 2>;
        using float3_t = vector<float_t, 3>;

        float2_t P0;
        float2_t P1;
        float2_t P2;

        static QuadraticBezier construct(float2_t P0, float2_t P1, float2_t P2)
        {
            QuadraticBezier ret = { P0, P1, P2 };
            return ret;
        }

        float2_t evaluate(float_t t)
        {
            float2_t position = 
                P0 * (1.0 - t) * (1.0 - t) 
                 + 2.0 * P1 * (1.0 - t) * t
                 +       P2 * t         * t;

            return position;
        }

        // TODO [Lucas]: move this function to Quadratic struct
        // https://pomax.github.io/bezierinfo/#yforx
        float_t tForMajorCoordinate(const int major, float_t x) 
        { 
            float_t a = P0[major] - x;
            float_t b = P1[major] - x;
            float_t c = P2[major] - x;
            int rootCount;
            float2_t roots = SolveQuadratic(float3_t(a, b, c), rootCount);
            // assert(rootCount == 1);
            return roots.x;
        }
    };

    template<typename float_t>
    struct Quadratic
    {
        using float2_t = vector<float_t, 2>;
        using float3_t = vector<float_t, 3>;
        
        // These are precompured values to calculate arc length which eventually is an integral of sqrt(a*x^2+b*x+c)
        struct ArcLengthPrecomputedValues
        {
            float_t lenA2;
            float_t AdotB;

            float_t a;
            float_t b;
            float_t c;

            float_t b_over_4a;
        };
        
        float2_t A;
        float2_t B;
        float2_t C;
        
        static Quadratic construct(float2_t A, float2_t B, float2_t C)
        {            
            Quadratic ret = { A, B, C };
            return ret;
        }
        
        static Quadratic constructFromBezier(QuadraticBezier<float_t> curve)
        {
            const float2_t A = curve.P0 - 2.0 * curve.P1 + curve.P2;
            const float2_t B = 2.0f * (curve.P1 - curve.P0);
            const float2_t C = curve.P0;
            
            Quadratic ret = { A, B, C };
            return ret;
        }
        
        float2_t evaluate(float_t t)
        {
            return A * (t * t) + B * t + C;
        }
        
        float_t calcArcLen(float_t t, ArcLengthPrecomputedValues preCompValues)
        {
            float_t lenTan = sqrt(t*(preCompValues.a*t+ preCompValues.b)+ preCompValues.c);
            float_t retval = 0.5f*t*lenTan;
            float_t sqrtc = sqrt(preCompValues.c);
            
            // we skip this because when |a| -> we have += 0/0 * 0 here resulting in NaN
            if (preCompValues.lenA2>=exp2(-23.f))
                retval += preCompValues.b_over_4a*(lenTan-sqrtc);

            // sin2 multiplied by length of A and B
            float det_over_16 = preCompValues.AdotB* preCompValues.AdotB- preCompValues.lenA2* preCompValues.c;
            // because `b` is linearly dependent on `a` this will also ensure `b_over_4a` is not NaN, ergo `a` has a minimum value
            if (det_over_16>=exp2(-23.f))
            {
            // TODO: finish by @Przemog
                //retval += det_over_16*...;
            }

            return retval;
        }
        
        // TODO: make clipper as 3rd parameter
        float2 ud(float2_t pos)
        {            
            // p(t)    = (1-t)^2*A + 2(1-t)t*B + t^2*C
            // p'(t)   = 2*t*(A-2*B+C) + 2*(B-A)
            // p'(0)   = 2(B-A)
            // p'(1)   = 2(C-B)
            // p'(1/2) = 2(C-A)
            
            float2_t Bdiv2 = B/2.0f;
            float2_t CsubPos = C - pos;

            // Reducing Quartic to Cubic Solution
            float_t kk = 1.0 / dot(A,A);
            float_t kx = kk * dot(Bdiv2,A);
            float_t ky = kk * (2.0*dot(Bdiv2,Bdiv2)+dot(CsubPos,A)) / 3.0;
            float_t kz = kk * dot(CsubPos,Bdiv2);      

            float2_t res;

            // Cardano's Solution to resolvent cubic of the form: y^3 + 3py + q = 0
            // where it was initially of the form x^3 + ax^2 + bx + c = 0 and x was replaced by y - a/3
            // so a/3 needs to be subtracted from the solution to the first form to get the actual solution
            float_t p = ky - kx*kx;
            float_t p3 = p*p*p;
            float_t q = kx*(2.0*kx*kx - 3.0*ky) + kz;
            float_t h = q*q + 4.0*p3;

            if(h >= 0.0) 
            { 
                h = sqrt(h);
                float2_t x = (float2_t(h, -h) - q) / 2.0;

                // Solving Catastrophic Cancellation when h and q are close (when p is near 0)
                if(abs(abs(h/q) - 1.0) < 0.0001)
                {
                   // Approximation of x where h and q are close with no carastrophic cancellation
                   // Derivation (for curious minds) -> h=√(q²+4p³)=q·√(1+4p³/q²)=q·√(1+w)
                      // Now let's do linear taylor expansion of √(1+x) to approximate √(1+w)
                      // Taylor expansion at 0 -> f(x)=f(0)+f'(0)*(x) = 1+½x 
                      // So √(1+w) can be approximated by 1+½w
                      // Simplifying later and the fact that w=4p³/q will result in the following.
                   x = float2_t(p3/q, -q - p3/q);
                }

                float2_t uv = sign(x)*pow(abs(x), float2_t(1.0/3.0,1.0/3.0));
                float_t t = uv.x + uv.y - kx;
                t = clamp( t, 0.0, 1.0 );

                // 1 root
                float2_t qos = CsubPos + (B + A*t)*t;
                res = float2_t( length(qos),t);
            }
            else
            {
                float_t z = sqrt(-p);
                float_t v = acos( q/(p*z*2.0) ) / 3.0;
                float_t m = cos(v);
                float_t n = sin(v)*1.732050808;
                float3_t t = float3_t(m + m, -n - m, n - m) * z - kx;
                t = clamp( t, 0.0, 1.0 );

                // 3 roots
                float2_t qos = CsubPos + (B + A*t.x)*t.x;
                float_t dis = dot(qos,qos);
                
                res = float2_t(dis,t.x);

                qos = CsubPos + (B + A*t.y)*t.y;
                dis = dot(qos,qos);
                if( dis<res.x ) res = float2_t(dis,t.y );

                qos = CsubPos + (B + A*t.z)*t.z;
                dis = dot(qos,qos);
                if( dis<res.x ) res = float2_t(dis,t.z );

                res.x = sqrt( res.x );
            }
            
            return res;
        }
        
        float_t signedDistance(float2_t pos, float_t thickness)
        {
            return abs(ud(pos)).x - thickness;
        }
        
        float_t calcArcLenInverse(float_t arcLen, float_t accuracyThreshold, float_t hint, ArcLengthPrecomputedValues preCompValues)
        {
            float_t xn = hint;

            if (arcLen <= accuracyThreshold)
                return arcLen;

                // TODO: implement halley method
            const int iterationThreshold = 32;
            for(int n = 0; n < iterationThreshold; n++)
            {
                float_t arcLenDiffAtParamGuess = arcLen - calcArcLen(xn, preCompValues);

                if (abs(arcLenDiffAtParamGuess) < accuracyThreshold)
                    return xn;

                float_t differentialAtGuess = length(2.0*A * xn + B);
                    // x_n+1 = x_n - f(x_n)/f'(x_n)
                xn -= (calcArcLen(xn, preCompValues) - arcLen) / differentialAtGuess;
            }

            return xn;
        }
    };
}
}
}

#endif