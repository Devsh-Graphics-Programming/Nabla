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
    };

    template<typename float_t>
    struct Quadratic
    {
        using float2_t = vector<float_t, 2>;
        using float3_t = vector<float_t, 3>;
        
        float2_t A;
        float2_t B;
        float2_t C;
        
        struct AnalyticArcLengthCalculator
        {
            using float2_t = vector<float_t, 2>;
        
            static AnalyticArcLengthCalculator construct(float_t lenA2, float_t AdotB, float_t a, float_t b, float_t  c, float_t b_over_4a)
            {
                AnalyticArcLengthCalculator ret = { lenA2, AdotB, a, b, c, b_over_4a };
                return ret;
            }
            
            // TODO: take quadratic A,B,C instead of control points
            static AnalyticArcLengthCalculator construct(Quadratic<float_t> quadratic)
            {
                AnalyticArcLengthCalculator ret;
                ret.lenA2 = dot(quadratic.A, quadratic.A);
                // sqrt(dot(A,A)) sqrt(dot(B,B)) cos(angle between A and B)
                ret.AdotB = dot(quadratic.A, quadratic.B);
        
                ret.a = 4.0 * ret.lenA2;
                ret.b = 4.0 * ret.AdotB;
                ret.c = dot(quadratic.B, quadratic.B);
        
                ret.b_over_4a = ret.AdotB / ret.a;
                return ret;
            }
            
            // These are precompured values to calculate arc length which eventually is an integral of sqrt(a*x^2+b*x+c)
            float_t lenA2;
            float_t AdotB;
        
            // differential arc-length is sqrt((dx/dt)^2+(dy/dt)^2), so for a Quad Bezier
            // sqrt((2A_x t + B_x)^2+(2A_y t + B_y)^2) == sqrt(4 (A_x^2+A_y^2) t + 4 (A_x B_x+A_y B_y) t + (B_x^2+B_y^2))
            float_t a;
            float_t b; // 2 sqrt(a) sqrt(c) cos(angle between A and B)
            float_t c;
        
            float_t b_over_4a;
            
            float_t calcArcLen(float_t t)
            {
                float_t lenTan = sqrt(t*(a*t+ b)+ c);
                float_t retval = 0.5f*t*lenTan;
                float_t sqrt_c = sqrt(c);
                
                // we skip this because when |a| -> we have += 0/0 * 0 here resulting in NaN
                // happens when P0, P1, P2 in a straight line and equally spaced
                if (lenA2 >= exp2(-23.f))
                {
                    // implementation might fall apart when beziers folds back on itself and `b = - 2 sqrt(a) sqrt(c)`
                    // https://www.wolframalpha.com/input?i=%28%28-2Sqrt%5Ba%5DSqrt%5Bc%5D%2B2ax%29+Sqrt%5Bc-2Sqrt%5Ba%5DSqrt%5Bc%5Dx%2Bax%5E2%5D+%2B+2+Sqrt%5Ba%5D+c%29%2F%284a%29
                    retval += b_over_4a*(lenTan - sqrt_c);

                    // sin2 multiplied by length of A and B
                    //  |A|^2 * |B|^2 * -(sin(A, B)^2)
                    float_t det_over_16 = AdotB * AdotB - lenA2 * c;
                    // because `b` is linearly dependent on `a` this will also ensure `b_over_4a` is not NaN, ergo `a` has a minimum value
                    if (abs(det_over_16)>=exp2(-23.f))
                    {
                        float_t sqrt_a = sqrt(a);
                        float_t subTerm0 = det_over_16*2.0f/(sqrt_a * sqrt_a * sqrt_a /*i know it can be faster*/);
                        float_t subTerm1 = log(b + 2.0f * sqrt_a * sqrt_c);
                        float_t subTerm2 = log(b + 2.0f * a * t + 2.0f * sqrt_a * lenTan);
                        
                        retval += subTerm0 * (subTerm1 - subTerm2);
                    }
                }
                
                //while(true)
                        //vk::RawBufferStore<uint32_t>(0xdeadbeefBADC0FFbull,0x45u,4u);

                return retval;
            }
            
            // keeping it for now
                        // TODO: remove
            float_t calcArcLenNumeric(float_t t)
            {
                
                double num = 0.0;
                const int steps = 100;
                    // from 0 to t
                const double rcp = double(t) / double(steps);
                for (int i = 0; i < steps; i++)
                {
                    double x = double(i) * rcp + 0.5 * rcp;
                    num += sqrt(x * (lenA2 * 4.0 * x + AdotB * 4.0) + c) * rcp;
                }
                
                return num;
                
            }
            
            float_t calcArcLenInverse(float_t arcLen, float_t accuracyThreshold, float_t hint, Quadratic<float_t> quadratic)
            {
                float_t xn = hint;

                if (arcLen <= accuracyThreshold)
                    return arcLen;

                    // TODO: implement halley method
                const uint32_t iterationThreshold = 32;
                for(uint32_t n = 0; n < iterationThreshold; n++)
                {
                    float_t arcLenDiffAtParamGuess = arcLen - calcArcLen(xn);

                    if (abs(arcLenDiffAtParamGuess) < accuracyThreshold)
                        return xn;

                    float_t differentialAtGuess = length(2.0*quadratic.A * xn + quadratic.B);
                        // x_n+1 = x_n - f(x_n)/f'(x_n)
                    xn -= (calcArcLen(xn) - arcLen) / differentialAtGuess;
                }
                
                //while(true)
                //    vk::RawBufferStore<uint32_t>(0xdeadbeefBADC0FFbull,0x45u,4u);

                return xn;
            }

        };
        
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
            return t * (A * t + B) + C;
        }
        
        struct DefaultClipper
        {
            static DefaultClipper construct()
            {
                DefaultClipper ret;
                return ret;
            }
        
            inline float2_t operator()(const float_t t)
            {
                return clamp(t, 0.0, 1.0);
            }
        };
        
        template<typename Clipper>
        float2 ud(float2_t pos, float_t thickness, Clipper clipper)
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
            
            const float_t MAX_DISTANCE_SQUARED = (thickness+500.0f)*(thickness+500.0f);

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
                float2_t t = uv.x + uv.y - kx;            
                
                float2_t tOrigQos = CsubPos + (B + A*t.x)*t.x;
                res = float2_t(dot(tOrigQos, tOrigQos), t.x);
                if(res.x > MAX_DISTANCE_SQUARED)
                {
                    res.x = sqrt(res.x);
                    return res;
                }
                
                t = clipper(t.x);
                
                // 1 root
                float2_t qos = CsubPos + (B + A*t.x)*t.x;
                res = float2_t(dot(qos,qos),t.x);
                
                if(t.x != t.y)
                {
                    qos = CsubPos + (B + A*t.y)*t.y;
                    float dis = dot(qos,qos);
                    if(dis < res.x) res = float2_t(dis, t.y);
                }
                
                res.x = sqrt(res.x);
            }
            else
            {
                float_t z = sqrt(-p);
                float_t v = acos( q/(p*z*2.0) ) / 3.0;
                float_t m = cos(v);
                float_t n = sin(v)*1.732050808;
                //float3_t t = float3_t(m + m, -n - m, n - m) * z - kx;
                
                float2_t t[3];
                t[0] = (m + m) * z - kx;
                t[1] = (-n - m) * z - kx;
                t[2] = (n - m) * z - kx;
                
                // 3 roots
                float_t dis;
                res.x = float_t(0xFFFFFFFFFFFFFFFF);
                for(uint32_t i = 0u; i < 3u; i++)
                {
                    float_t tOrigQos = CsubPos + (B + A*t[i].x)*t[i].x;
                    float_t tOrigDis = dot(tOrigQos, tOrigQos);
                    if(tOrigDis > MAX_DISTANCE_SQUARED)
                    {
                        res.x = tOrigDis;
                        continue;
                    }
                
                    t[i] = clipper(t[i].x);
                    
                    float2_t qos = CsubPos + (B + A*t[i].x)*t[i].x;
                    dis = dot(qos, qos);
                    if( dis<res.x ) res = float2_t(dis,t[i].x );
                    
                    if(t[i].x != t[i].y)
                    {
                        qos = CsubPos + (B + A*t[i].y)*t[i].y;
                        dis = dot(qos, qos);
                        if(dis < res.x) res = float2_t(dis, t[i].y); 
                    }
                }
                

                res.x = sqrt( res.x );
            }
            
            return res;
        }
       
        template<typename Clipper/* = DefaultClipper*/>
        float_t signedDistance(float2_t pos, float_t thickness, Clipper clipper/* = DefaultClipper::construct()*/)
        {
            return abs(ud<Clipper>(pos, thickness, clipper)).x - thickness;
        }
        
        // TODO: To be deleted probably
        float_t signedDistance(float2_t pos, float_t thickness)
        {
            return signedDistance<DefaultClipper>(pos, thickness, DefaultClipper::construct());
        }
    };
}
}
}

#endif