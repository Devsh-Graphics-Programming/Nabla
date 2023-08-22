// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

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
        using vec2 = vector<float_t, 2>;
        using vec3 = vector<float_t, 3>;

        struct ArcLengthPrecomputedValues
        {
            float_t lenA2;
            float_t AdotB;

            float_t a;
            float_t b;
            float_t c;

            float_t b_over_4a;
        };

        vec2 P0;
        vec2 P1;
        vec2 P2;
        ArcLengthPrecomputedValues preCompValues;

        static QuadraticBezier construct(vec2 P0, vec2 P1, vec2 P2)
        {
            QuadraticBezier ret;
            ret.P0 = P0;
            ret.P1 = P1;
            ret.P2 = P2;

            return ret;
        }

        static QuadraticBezier construct(vec2 P0, vec2 P1, vec2 P2, float_t lenA2, float_t AdotB, float_t a, float_t b, float_t c, float_t b_over_4a)
        {
            QuadraticBezier ret = {P0, P1, P2, lenA2, AdotB, a, b, c, b_over_4a};
            return ret;
        }

        vector<float_t, 2> evaluate(float_t t)
        {
            float2 position = 
                P0 * (1.0 - t) * (1.0 - t) 
                 + 2.0 * P1 * (1.0 - t) * t
                 +       P2 * t         * t;

            return position;
        }

        // https://pomax.github.io/bezierinfo/#yforx
        float_t tForMajorCoordinate(const int major, float_t x) 
        { 
            float_t a = P0[major] - x;
            float_t b = P1[major] - x;
            float_t c = P2[major] - x;
            int rootCount;
            vec2 roots = SolveQuadratic(vec3(a, b, c), rootCount);
            // assert(rootCount == 1);
            return roots.x;
        }

        float_t calcArcLen(float_t t)
        {
            float_t lenTan = sqrt(t*(preCompValues.a*t+ preCompValues.b)+ preCompValues.c);
            float_t retval = 0.5f*t*lenTan;
            // we skip this because when |a| -> we have += 0/0 * 0 here resulting in NaN
            if (preCompValues.lenA2>=exp2(-23.f))
                retval += preCompValues.b_over_4a*(lenTan-sqrt(preCompValues.c));

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

        float_t calcArcLenInverse(float_t arcLen, float_t accuracyThreshold, float_t hint)
        {
            float_t xn = hint;

            if (arcLen <= accuracyThreshold)
                return arcLen;

            vec2 B = 2.0*(P1 - P0);
            vec2 twoA = 2.0*(P2 - P1) - B;

            const int iterationThreshold = 32;
            for(int n = 0; n < iterationThreshold; n++)
            {
                float_t arcLenDiffAtParamGuess = arcLen - calcArcLen(xn);

                if (abs(arcLenDiffAtParamGuess) < accuracyThreshold)
                    return xn;

                float_t differentialAtGuess = length(twoA * xn + B);
                    // x_n+1 = x_n - f(x_n)/f'(x_n)
                xn -= (calcArcLen(xn) - arcLen) / differentialAtGuess;
            }

            return xn;
        }
    };

    template<typename float_t>
    struct QuadraticBezierOutline
    {
        using vec2 = vector<float_t, 2>;
        using vec3 = vector<float_t, 3>;

        QuadraticBezier<float_t> bezier;
        float_t thickness;

        static QuadraticBezierOutline construct(vec2 P0, vec2 P1, vec2 P2, float_t thickness)
        {
            QuadraticBezier<float_t> bezier = QuadraticBezier<float_t>::construct(P0, P1, P2);
            QuadraticBezierOutline<float_t> ret = { bezier, thickness };
            return ret;
        }

        // original from https://www.shadertoy.com/view/lsyfWc
        float2 ud(vec2 pos)
        {
            const vec2 P0 = bezier.P0;
            const vec2 P1 = bezier.P1;
            const vec2 P2 = bezier.P2;
                    
            // p(t)    = (1-t)^2*A + 2(1-t)t*B + t^2*C
            // p'(t)   = 2*t*(A-2*B+C) + 2*(B-A)
            // p'(0)   = 2(B-A)
            // p'(1)   = 2(C-B)
            // p'(1/2) = 2(C-A)
                    
            vec2 A = P1 - P0;
            vec2 B = P0 - 2.0*P1 + P2;
            vec2 C = P0 - pos;

            // Reducing Quartic to Cubic Solution
            float_t kk = 1.0 / dot(B,B);
            float_t kx = kk * dot(A,B);
            float_t ky = kk * (2.0*dot(A,A)+dot(C,B)) / 3.0;
            float_t kz = kk * dot(C,A);      

            vec2 res;

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
                vec2 x = (vec2(h, -h) - q) / 2.0;

                // Solving Catastrophic Cancellation when h and q are close (when p is near 0)
                if(abs(abs(h/q) - 1.0) < 0.0001)
                {
                   // Approximation of x where h and q are close with no carastrophic cancellation
                   // Derivation (for curious minds) -> h=√(q²+4p³)=q·√(1+4p³/q²)=q·√(1+w)
                      // Now let's do linear taylor expansion of √(1+x) to approximate √(1+w)
                      // Taylor expansion at 0 -> f(x)=f(0)+f'(0)*(x) = 1+½x 
                      // So √(1+w) can be approximated by 1+½w
                      // Simplifying later and the fact that w=4p³/q will result in the following.
                   x = vec2(p3/q, -q - p3/q);
                }

                vec2 uv = sign(x)*pow(abs(x), vec2(1.0/3.0,1.0/3.0));
                float_t t = uv.x + uv.y - kx;
                t = clamp( t, 0.0, 1.0 );

                // 1 root
                vec2 qos = C + (2.0*A + B*t)*t;
                res = vec2( length(qos),t);
            }
            else
            {
                float_t z = sqrt(-p);
                float_t v = acos( q/(p*z*2.0) ) / 3.0;
                float_t m = cos(v);
                float_t n = sin(v)*1.732050808;
                vec3 t = vec3(m + m, -n - m, n - m) * z - kx;
                t = clamp( t, 0.0, 1.0 );

                // 3 roots
                vec2 qos = C + (2.0*A + B*t.x)*t.x;
                float_t dis = dot(qos,qos);
                
                res = vec2(dis,t.x);

                qos = C + (2.0*A + B*t.y)*t.y;
                dis = dot(qos,qos);
                if( dis<res.x ) res = vec2(dis,t.y );

                qos = C + (2.0*A + B*t.z)*t.z;
                dis = dot(qos,qos);
                if( dis<res.x ) res = vec2(dis,t.z );

                res.x = sqrt( res.x );
            }
            
            return res;
        }

        float_t signedDistance(vec2 pos)
        {
            return abs(ud(pos)) - thickness;
        }
    };

    struct CubicBezier
    {
        float2 P0;
        float2 P1;
        float2 P2;
        float2 P3;
        float thickness;

        static CubicBezier construct(float2 A, float2 B, float2 C, float2 D, float thickness)
        {
            CubicBezier ret;
            ret.P0 = A;
            ret.P1 = B;
            ret.P2 = C;
            ret.P3 = D;
            ret.thickness = thickness;
            return ret;
        }

        //lagrange positive real root upper bound
        //see for example: https://doi.org/10.1016/j.jsc.2014.09.038
        float upper_bound_lagrange5(float a0, float a1, float a2, float a3, float a4){

            float4 coeffs1 = float4(a0,a1,a2,a3);

            float4 neg1 = max(-coeffs1,float4(0,0,0,0));
            float neg2 = max(-a4,0.);

            const float4 indizes1 = float4(0,1,2,3);
            const float indizes2 = 4.;

            float4 bounds1 = pow(neg1,1./(5.-indizes1));
            float bounds2 = pow(neg2,1./(5.-indizes2));

            float2 min1_2 = min(bounds1.xz,bounds1.yw);
            float2 max1_2 = max(bounds1.xz,bounds1.yw);

            float maxmin = max(min1_2.x,min1_2.y);
            float minmax = min(max1_2.x,max1_2.y);

            float max3 = max(max1_2.x,max1_2.y);

            float max_max = max(max3,bounds2);
            float max_max2 = max(min(max3,bounds2),max(minmax,maxmin));

            return max_max + max_max2;
        }

        //lagrange upper bound applied to f(-x) to get lower bound
        float lower_bound_lagrange5(float a0, float a1, float a2, float a3, float a4){

            float4 coeffs1 = float4(-a0,a1,-a2,a3);

            float4 neg1 = max(-coeffs1,float4(0,0,0,0));
            float neg2 = max(-a4,0.);

            const float4 indizes1 = float4(0,1,2,3);
            const float indizes2 = 4.;

            float4 bounds1 = pow(neg1,1./(5.-indizes1));
            float bounds2 = pow(neg2,1./(5.-indizes2));

            float2 min1_2 = min(bounds1.xz,bounds1.yw);
            float2 max1_2 = max(bounds1.xz,bounds1.yw);

            float maxmin = max(min1_2.x,min1_2.y);
            float minmax = min(max1_2.x,max1_2.y);

            float max3 = max(max1_2.x,max1_2.y);

            float max_max = max(max3,bounds2);
            float max_max2 = max(min(max3,bounds2),max(minmax,maxmin));

            return -max_max - max_max2;
        }

        float2 parametric_cub_bezier(float t, float2 p0, float2 p1, float2 p2, float2 p3){
            float2 a0 = (-p0 + 3. * p1 - 3. * p2 + p3);
            float2 a1 = (3. * p0  -6. * p1 + 3. * p2);
            float2 a2 = (-3. * p0 + 3. * p1);
            float2 a3 = p0;

            return (((a0 * t) + a1) * t + a2) * t + a3;
        }

        void sort_roots3(inout float3 roots){
            float3 tmp;

            tmp[0] = min(roots[0],min(roots[1],roots[2]));
            tmp[1] = max(roots[0],min(roots[1],roots[2]));
            tmp[2] = max(roots[0],max(roots[1],roots[2]));

            roots=tmp;
        }

        void sort_roots4(inout float4 roots){
            float4 tmp;

            float2 min1_2 = min(roots.xz,roots.yw);
            float2 max1_2 = max(roots.xz,roots.yw);

            float maxmin = max(min1_2.x,min1_2.y);
            float minmax = min(max1_2.x,max1_2.y);

            tmp[0] = min(min1_2.x,min1_2.y);
            tmp[1] = min(maxmin,minmax);
            tmp[2] = max(minmax,maxmin);
            tmp[3] = max(max1_2.x,max1_2.y);

            roots = tmp;
        }

        float eval_poly5(float a0, float a1, float a2, float a3, float a4, float x){

            float f = ((((x + a4) * x + a3) * x + a2) * x + a1) * x + a0;

            return f;
        }

        //halley's method
        //basically a variant of newton raphson which converges quicker and has bigger basins of convergence
        //see http://mathworld.wolfram.com/HalleysMethod.html
        //or https://en.wikipedia.org/wiki/Halley%27s_method
        float halley_iteration5(float a0, float a1, float a2, float a3, float a4, float x){

            float f = ((((x + a4) * x + a3) * x + a2) * x + a1) * x + a0;
            float f1 = (((5. * x + 4. * a4) * x + 3. * a3) * x + 2. * a2) * x + a1;
            float f2 = ((20. * x + 12. * a4) * x + 6. * a3) * x + 2. * a2;

            return x - (2. * f * f1) / (2. * f1 * f1 - f * f2);
        }

        float halley_iteration4(float4 coeffs, float x){

            float f = (((x + coeffs[3]) * x + coeffs[2]) * x + coeffs[1]) * x + coeffs[0];
            float f1 = ((4. * x + 3. * coeffs[3]) * x + 2. * coeffs[2]) * x + coeffs[1];
            float f2 = (12. * x + 6. * coeffs[3]) * x + 2. * coeffs[2];

            return x - (2. * f * f1) / (2. * f1 * f1 - f * f2);
        }

        // Modified from http://tog.acm.org/resources/GraphicsGems/gems/Roots3And4.c
        // Credits to Doublefresh for hinting there
        int solve_quadric(float2 coeffs, inout float2 roots)
        {
            // normal form: x^2 + px + q = 0
            float p = coeffs[1] / 2.;
            float q = coeffs[0];

            float D = p * p - q;

            if (D < 0.){
                return 0;
            }
            else if (D > 0.){
                roots[0] = -sqrt(D) - p;
                roots[1] = sqrt(D) - p;

                return 2;
            }
            else
            {
                roots[0] = -p;
                return 1;
            }
        }

        //From Trisomie21
        //But instead of his cancellation fix i'm using a newton iteration
        int solve_cubic(float3 coeffs, inout float3 r)
        {
            float a = coeffs[2];
            float b = coeffs[1];
            float c = coeffs[0];

            float p = b - a*a / 3.0;
            float q = a * (2.0*a*a - 9.0*b) / 27.0 + c;
            float p3 = p*p*p;
            float d = q*q + 4.0*p3 / 27.0;
            float offset = -a / 3.0;
            if(d >= 0.0) { // Single solution
                float z = sqrt(d);
                float u = (-q + z) / 2.0;
                float v = (-q - z) / 2.0;
                u = sign(u)*pow(abs(u),1.0/3.0);
                v = sign(v)*pow(abs(v),1.0/3.0);
                r[0] = offset + u + v;	

                //Single newton iteration to account for cancellation
                float f = ((r[0] + a) * r[0] + b) * r[0] + c;
                float f1 = (3. * r[0] + 2. * a) * r[0] + b;

                r[0] -= f / f1;

                return 1;
            }
            float u = sqrt(-p / 3.0);
            float v = acos(-sqrt( -27.0 / p3) * q / 2.0) / 3.0;
            float m = cos(v), n = sin(v)*1.732050808;

            //Single newton iteration to account for cancellation
            //(once for every root)
            r[0] = offset + u * (m + m);
            r[1] = offset - u * (n + m);
            r[2] = offset + u * (n - m);

            float3 f = ((r + a) * r + b) * r + c;
            float3 f1 = (3. * r + 2. * a) * r + b;

            r -= f / f1;

            return 3;
        }

        // Modified from http://tog.acm.org/resources/GraphicsGems/gems/Roots3And4.c
        // Credits to Doublefresh for hinting there
        int solve_quartic(float4 coeffs, inout float4 s)
        {
            float a = coeffs[3];
            float b = coeffs[2];
            float c = coeffs[1];
            float d = coeffs[0];

            /*  substitute x = y - A/4 to eliminate cubic term:
            x^4 + px^2 + qx + r = 0 */

            float sq_a = a * a;
            float p = - 3./8. * sq_a + b;
            float q = 1./8. * sq_a * a - 1./2. * a * b + c;
            float r = - 3./256.*sq_a*sq_a + 1./16.*sq_a*b - 1./4.*a*c + d;

            int num;

            /* doesn't seem to happen for me */
            //if(abs(r)<nbl_hlsl_FLT_EPSILON){
            //	/* no absolute term: y(y^3 + py + q) = 0 */

            //	float3 cubic_coeffs;

            //	cubic_coeffs[0] = q;
            //	cubic_coeffs[1] = p;
            //	cubic_coeffs[2] = 0.;

            //	num = solve_cubic(cubic_coeffs, s.xyz);

            //	s[num] = 0.;
            //	num++;
            //}
            {
                /* solve the resolvent cubic ... */
                float3 cubic_coeffs;

                cubic_coeffs[0] = 1.0/2. * r * p - 1.0/8. * q * q;
                cubic_coeffs[1] = - r;
                cubic_coeffs[2] = - 1.0/2. * p;

                solve_cubic(cubic_coeffs, s.xyz);

                /* ... and take the one real solution ... */

                float z = s[0];

                /* ... to build two quadric equations */

                float u = z * z - r;
                float v = 2. * z - p;

                if(u > -nbl_hlsl_FLT_EPSILON){
                    u = sqrt(abs(u));
                }
                else{
                    return 0;
                }

                if(v > -nbl_hlsl_FLT_EPSILON){
                    v = sqrt(abs(v));
                }
                else{
                    return 0;
                }

                float2 quad_coeffs;

                quad_coeffs[0] = z - u;
                quad_coeffs[1] = q < 0. ? -v : v;

                num = solve_quadric(quad_coeffs, s.xy);

                quad_coeffs[0]= z + u;
                quad_coeffs[1] = q < 0. ? v : -v;

                float2 tmp=float2(1e38,1e38);
                int old_num=num;

                num += solve_quadric(quad_coeffs, tmp);
                if(old_num!=num)
                {
                    if(old_num == 0)
                    {
                        s[0] = tmp[0];
                        s[1] = tmp[1];
                    }
                    else
                    {
                        //old_num == 2
                        s[2] = tmp[0];
                        s[3] = tmp[1];
                    }
                }
            }

            /* resubstitute */

            float sub = 1./4. * a;

            /* single halley iteration to fix cancellation */
            for(int i=0;i<4;i+=2){
                if(i < num){
                    s[i] -= sub;
                    s[i] = halley_iteration4(coeffs,s[i]);

                    s[i+1] -= sub;
                    s[i+1] = halley_iteration4(coeffs,s[i+1]);
                }
            }

            return num;
        }

        float cubic_bezier_dis(float2 uv, float2 p0, float2 p1, float2 p2, float2 p3)
        {

            //switch points when near to end point to minimize numerical error
            //only needed when control point(s) very far away
            #if 0
            float2 mid_curve = parametric_cub_bezier(.5,p0,p1,p2,p3);
            float2 mid_points = (p0 + p3)/2.;

            float2 tang = mid_curve-mid_points;
            float2 nor = float2(tang.y,-tang.x);

            if(sign(dot(nor,uv-mid_curve)) != sign(dot(nor,p0-mid_curve))){
                float2 tmp = p0;
                p0 = p3;
                p3 = tmp;

                tmp = p2;
                p2 = p1;
                p1 = tmp;
            }
            #endif

            float2 a3 = (-p0 + 3. * p1 - 3. * p2 + p3);
            float2 a2 = (3. * p0 - 6. * p1 + 3. * p2);
            float2 a1 = (-3. * p0 + 3. * p1);
            float2 a0 = p0 - uv;
            
            //compute polynomial describing distance to current pixel dependent on a parameter t
            float bc6 = dot(a3,a3);
            float bc5 = 2.*dot(a3,a2);
            float bc4 = dot(a2,a2) + 2.*dot(a1,a3);
            float bc3 = 2.*(dot(a1,a2) + dot(a0,a3));
            float bc2 = dot(a1,a1) + 2.*dot(a0,a2);
            float bc1 = 2.*dot(a0,a1);
            float bc0 = dot(a0,a0);

            bc5 /= bc6;
            bc4 /= bc6;
            bc3 /= bc6;
            bc2 /= bc6;
            bc1 /= bc6;
            bc0 /= bc6;
            
            //compute derivatives of this polynomial

            float b0 = bc1 / 6.;
            float b1 = 2. * bc2 / 6.;
            float b2 = 3. * bc3 / 6.;
            float b3 = 4. * bc4 / 6.;
            float b4 = 5. * bc5 / 6.;

            float4 c1 = float4(b1,2.*b2,3.*b3,4.*b4)/5.;
            float3 c2 = float3(c1[1],2.*c1[2],3.*c1[3])/4.;
            float2 c3 = float2(c2[1],2.*c2[2])/3.;
            float c4 = c3[1]/2.;

            float4 roots_drv = float4(1e38,1e38,1e38,1e38);

            int num_roots_drv = solve_quartic(c1,roots_drv);
            sort_roots4(roots_drv);

            float ub = upper_bound_lagrange5(b0,b1,b2,b3,b4);
            float lb = lower_bound_lagrange5(b0,b1,b2,b3,b4);

            float3 a = float3(1e38,1e38,1e38);
            float3 b = float3(1e38,1e38,1e38);

            float3 roots = float3(1e38,1e38,1e38);

            int num_roots = 0;
            
            //compute root isolating intervals by roots of derivative and outer root bounds
            //only roots going form - to + considered, because only those result in a minimum
            if(num_roots_drv==4)
            {
                if(eval_poly5(b0,b1,b2,b3,b4,roots_drv[0]) > 0.)
                {
                    a[0]=lb;
                    b[0]=roots_drv[0];
                    num_roots=1;
                }

                if(sign(eval_poly5(b0,b1,b2,b3,b4,roots_drv[1])) != sign(eval_poly5(b0,b1,b2,b3,b4,roots_drv[2])))
                {
                    if(num_roots == 0){
                        a[0]=roots_drv[1];
                        b[0]=roots_drv[2];
                        num_roots=1;
                    }
                    else{
                        a[1]=roots_drv[1];
                        b[1]=roots_drv[2];
                        num_roots=2;
                    }
                }

                if(eval_poly5(b0,b1,b2,b3,b4,roots_drv[3]) < 0.)
                {
                    if(num_roots == 0){
                        a[0]=roots_drv[3];
                        b[0]=ub;
                        num_roots=1;
                    }
                    else if(num_roots == 1){
                        a[1]=roots_drv[3];
                        b[1]=ub;
                        num_roots=2;
                    }
                    else{
                        a[2]=roots_drv[3];
                        b[2]=ub;
                        num_roots=3;
                    }
                }
            }
            else
            {
                if(num_roots_drv==2)
                {
                    if(eval_poly5(b0,b1,b2,b3,b4,roots_drv[0]) < 0.){
                        num_roots=1;
                        a[0]=roots_drv[1];
                        b[0]=ub;
                    }
                    else if(eval_poly5(b0,b1,b2,b3,b4,roots_drv[1]) > 0.){
                        num_roots=1;
                        a[0]=lb;
                        b[0]=roots_drv[0];
                    }
                    else{
                        num_roots=2;

                        a[0]=lb;
                        b[0]=roots_drv[0];

                        a[1]=roots_drv[1];
                        b[1]=ub;
                    }

                }
                else{ //num_roots_drv==0
                    float3 roots_snd_drv=float3(1e38,1e38,1e38);
                    int num_roots_snd_drv=solve_cubic(c2,roots_snd_drv);

                    float2 roots_trd_drv=float2(1e38,1e38);
                    int num_roots_trd_drv=solve_quadric(c3,roots_trd_drv);
                    num_roots=1;

                    a[0]=lb;
                    b[0]=ub;
                }
                
                //further subdivide intervals to guarantee convergence of halley's method
                //by using roots of further derivatives
                float3 roots_snd_drv=float3(1e38,1e38,1e38);
                int num_roots_snd_drv=solve_cubic(c2,roots_snd_drv);
                sort_roots3(roots_snd_drv);

                int num_roots_trd_drv=0;
                float2 roots_trd_drv=float2(1e38,1e38);

                if(num_roots_snd_drv!=3){
                    num_roots_trd_drv=solve_quadric(c3,roots_trd_drv);
                }

                for(int i=0;i<3;i++){
                    if(i < num_roots){
                        {
                        for(int j=0;j<3;j+=2){
                            if(j < num_roots_snd_drv){
                                if(a[i] < roots_snd_drv[j] && b[i] > roots_snd_drv[j]){
                                    if(eval_poly5(b0,b1,b2,b3,b4,roots_snd_drv[j]) > 0.){
                                        b[i]=roots_snd_drv[j];
                                    }
                                    else{
                                        a[i]=roots_snd_drv[j];
                                    }
                                }
                            }
                        }
                        }
                        
                        for(int j=0;j<2;j++){
                            if(j < num_roots_trd_drv){
                                if(a[i] < roots_trd_drv[j] && b[i] > roots_trd_drv[j]){
                                    if(eval_poly5(b0,b1,b2,b3,b4,roots_trd_drv[j]) > 0.){
                                        b[i]=roots_trd_drv[j];
                                    }
                                    else{
                                        a[i]=roots_trd_drv[j];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            float d0 = 1e38;

            //compute roots with halley's method
            const int halley_iterations = 8;
            
            for(int i=0;i<3;i++){
                if(i < num_roots){
                    roots[i] = .5 * (a[i] + b[i]);

                    for(int j=0;j<halley_iterations;j++){
                        roots[i] = halley_iteration5(b0,b1,b2,b3,b4,roots[i]);
                    }
                    

                    //compute squared distance to nearest point on curve
                    roots[i] = clamp(roots[i],0.,1.);
                    float2 to_curve = uv - parametric_cub_bezier(roots[i],p0,p1,p2,p3);
                    d0 = min(d0,dot(to_curve,to_curve));
                }
            }

            return sqrt(d0);
        }

        float signedDistance(float2 pos)
        {
            return abs(cubic_bezier_dis(pos, P0, P1, P2, P3)) - thickness;
        }
    };
}
}
}