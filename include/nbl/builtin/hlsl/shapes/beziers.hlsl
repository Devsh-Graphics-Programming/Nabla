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
    float32_t2 SolveQuadratic(float32_t3 c, out int32_t numRealValues)
    {
        // bhaskara: x = (-b ± √(b² – 4ac)) / (2a)
        float32_t b = c.y / (2 * c.z);
        float32_t q = c.x / c.z;
        float32_t delta = b * b - q;

        if (delta == 0.0) // Δ = 0
        {
            numRealValues = 1;
            return float32_t2(-b, 0.0);
        }
        if (delta < 0) // Δ < 0 (no real values)
        {
            numRealValues = 0;
            return 0.0;
        }

        // Δ > 0 (two distinct real values)
        float32_t sqrtD = sqrt(delta);
        numRealValues = 2;
        return float32_t2(sqrtD - b, sqrtD + b);
    }

    struct QuadraticBezier
    {
        float32_t2 A;
        float32_t2 B;
        float32_t2 C;

        static QuadraticBezier construct(float32_t2 a, float32_t2 b, float32_t2 c)
        {
            QuadraticBezier ret = { a, b, c };
            return ret;
        }

        float32_t2 evaluate(float32_t t)
        {
            float32_t2 position = A * (1.0 - t) * (1.0 - t) 
                      + 2.0 * B * (1.0 - t) * t
                      +       C * t         * t;
            return position;
        }

        // https://pomax.github.io/bezierinfo/#yforx
        float32_t tForMajorCoordinate(const int32_t major, float32_t x) 
        { 
            float32_t a = A[major] - x;
            float32_t b = B[major] - x;
            float32_t c = C[major] - x;
            int32_t rootCount;
            float32_t2 roots = SolveQuadratic(float32_t3(a, b, c), rootCount);
            // assert(rootCount == 1);
            return roots.x;
        }
    };

    struct QuadraticBezierOutline
    {
        QuadraticBezier bezier;
        float32_t thickness;

        static QuadraticBezierOutline construct(float32_t2 a, float32_t2 b, float32_t2 c, float32_t thickness)
        {
            QuadraticBezier bezier = { a, b, c };
            QuadraticBezierOutline ret = { bezier, thickness };
            return ret;
        }

        // original from https://www.shadertoy.com/view/lsyfWc
        float32_t ud(float32_t2 pos)
        {
            const float32_t2 A = bezier.A;
            const float32_t2 B = bezier.B;
            const float32_t2 C = bezier.C;
                    
            // p(t)    = (1-t)^2*A + 2(1-t)t*B + t^2*C
            // p'(t)   = 2*t*(A-2*B+C) + 2*(B-A)
            // p'(0)   = 2(B-A)
            // p'(1)   = 2(C-B)
            // p'(1/2) = 2(C-A)
                    
            float32_t2 a = B - A;
            float32_t2 b = A - 2.0*B + C;
            float32_t2 c = A - pos;

            // Reducing Quartic to Cubic Solution
            float32_t kk = 1.0 / dot(b,b);
            float32_t kx = kk * dot(a,b);
            float32_t ky = kk * (2.0*dot(a,a)+dot(c,b)) / 3.0;
            float32_t kz = kk * dot(c,a);      

            float32_t2 res;

            // Cardano's Solution to resolvent cubic of the form: y^3 + 3py + q = 0
            // where it was initially of the form x^3 + ax^2 + bx + c = 0 and x was replaced by y - a/3
            // so a/3 needs to be subtracted from the solution to the first form to get the actual solution
            float32_t p = ky - kx*kx;
            float32_t p3 = p*p*p;
            float32_t q = kx*(2.0*kx*kx - 3.0*ky) + kz;
            float32_t h = q*q + 4.0*p3;

            if(h >= 0.0) 
            { 
                h = sqrt(h);
                float32_t2 x = (float32_t2(h, -h) - q) / 2.0;

                // Solving Catastrophic Cancellation when h and q are close (when p is near 0)
                if(abs(abs(h/q) - 1.0) < 0.0001)
                {
                   // Approximation of x where h and q are close with no carastrophic cancellation
                   // Derivation (for curious minds) -> h=√(q²+4p³)=q·√(1+4p³/q²)=q·√(1+w)
                      // Now let's do linear taylor expansion of √(1+x) to approximate √(1+w)
                      // Taylor expansion at 0 -> f(x)=f(0)+f'(0)*(x) = 1+½x 
                      // So √(1+w) can be approximated by 1+½w
                      // Simplifying later and the fact that w=4p³/q will result in the following.
                   x = float32_t2(p3/q, -q - p3/q);
                }

                float32_t2 uv = sign(x)*pow(abs(x), float32_t2(1.0/3.0,1.0/3.0));
                float32_t t = uv.x + uv.y - kx;
                t = clamp( t, 0.0, 1.0 );

                // 1 root
                float32_t2 qos = c + (2.0*a + b*t)*t;
                res = float32_t2( length(qos),t);
            }
            else
            {
                float32_t z = sqrt(-p);
                float32_t v = acos( q/(p*z*2.0) ) / 3.0;
                float32_t m = cos(v);
                float32_t n = sin(v)*1.732050808;
                float32_t3 t = float32_t3(m + m, -n - m, n - m) * z - kx;
                t = clamp( t, 0.0, 1.0 );

                // 3 roots
                float32_t2 qos = c + (2.0*a + b*t.x)*t.x;
                float32_t dis = dot(qos,qos);
                
                res = float32_t2(dis,t.x);

                qos = c + (2.0*a + b*t.y)*t.y;
                dis = dot(qos,qos);
                if( dis<res.x ) res = float32_t2(dis,t.y );

                qos = c + (2.0*a + b*t.z)*t.z;
                dis = dot(qos,qos);
                if( dis<res.x ) res = float32_t2(dis,t.z );

                res.x = sqrt( res.x );
            }
            
            return res.x;
        }

        float32_t signedDistance(float32_t2 pos)
        {
            return abs(ud(pos)) - thickness;
        }
    };

    struct CubicBezier
    {
        float32_t2 P0;
        float32_t2 P1;
        float32_t2 P2;
        float32_t2 P3;
        float32_t thickness;

        static CubicBezier construct(float32_t2 a, float32_t2 b, float32_t2 c, float32_t2 d, float32_t thickness)
        {
            CubicBezier ret;
            ret.P0 = a;
            ret.P1 = b;
            ret.P2 = c;
            ret.P3 = d;
            ret.thickness = thickness;
            return ret;
        }

        //lagrange positive real root upper bound
        //see for example: https://doi.org/10.1016/j.jsc.2014.09.038
        float32_t upper_bound_lagrange5(float32_t a0, float32_t a1, float32_t a2, float32_t a3, float32_t a4){

            float32_t4 coeffs1 = float32_t4(a0,a1,a2,a3);

            float32_t4 neg1 = max(-coeffs1,float32_t4(0,0,0,0));
            float32_t neg2 = max(-a4,0.);

            const float32_t4 indizes1 = float32_t4(0,1,2,3);
            const float32_t indizes2 = 4.;

            float32_t4 bounds1 = pow(neg1,1./(5.-indizes1));
            float32_t bounds2 = pow(neg2,1./(5.-indizes2));

            float32_t2 min1_2 = min(bounds1.xz,bounds1.yw);
            float32_t2 max1_2 = max(bounds1.xz,bounds1.yw);

            float32_t maxmin = max(min1_2.x,min1_2.y);
            float32_t minmax = min(max1_2.x,max1_2.y);

            float32_t max3 = max(max1_2.x,max1_2.y);

            float32_t max_max = max(max3,bounds2);
            float32_t max_max2 = max(min(max3,bounds2),max(minmax,maxmin));

            return max_max + max_max2;
        }

        //lagrange upper bound applied to f(-x) to get lower bound
        float32_t lower_bound_lagrange5(float32_t a0, float32_t a1, float32_t a2, float32_t a3, float32_t a4){

            float32_t4 coeffs1 = float32_t4(-a0,a1,-a2,a3);

            float32_t4 neg1 = max(-coeffs1,float32_t4(0,0,0,0));
            float32_t neg2 = max(-a4,0.);

            const float32_t4 indizes1 = float32_t4(0,1,2,3);
            const float32_t indizes2 = 4.;

            float32_t4 bounds1 = pow(neg1,1./(5.-indizes1));
            float32_t bounds2 = pow(neg2,1./(5.-indizes2));

            float32_t2 min1_2 = min(bounds1.xz,bounds1.yw);
            float32_t2 max1_2 = max(bounds1.xz,bounds1.yw);

            float32_t maxmin = max(min1_2.x,min1_2.y);
            float32_t minmax = min(max1_2.x,max1_2.y);

            float32_t max3 = max(max1_2.x,max1_2.y);

            float32_t max_max = max(max3,bounds2);
            float32_t max_max2 = max(min(max3,bounds2),max(minmax,maxmin));

            return -max_max - max_max2;
        }

        float32_t2 parametric_cub_bezier(float32_t t, float32_t2 p0, float32_t2 p1, float32_t2 p2, float32_t2 p3){
            float32_t2 a0 = (-p0 + 3. * p1 - 3. * p2 + p3);
            float32_t2 a1 = (3. * p0  -6. * p1 + 3. * p2);
            float32_t2 a2 = (-3. * p0 + 3. * p1);
            float32_t2 a3 = p0;

            return (((a0 * t) + a1) * t + a2) * t + a3;
        }

        void sort_roots3(inout float32_t3 roots){
            float32_t3 tmp;

            tmp[0] = min(roots[0],min(roots[1],roots[2]));
            tmp[1] = max(roots[0],min(roots[1],roots[2]));
            tmp[2] = max(roots[0],max(roots[1],roots[2]));

            roots=tmp;
        }

        void sort_roots4(inout float32_t4 roots){
            float32_t4 tmp;

            float32_t2 min1_2 = min(roots.xz,roots.yw);
            float32_t2 max1_2 = max(roots.xz,roots.yw);

            float32_t maxmin = max(min1_2.x,min1_2.y);
            float32_t minmax = min(max1_2.x,max1_2.y);

            tmp[0] = min(min1_2.x,min1_2.y);
            tmp[1] = min(maxmin,minmax);
            tmp[2] = max(minmax,maxmin);
            tmp[3] = max(max1_2.x,max1_2.y);

            roots = tmp;
        }

        float32_t eval_poly5(float32_t a0, float32_t a1, float32_t a2, float32_t a3, float32_t a4, float32_t x){

            float32_t f = ((((x + a4) * x + a3) * x + a2) * x + a1) * x + a0;

            return f;
        }

        //halley's method
        //basically a variant of newton raphson which converges quicker and has bigger basins of convergence
        //see http://mathworld.wolfram.com/HalleysMethod.html
        //or https://en.wikipedia.org/wiki/Halley%27s_method
        float32_t halley_iteration5(float32_t a0, float32_t a1, float32_t a2, float32_t a3, float32_t a4, float32_t x){

            float32_t f = ((((x + a4) * x + a3) * x + a2) * x + a1) * x + a0;
            float32_t f1 = (((5. * x + 4. * a4) * x + 3. * a3) * x + 2. * a2) * x + a1;
            float32_t f2 = ((20. * x + 12. * a4) * x + 6. * a3) * x + 2. * a2;

            return x - (2. * f * f1) / (2. * f1 * f1 - f * f2);
        }

        float32_t halley_iteration4(float32_t4 coeffs, float32_t x){

            float32_t f = (((x + coeffs[3]) * x + coeffs[2]) * x + coeffs[1]) * x + coeffs[0];
            float32_t f1 = ((4. * x + 3. * coeffs[3]) * x + 2. * coeffs[2]) * x + coeffs[1];
            float32_t f2 = (12. * x + 6. * coeffs[3]) * x + 2. * coeffs[2];

            return x - (2. * f * f1) / (2. * f1 * f1 - f * f2);
        }

        // Modified from http://tog.acm.org/resources/GraphicsGems/gems/Roots3And4.c
        // Credits to Doublefresh for hinting there
        int32_t solve_quadric(float32_t2 coeffs, inout float32_t2 roots)
        {
            // normal form: x^2 + px + q = 0
            float32_t p = coeffs[1] / 2.;
            float32_t q = coeffs[0];

            float32_t D = p * p - q;

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
        int32_t solve_cubic(float32_t3 coeffs, inout float32_t3 r)
        {
            float32_t a = coeffs[2];
            float32_t b = coeffs[1];
            float32_t c = coeffs[0];

            float32_t p = b - a*a / 3.0;
            float32_t q = a * (2.0*a*a - 9.0*b) / 27.0 + c;
            float32_t p3 = p*p*p;
            float32_t d = q*q + 4.0*p3 / 27.0;
            float32_t offset = -a / 3.0;
            if(d >= 0.0) { // Single solution
                float32_t z = sqrt(d);
                float32_t u = (-q + z) / 2.0;
                float32_t v = (-q - z) / 2.0;
                u = sign(u)*pow(abs(u),1.0/3.0);
                v = sign(v)*pow(abs(v),1.0/3.0);
                r[0] = offset + u + v;	

                //Single newton iteration to account for cancellation
                float32_t f = ((r[0] + a) * r[0] + b) * r[0] + c;
                float32_t f1 = (3. * r[0] + 2. * a) * r[0] + b;

                r[0] -= f / f1;

                return 1;
            }
            float32_t u = sqrt(-p / 3.0);
            float32_t v = acos(-sqrt( -27.0 / p3) * q / 2.0) / 3.0;
            float32_t m = cos(v), n = sin(v)*1.732050808;

            //Single newton iteration to account for cancellation
            //(once for every root)
            r[0] = offset + u * (m + m);
            r[1] = offset - u * (n + m);
            r[2] = offset + u * (n - m);

            float32_t3 f = ((r + a) * r + b) * r + c;
            float32_t3 f1 = (3. * r + 2. * a) * r + b;

            r -= f / f1;

            return 3;
        }

        // Modified from http://tog.acm.org/resources/GraphicsGems/gems/Roots3And4.c
        // Credits to Doublefresh for hinting there
        int32_t solve_quartic(float32_t4 coeffs, inout float32_t4 s)
        {
            float32_t a = coeffs[3];
            float32_t b = coeffs[2];
            float32_t c = coeffs[1];
            float32_t d = coeffs[0];

            /*  substitute x = y - A/4 to eliminate cubic term:
            x^4 + px^2 + qx + r = 0 */

            float32_t sq_a = a * a;
            float32_t p = - 3./8. * sq_a + b;
            float32_t q = 1./8. * sq_a * a - 1./2. * a * b + c;
            float32_t r = - 3./256.*sq_a*sq_a + 1./16.*sq_a*b - 1./4.*a*c + d;

            int32_t num;

            /* doesn't seem to happen for me */
            //if(abs(r)<nbl_hlsl_FLT_EPSILON){
            //	/* no absolute term: y(y^3 + py + q) = 0 */

            //	float32_t3 cubic_coeffs;

            //	cubic_coeffs[0] = q;
            //	cubic_coeffs[1] = p;
            //	cubic_coeffs[2] = 0.;

            //	num = solve_cubic(cubic_coeffs, s.xyz);

            //	s[num] = 0.;
            //	num++;
            //}
            {
                /* solve the resolvent cubic ... */
                float32_t3 cubic_coeffs;

                cubic_coeffs[0] = 1.0/2. * r * p - 1.0/8. * q * q;
                cubic_coeffs[1] = - r;
                cubic_coeffs[2] = - 1.0/2. * p;

                solve_cubic(cubic_coeffs, s.xyz);

                /* ... and take the one real solution ... */

                float32_t z = s[0];

                /* ... to build two quadric equations */

                float32_t u = z * z - r;
                float32_t v = 2. * z - p;

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

                float32_t2 quad_coeffs;

                quad_coeffs[0] = z - u;
                quad_coeffs[1] = q < 0. ? -v : v;

                num = solve_quadric(quad_coeffs, s.xy);

                quad_coeffs[0]= z + u;
                quad_coeffs[1] = q < 0. ? v : -v;

                float32_t2 tmp=float32_t2(1e38,1e38);
                int32_t old_num=num;

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

            float32_t sub = 1./4. * a;

            /* single halley iteration to fix cancellation */
            for(int32_t i=0;i<4;i+=2){
                if(i < num){
                    s[i] -= sub;
                    s[i] = halley_iteration4(coeffs,s[i]);

                    s[i+1] -= sub;
                    s[i+1] = halley_iteration4(coeffs,s[i+1]);
                }
            }

            return num;
        }

        float32_t cubic_bezier_dis(float32_t2 uv, float32_t2 p0, float32_t2 p1, float32_t2 p2, float32_t2 p3)
        {

            //switch points when near to end point to minimize numerical error
            //only needed when control point(s) very far away
            #if 0
            float32_t2 mid_curve = parametric_cub_bezier(.5,p0,p1,p2,p3);
            float32_t2 mid_points = (p0 + p3)/2.;

            float32_t2 tang = mid_curve-mid_points;
            float32_t2 nor = float32_t2(tang.y,-tang.x);

            if(sign(dot(nor,uv-mid_curve)) != sign(dot(nor,p0-mid_curve))){
                float32_t2 tmp = p0;
                p0 = p3;
                p3 = tmp;

                tmp = p2;
                p2 = p1;
                p1 = tmp;
            }
            #endif

            float32_t2 a3 = (-p0 + 3. * p1 - 3. * p2 + p3);
            float32_t2 a2 = (3. * p0 - 6. * p1 + 3. * p2);
            float32_t2 a1 = (-3. * p0 + 3. * p1);
            float32_t2 a0 = p0 - uv;
            
            //compute polynomial describing distance to current pixel dependent on a parameter t
            float32_t bc6 = dot(a3,a3);
            float32_t bc5 = 2.*dot(a3,a2);
            float32_t bc4 = dot(a2,a2) + 2.*dot(a1,a3);
            float32_t bc3 = 2.*(dot(a1,a2) + dot(a0,a3));
            float32_t bc2 = dot(a1,a1) + 2.*dot(a0,a2);
            float32_t bc1 = 2.*dot(a0,a1);
            float32_t bc0 = dot(a0,a0);

            bc5 /= bc6;
            bc4 /= bc6;
            bc3 /= bc6;
            bc2 /= bc6;
            bc1 /= bc6;
            bc0 /= bc6;
            
            //compute derivatives of this polynomial

            float32_t b0 = bc1 / 6.;
            float32_t b1 = 2. * bc2 / 6.;
            float32_t b2 = 3. * bc3 / 6.;
            float32_t b3 = 4. * bc4 / 6.;
            float32_t b4 = 5. * bc5 / 6.;

            float32_t4 c1 = float32_t4(b1,2.*b2,3.*b3,4.*b4)/5.;
            float32_t3 c2 = float32_t3(c1[1],2.*c1[2],3.*c1[3])/4.;
            float32_t2 c3 = float32_t2(c2[1],2.*c2[2])/3.;
            float32_t c4 = c3[1]/2.;

            float32_t4 roots_drv = float32_t4(1e38,1e38,1e38,1e38);

            int32_t num_roots_drv = solve_quartic(c1,roots_drv);
            sort_roots4(roots_drv);

            float32_t ub = upper_bound_lagrange5(b0,b1,b2,b3,b4);
            float32_t lb = lower_bound_lagrange5(b0,b1,b2,b3,b4);

            float32_t3 a = float32_t3(1e38,1e38,1e38);
            float32_t3 b = float32_t3(1e38,1e38,1e38);

            float32_t3 roots = float32_t3(1e38,1e38,1e38);

            int32_t num_roots = 0;
            
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
                    float32_t3 roots_snd_drv=float32_t3(1e38,1e38,1e38);
                    int32_t num_roots_snd_drv=solve_cubic(c2,roots_snd_drv);

                    float32_t2 roots_trd_drv=float32_t2(1e38,1e38);
                    int32_t num_roots_trd_drv=solve_quadric(c3,roots_trd_drv);
                    num_roots=1;

                    a[0]=lb;
                    b[0]=ub;
                }
                
                //further subdivide intervals to guarantee convergence of halley's method
                //by using roots of further derivatives
                float32_t3 roots_snd_drv=float32_t3(1e38,1e38,1e38);
                int32_t num_roots_snd_drv=solve_cubic(c2,roots_snd_drv);
                sort_roots3(roots_snd_drv);

                int32_t num_roots_trd_drv=0;
                float32_t2 roots_trd_drv=float32_t2(1e38,1e38);

                if(num_roots_snd_drv!=3){
                    num_roots_trd_drv=solve_quadric(c3,roots_trd_drv);
                }

                for(int32_t i=0;i<3;i++){
                    if(i < num_roots){
                        {
                        for(int32_t j=0;j<3;j+=2){
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
                        
                        for(int32_t j=0;j<2;j++){
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

            float32_t d0 = 1e38;

            //compute roots with halley's method
            const int32_t halley_iterations = 8;
            
            for(int32_t i=0;i<3;i++){
                if(i < num_roots){
                    roots[i] = .5 * (a[i] + b[i]);

                    for(int32_t j=0;j<halley_iterations;j++){
                        roots[i] = halley_iteration5(b0,b1,b2,b3,b4,roots[i]);
                    }
                    

                    //compute squared distance to nearest point on curve
                    roots[i] = clamp(roots[i],0.,1.);
                    float32_t2 to_curve = uv - parametric_cub_bezier(roots[i],p0,p1,p2,p3);
                    d0 = min(d0,dot(to_curve,to_curve));
                }
            }

            return sqrt(d0);
        }


        float32_t signedDistance(float32_t2 pos)
        {
            return abs(cubic_bezier_dis(pos, P0, P1, P2, P3)) - thickness;
        }
    };
}
}
}