// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
namespace nbl
{
namespace hlsl
{
namespace shapes
{
    float dot2( float2 v ) { return dot(v,v); }
    float cross2d( float2 a, float2 b ) { return a.x*b.y - a.y*b.x; }

    struct QuadraticBezier
    {
        float2 A;
        float2 B;
        float2 C;
        float thickness;

        static QuadraticBezier construct(float2 a, float2 b, float2 c, float thickness)
        {
            QuadraticBezier ret = { a, b, c, thickness };
            return ret;
        }

        float signedDistance(float2 pos)
        {
            float2 a = B - A;
            float2 b = A - 2.0*B + C;
            float2 c = a * 2.0;
            float2 d = A - pos;

            float kk = 1.0/dot(b,b);
            float kx = kk * dot(a,b);
            float ky = kk * (2.0*dot(a,a)+dot(d,b))/3.0;
            float kz = kk * dot(d,a);      

            float res = 0.0;
            float sgn = 0.0;

            float p  = ky - kx*kx;
            float q  = kx*(2.0*kx*kx - 3.0*ky) + kz;
            float p3 = p*p*p;
            float q2 = q*q;
            float h  = q2 + 4.0*p3;

            if( h>=0.0 ) 
            {   
                // 1 root
                h = sqrt(h);
                float2 x = (float2(h,-h)-q)/2.0;

                #if 0
                // When p≈0 and p<0, h-q has catastrophic cancelation. So, we do
                // h=√(q²+4p³)=q·√(1+4p³/q²)=q·√(1+w) instead. Now we approximate
                // √ by a linear Taylor expansion into h≈q(1+½w) so that the q's
                // cancel each other in h-q. Expanding and simplifying further we
                // get x=float2(p³/q,-p³/q-q). And using a second degree Taylor
                // expansion instead: x=float2(k,-k-q) with k=(1-p³/q²)·p³/q
                if( abs(p)<0.001 )
                {
                    float k = p3/q;              // linear approx
                //float k = (1.0-p3/q2)*p3/q;  // quadratic approx 
                    x = float2(k,-k-q);  
                }
                #endif

                float2 uv = sign(x)*pow(abs(x), float2(1.0/3.0, 1.0/3.0));
                float t = clamp( uv.x+uv.y-kx, 0.0, 1.0 );
                float2  q = d+(c+b*t)*t;
                res = dot2(q);
                sgn = cross2d(c+2.0*b*t,q);
            }
            else 
            {   // 3 roots
                float z = sqrt(-p);
                float v = acos(q/(p*z*2.0))/3.0;
                float m = cos(v);
                float n = sin(v)*1.732050808;
                float3  t = clamp( float3(m+m,-n-m,n-m)*z-kx, 0.0, 1.0 );
                float2  qx=d+(c+b*t.x)*t.x; float dx=dot2(qx), sx = cross2d(c+2.0*b*t.x,qx);
                float2  qy=d+(c+b*t.y)*t.y; float dy=dot2(qy), sy = cross2d(c+2.0*b*t.y,qy);
                if( dx<dy ) { res=dx; sgn=sx; } else {res=dy; sgn=sy; }
            }
            
            return sqrt( res )*sign(sgn);
        }
    };

    struct CubicBezier
    {
        float2 a;
        float2 b;
        float2 c;
        float2 d;
        float thickness;

        static CubicBezier construct(float2 a, float2 b, float2 c, float2 d, float thickness)
        {
            CubicBezier ret = { a, b, c, d, thickness };
            return ret;
        }

        float signedDistance(float2 p)
        {
            return 0;
        }
    };
}
}
}