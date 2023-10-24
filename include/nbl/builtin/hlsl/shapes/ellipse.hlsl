// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SHAPES_ELLIPSE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SHAPES_ELLIPSE_INCLUDED_

#include <nbl/builtin/hlsl/shapes/circle.hlsl>

namespace nbl
{
namespace hlsl
{
namespace shapes
{
            float32_t msign(in float32_t x) { return (x < 0.0) ? -1.0 : 1.0; }
            // Ellipse Shape Centered at (0,0) and Major-Axis aligned with X-Axis
            struct Ellipse_t
            {
                float32_t majorAxisLength;
                float32_t eccentricity;

                static Ellipse_t construct(float32_t majorAxisLength, float32_t eccentricity)
                {
                    Ellipse_t ret = { majorAxisLength, eccentricity };
                    return ret;
                }

                // https://iquilezles.org/articles/ellipsedist/ with modifications to add rotation and different inputs and fixed degenerate points
                // major axis is in the direction of +x and minor axis is in the direction of +y
                // @param p should be in ellipse space -> center of ellipse is (0,0)
                float32_t signedDistance(float32_t2 p)
                {
                    if (eccentricity == 1.0)
                        return length(p) - majorAxisLength;

                    float32_t minorAxisLength = majorAxisLength * eccentricity;
                    float32_t2 ab = float32_t2(majorAxisLength, minorAxisLength);
                    p = abs(p);

                    if (p.x > p.y) { p = p.yx; ab = ab.yx; }

                    float32_t l = ab.y * ab.y - ab.x * ab.x;

                    float32_t m = ab.x * p.x / l;
                    float32_t n = ab.y * p.y / l;
                    float32_t m2 = m * m;
                    float32_t n2 = n * n;

                    float32_t c = (m2 + n2 - 1.0) / 3.0;
                    float32_t c3 = c * c * c;

                    float32_t d = c3 + m2 * n2;
                    float32_t q = d + m2 * n2;
                    float32_t g = m + m * n2;

                    float32_t co;

                    if (d < 0.0)
                    {
                        float32_t h = acos(q / c3) / 3.0;
                        float32_t s = cos(h) + 2.0;
                        float32_t t = sin(h) * sqrt(3.0);
                        float32_t rx = sqrt(max(m2 - c * (s + t), 0.0));
                        float32_t ry = sqrt(max(m2 - c * (s - t), 0.0));
                        co = ry + sign(l) * rx + abs(g) / (rx * ry);
                    }
                    else
                    {
                        float32_t h = 2.0 * m * n * sqrt(d);
                        float32_t s = msign(q + h) * pow(abs(q + h), 1.0 / 3.0);
                        float32_t t = msign(q - h) * pow(abs(q - h), 1.0 / 3.0);
                        float32_t rx = -(s + t) - c * 4.0 + 2.0 * m2;
                        float32_t ry = (s - t) * sqrt(3.0);
                        float32_t rm = sqrt(max(rx * rx + ry * ry, 0.0));
                        co = ry / sqrt(max(rm - rx, 0.0)) + 2.0 * g / rm;
                    }
                    co = (co - m) / 2.0;

                    float32_t si = sqrt(max(1.0 - co * co, 0.0));

                    float32_t2 r = ab * float32_t2(co, si);

                    return length(r - p) * msign(p.y - r.y);
                }
            };

            // Ellipse Outline
            struct EllipseOutline_t
            {
                float32_t2 center;
                float32_t2 majorAxis;
                float32_t eccentricity;
                float32_t thickness;

                static EllipseOutline_t construct(float32_t2 center, float32_t2 majorAxis, float32_t eccentricity, float32_t thickness)
                {
                    EllipseOutline_t ret = { center, majorAxis, eccentricity, thickness };
                    return ret;
                }

                float32_t signedDistance(float32_t2 p)
                {
                    float32_t majorAxisLength = length(majorAxis);
                    float32_t2 dir = majorAxis / majorAxisLength;
                    p = p - center;
                    p = mul(float32_t2x2(dir.x, dir.y, -dir.y, dir.x), p);

                    float32_t ellipseDist = Ellipse_t::construct(majorAxisLength, eccentricity).signedDistance(p);
                    return abs(ellipseDist) - thickness;
                }
            };

            // Ellipse Outline
            struct EllipseOutlineBounded_t
            {
                float32_t2 center;
                float32_t2 majorAxis;
                float32_t2 bounds;
                float32_t eccentricity;
                float32_t thickness;

                // @param bounds is in [0, 2PI]
                //      bounds.y-bounds.x should be <= PI
                static EllipseOutlineBounded_t construct(float32_t2 center, float32_t2 majorAxis, float32_t2 bounds, float32_t eccentricity, float32_t thickness)
                {
                    EllipseOutlineBounded_t ret = { center, majorAxis, bounds, eccentricity, thickness };
                    return ret;
                }

                float32_t signedDistance(float32_t2 p)
                {
                    float32_t majorAxisLength = length(majorAxis);
                    float32_t2 dir = majorAxis / majorAxisLength;
                    p = p - center;
                    p = mul(float32_t2x2(dir.x, dir.y, -dir.y, dir.x), p);

                    float32_t2 pNormalized = normalize(p);
                    float32_t theta = atan2(pNormalized.y, -pNormalized.x * eccentricity) + nbl_hlsl_PI;

                    float32_t minorAxisLength = majorAxisLength * eccentricity;

                    float32_t2 startPoint = float32_t2(majorAxisLength * cos(bounds.x), -minorAxisLength * sin(bounds.x));
                    float32_t2 endPoint = float32_t2(majorAxisLength * cos(bounds.y), -minorAxisLength * sin(bounds.y));

                    float32_t2 startTangent = float32_t2(-majorAxisLength * sin(bounds.x), -minorAxisLength * cos(bounds.x));
                    float32_t2 endTangent = float32_t2(-majorAxisLength * sin(bounds.y), -minorAxisLength * cos(bounds.y));
                    float32_t dotStart = dot(startTangent, float32_t2(p - startPoint));
                    float32_t dotEnd = dot(endTangent, float32_t2(p - endPoint));

                    if (dotStart > 0 && dotEnd < 0)
                    {
                        float32_t ellipseDist = Ellipse_t::construct(majorAxisLength, eccentricity).signedDistance(p);
                        return abs(ellipseDist) - thickness;
                    }
                    else
                    {
                        float32_t sdCircle1 = Circle_t::construct(startPoint, thickness).signedDistance(p);
                        float32_t sdCircle2 = Circle_t::construct(endPoint, thickness).signedDistance(p);
                        return min(sdCircle1, sdCircle2);
                    }
                }
            };
}
}
}

#endif