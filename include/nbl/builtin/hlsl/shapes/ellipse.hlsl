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
            float msign(in float x) { return (x < 0.0) ? -1.0 : 1.0; }
            // Ellipse Shape Centered at (0,0) and Major-Axis aligned with X-Axis
            struct Ellipse_t
            {
                float majorAxisLength;
                float eccentricity;

                static Ellipse_t construct(float majorAxisLength, float eccentricity)
                {
                    Ellipse_t ret = { majorAxisLength, eccentricity };
                    return ret;
                }

                // https://iquilezles.org/articles/ellipsedist/ with modifications to add rotation and different inputs and fixed degenerate points
                // major axis is in the direction of +x and minor axis is in the direction of +y
                // @param p should be in ellipse space -> center of ellipse is (0,0)
                float signedDistance(float2 p)
                {
                    if (eccentricity == 1.0)
                        return length(p) - majorAxisLength;

                    float minorAxisLength = majorAxisLength * eccentricity;
                    float2 ab = float2(majorAxisLength, minorAxisLength);
                    p = abs(p);

                    if (p.x > p.y) { p = p.yx; ab = ab.yx; }

                    float l = ab.y * ab.y - ab.x * ab.x;

                    float m = ab.x * p.x / l;
                    float n = ab.y * p.y / l;
                    float m2 = m * m;
                    float n2 = n * n;

                    float c = (m2 + n2 - 1.0) / 3.0;
                    float c3 = c * c * c;

                    float d = c3 + m2 * n2;
                    float q = d + m2 * n2;
                    float g = m + m * n2;

                    float co;

                    if (d < 0.0)
                    {
                        float h = acos(q / c3) / 3.0;
                        float s = cos(h) + 2.0;
                        float t = sin(h) * sqrt(3.0);
                        float rx = sqrt(max(m2 - c * (s + t), 0.0));
                        float ry = sqrt(max(m2 - c * (s - t), 0.0));
                        co = ry + sign(l) * rx + abs(g) / (rx * ry);
                    }
                    else
                    {
                        float h = 2.0 * m * n * sqrt(d);
                        float s = msign(q + h) * pow(abs(q + h), 1.0 / 3.0);
                        float t = msign(q - h) * pow(abs(q - h), 1.0 / 3.0);
                        float rx = -(s + t) - c * 4.0 + 2.0 * m2;
                        float ry = (s - t) * sqrt(3.0);
                        float rm = sqrt(max(rx * rx + ry * ry, 0.0));
                        co = ry / sqrt(max(rm - rx, 0.0)) + 2.0 * g / rm;
                    }
                    co = (co - m) / 2.0;

                    float si = sqrt(max(1.0 - co * co, 0.0));

                    float2 r = ab * float2(co, si);

                    return length(r - p) * msign(p.y - r.y);
                }
            };

            // Ellipse Outline
            struct EllipseOutline_t
            {
                float2 center;
                float2 majorAxis;
                float eccentricity;
                float thickness;

                static EllipseOutline_t construct(float2 center, float2 majorAxis, float eccentricity, float thickness)
                {
                    EllipseOutline_t ret = { center, majorAxis, eccentricity, thickness };
                    return ret;
                }

                float signedDistance(float2 p)
                {
                    float majorAxisLength = length(majorAxis);
                    float2 dir = majorAxis / majorAxisLength;
                    p = p - center;
                    p = mul(float2x2(dir.x, dir.y, -dir.y, dir.x), p);

                    float ellipseDist = Ellipse_t::construct(majorAxisLength, eccentricity).signedDistance(p);
                    return abs(ellipseDist) - thickness;
                }
            };

            // Ellipse Outline
            struct EllipseOutlineBounded_t
            {
                float2 center;
                float2 majorAxis;
                float2 bounds;
                float eccentricity;
                float thickness;

                // @param bounds is in [0, 2PI]
                //      bounds.y-bounds.x should be <= PI
                static EllipseOutlineBounded_t construct(float2 center, float2 majorAxis, float2 bounds, float eccentricity, float thickness)
                {
                    EllipseOutlineBounded_t ret = { center, majorAxis, bounds, eccentricity, thickness };
                    return ret;
                }

                float signedDistance(float2 p)
                {
                    float majorAxisLength = length(majorAxis);
                    float2 dir = majorAxis / majorAxisLength;
                    p = p - center;
                    p = mul(float2x2(dir.x, dir.y, -dir.y, dir.x), p);

                    float2 pNormalized = normalize(p);
                    float theta = atan2(pNormalized.y, -pNormalized.x * eccentricity) + nbl_hlsl_PI;

                    float minorAxisLength = majorAxisLength * eccentricity;

                    float2 startPoint = float2(majorAxisLength * cos(bounds.x), -minorAxisLength * sin(bounds.x));
                    float2 endPoint = float2(majorAxisLength * cos(bounds.y), -minorAxisLength * sin(bounds.y));

                    float2 startTangent = float2(-majorAxisLength * sin(bounds.x), -minorAxisLength * cos(bounds.x));
                    float2 endTangent = float2(-majorAxisLength * sin(bounds.y), -minorAxisLength * cos(bounds.y));
                    float dotStart = dot(startTangent, float2(p - startPoint));
                    float dotEnd = dot(endTangent, float2(p - endPoint));

                    if (dotStart > 0 && dotEnd < 0)
                    {
                        float ellipseDist = Ellipse_t::construct(majorAxisLength, eccentricity).signedDistance(p);
                        return abs(ellipseDist) - thickness;
                    }
                    else
                    {
                        float sdCircle1 = Circle_t::construct(startPoint, thickness).signedDistance(p);
                        float sdCircle2 = Circle_t::construct(endPoint, thickness).signedDistance(p);
                        return min(sdCircle1, sdCircle2);
                    }
                }
            };
}
}
}

#endif