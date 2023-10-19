
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_BXDF_NDF_BECKMANN_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_NDF_BECKMANN_INCLUDED_

#include <nbl/builtin/hlsl/math/constants.hlsl>
#include <nbl/builtin/hlsl/math/functions.hlsl>
#include <nbl/builtin/hlsl/bxdf/common.hlsl>
#include <nbl/builtin/hlsl/bxdf/ndf.hlsl>


namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace ndf
{


float beckmann(in float a2, in float NdotH2)
{
    float nom = exp( (NdotH2-1.0)/(a2*NdotH2) ); // exp(x) == exp2(x/log(2)) ?
    float denom = a2*NdotH2*NdotH2;

    return math::RECIPROCAL_PI * nom/denom;
}

float beckmann(in float ax, in float ay, in float ax2, in float ay2, in float TdotH2, in float BdotH2, in float NdotH2)
{
    float nom = exp(-(TdotH2/ax2+BdotH2/ay2)/NdotH2);
    float denom = ax * ay * NdotH2 * NdotH2;

    return math::RECIPROCAL_PI * nom / denom;
}


struct IsotropicBeckmann : NDFBase<>
{
    float a;
    float a2;

    static IsotropicBeckmann create(float _a)
    {
        IsotropicBeckmann b;
        b.a = _a;
        b.a2 = _a*_a;
        return b;
    }
    static IsotropicBeckmann create(float _a, float _a2)
    {
        IsotropicBeckmann b;
        b.a  = _a;
        b.a2 = _a2;
        return b;
    }


    static float C2(in float NdotX2, in float _a2)
    {
        return NdotX2 / (_a2 * (1.0 - NdotX2));
    }
    static float C2(in float TdotX2, in float BdotX2, in float NdotX2, in float _ax2, in float _ay2)
    {
        return NdotX2 / (TdotX2 * _ax2 + BdotX2 * _ay2);
    }

    template <class IncomingRayDirInfo>
    static float3 generateH_impl(in surface_interactions::Anisotropic<IncomingRayDirInfo> interaction, inout float3 u, out AnisotropicMicrofacetCache cache, in float _ax, in float _ay)
    {
        const float3 localV = interaction.getTangentSpaceV();

        //stretch
        float3 V = normalize(float3(_ax * localV.x, _ay * localV.y, localV.z));

        float2 slope;
        if (V.z > 0.9999)//V.z=NdotV=cosTheta in tangent space
        {
            float r = sqrt(-log(1.0 - u.x));
            float sinPhi = sin(2.0 * math::PI * u.y);
            float cosPhi = cos(2.0 * math::PI * u.y);
            slope = float2(r, r) * float2(cosPhi, sinPhi);
        }
        else
        {
            float cosTheta = V.z;
            float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
            float tanTheta = sinTheta / cosTheta;
            float cotTheta = 1.0 / tanTheta;

            float a = -1.0;
            float c = math::erf(cosTheta);
            float sample_x = max(u.x, 1.0e-6f);
            float theta = acos(cosTheta);
            float fit = 1.0 + theta * (-0.876 + theta * (0.4265 - 0.0594 * theta));
            float b = c - (1.0 + c) * pow(1.0 - sample_x, fit);

            float normalization = 1.0 / (1.0 + c + math::SQRT_RECIPROCAL_PI * tanTheta * exp(-cosTheta * cosTheta));

            const int ITER_THRESHOLD = 10;
            const float MAX_ACCEPTABLE_ERR = 1.0e-5;
            int it = 0;
            float value = 1000.0;
            while (++it<ITER_THRESHOLD && abs(value)>MAX_ACCEPTABLE_ERR)
            {
                if (!(b >= a && b <= c))
                    b = 0.5 * (a + c);

                float invErf = math::erfInv(b);
                value = normalization * (1.0 + b + math::SQRT_RECIPROCAL_PI * tanTheta * exp(-invErf * invErf)) - sample_x;
                float derivative = normalization * (1.0 - invErf * cosTheta);

                if (value > 0.0)
                    c = b;
                else
                    a = b;

                b -= value / derivative;
            }
            // TODO: investigate if we can replace these two erf^-1 calls with a box muller transform
            slope.x = math::erfInv(b);
            slope.y = math::erfInv(2.0f * max(u.y, 1.0e-6f) - 1.0f);
        }

        float sinTheta = sqrt(1.0f - V.z * V.z);
        float cosPhi = sinTheta == 0.0f ? 1.0f : clamp(V.x / sinTheta, -1.0f, 1.0f);
        float sinPhi = sinTheta == 0.0f ? 0.0f : clamp(V.y / sinTheta, -1.0f, 1.0f);
        //rotate
        float tmp = cosPhi * slope.x - sinPhi * slope.y;
        slope.y = sinPhi * slope.x + cosPhi * slope.y;
        slope.x = tmp;

        //unstretch
        slope = float2(_ax, _ay) * slope;

        const float3 localH = normalize(float3(-slope, 1.0));

        cache = AnisotropicMicrofacetCache::create(localV, localH);

        return localH;
    }

    template <typename IncomingRayDirInfo>
    float3 generateH(in surface_interactions::Anisotropic<IncomingRayDirInfo> interaction, inout float3 u, out AnisotropicMicrofacetCache cache)
    {
        return generateH_impl(interaction, u, cache, a, a);
    }

    scalar_t D(in float NdotH2)
    {
        float nom = exp((NdotH2 - 1.0) / (a2 * NdotH2)); // exp(x) == exp2(x/log(2)) ?
        float denom = a2 * NdotH2 * NdotH2;

        return math::RECIPROCAL_PI * nom / denom;
    }

    scalar_t Lambda_impl(in float c2)
    {
        float c = sqrt(c2);
        float nom = 1.0 - 1.259 * c + 0.396 * c2;
        float denom = 2.181 * c2 + 3.535 * c;
        return lerp(0.0, nom / denom, c < 1.6);
    }
    scalar_t Lambda(in float NdotX2)
    {
        return Lambda_impl(C2(NdotX2, a2));
    }

    // TODO what about aniso variants of D and Lambda?
    // return nan?
    // since we dont have SFINAE in HLSL, they must be defined for ndf_traits to compile with the type
};

struct Beckmann : IsotropicBeckmann
{
    static Beckmann create(float _ax, float _ay, float _ax2, float _ay2)
    {
        Beckmann b;
        b.a = _ax;
        b.ay = _ay;
        b.a2 = _ax2;
        b.ay2 = _ay2;
        return b;
    }
    static Beckmann create(float _ax, float _ay)
    {
        return create(_ax, _ay, _ax*_ax, _ay*_ay);
    }


    static float C2(in float TdotX2, in float BdotX2, in float NdotX2, in float _ax2, in float _ay2)
    {
        return NdotX2 / (TdotX2 * _ax2 + BdotX2 * _ay2);
    }


    template <typename IncomingRayDirInfo>
    float3 generateH(in surface_interactions::Anisotropic<IncomingRayDirInfo> interaction, inout float3 u, out AnisotropicMicrofacetCache cache)
    {
        return generateH_impl(interaction, u, cache, a, ay);
    }


    scalar_t D(in float TdotH2, in float BdotH2, in float NdotH2)
    {
        float nom = exp(-(TdotH2 / a2 + BdotH2 / ay2) / NdotH2);
        float denom = a * ay * NdotH2 * NdotH2;

        return math::RECIPROCAL_PI * nom / denom;
    }

    float Lambda(in float TdotX2, in float BdotX2, in float NdotX2)
    {
        return Lambda_impl(C2(TdotX2, BdotX2, NdotX2, a2, ay2));
    }

    //float ax; // inherited from base as `a`
    //float ax2; // inherited from base as `a2`
    float ay;
    float ay2;
};

}
}	
}
}

#endif