// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _IRR_BXDF_GEOM_SMITH_BECKMANN_INCLUDED_
#define _IRR_BXDF_GEOM_SMITH_BECKMANN_INCLUDED_

float irr_glsl_smith_beckmann_C2(in float NdotX2, in float a2)
{
    return NdotX2 / (a2 * (1.0 - NdotX2));
}
float irr_glsl_smith_beckmann_C2(in float TdotX2, in float BdotX2, in float NdotX2, in float ax2, in float ay2)
{
    return NdotX2/(TdotX2*ax2+BdotX2*ay2);
}
//G1 = 1/(1+_Lambda)
float irr_glsl_smith_beckmann_Lambda(in float c2)
{
    float c = sqrt(c2);
    float nom = 1.0 - 1.259*c + 0.396*c2;
    float denom = 2.181*c2 + 3.535*c;
    return mix(0.0, nom/denom, c<1.6);
}
float irr_glsl_smith_beckmann_Lambda(in float NdotX2, in float a2)
{
    return irr_glsl_smith_beckmann_Lambda(irr_glsl_smith_beckmann_C2(NdotX2, a2));
}

float irr_glsl_beckmann_smith_correlated(in float NdotV2, in float NdotL2, in float a2)
{
    float c2 = irr_glsl_smith_beckmann_C2(NdotV2, a2);
    float L_v = irr_glsl_smith_beckmann_Lambda(c2);
    c2 = irr_glsl_smith_beckmann_C2(NdotL2, a2);
    float L_l = irr_glsl_smith_beckmann_Lambda(c2);
    return 1.0 / (1.0 + L_v + L_l);
}
float irr_glsl_beckmann_smith_correlated(in float TdotV2, in float BdotV2, in float NdotV2, in float TdotL2, in float BdotL2, in float NdotL2, in float ax2, in float ay2)
{
    float c2 = irr_glsl_smith_beckmann_C2(TdotV2, BdotV2, NdotV2, ax2, ay2);
    float L_v = irr_glsl_smith_beckmann_Lambda(c2);
    c2 = irr_glsl_smith_beckmann_C2(TdotL2, BdotL2, NdotL2, ax2, ay2);
    float L_l = irr_glsl_smith_beckmann_Lambda(c2);
    return 1.0 / (1.0 + L_v + L_l);
}

float irr_glsl_beckmann_smith_G2_over_G1(in float lambdaV_plus_one, in float NdotL2, in float a2)
{
    float lambdaL = irr_glsl_smith_beckmann_Lambda(NdotL2, a2);

    return lambdaV_plus_one / (lambdaV_plus_one+lambdaL);
}
float irr_glsl_beckmann_smith_G2_over_G1(in float lambdaV_plus_one, in float TdotL2, in float BdotL2, in float NdotL2, in float ax2, in float ay2)
{
    float c2 = irr_glsl_smith_beckmann_C2(TdotL2, BdotL2, NdotL2, ax2, ay2);
	float lambdaL = irr_glsl_smith_beckmann_Lambda(c2);

    return lambdaV_plus_one / (lambdaV_plus_one + lambdaL);
}

#endif
