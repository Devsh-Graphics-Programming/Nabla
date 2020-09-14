#ifndef _IRR_BXDF_GEOM_SMITH_COMMON_INCLUDED_
#define _IRR_BXDF_GEOM_SMITH_COMMON_INCLUDED_

float irr_glsl_smith_G1(in float lambda)
{
    return 1.0 / (1.0 + lambda);
}

#endif
