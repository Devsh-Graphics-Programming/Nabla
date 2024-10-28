// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_TEXT_RENDERING_MSDF_INCLUDED_
#define _NBL_BUILTIN_HLSL_TEXT_RENDERING_MSDF_INCLUDED_

namespace nbl 
{
namespace hlsl
{
namespace text
{

float median(float r, float g, float b) {
    return max(min(r, g), min(max(r, g), b));
}

/*

Returns the distance to the shape in the given MSDF.
This can then be processed using smoothstep to provide anti-alising when rendering the shape.

@params:
- msdfSample: sampled SNORM value from the MSDF texture generated by msdfgen library.

- pixelRange = screenPxRangeValue * msdfPixelRange * 0.5, It is the value to convert snorm distance to screen space distance
    - msdfPixelRange: specifies the width of the range around the shape between the minimum and maximum representable signed distance in shape units or distance field pixels, respectivelly.
        for example if msdfPixelRange is 4, then the range of distance values are [-2, +2], and it can be computed by snormValue * MSDFPixelRange/2.0
        so an snorm value of 1.0 means a distance of 2 pixels outside the shape (in msdf texture space)
        This value is set when rendering the MSDF with MSDFgen.
    - screenPxRangeValue: the value used to convert the distance values in the msdf texture/atlas to distance in screen space. 
        In other words it's DistanceInScreenSpace/DistanceInMSDFTextureSpace, the larger the glyph (or more zoomed in) the larger this value is.
        In 2D Text Rendering it is computed by `GlyphScreenSpaceSize/GlyphTextureSpaceSize`
         where GlyphTextureSpaceSize is the size of the glyph inside the msdf texture/atlas
*/
float msdfDistance(float3 msdfSample, float pixelRange) {
    return median(msdfSample.r, msdfSample.g, msdfSample.b) * pixelRange;
}

}
}
}

#endif
