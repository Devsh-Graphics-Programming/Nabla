// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#version 420 core
layout(binding = 0) uniform sampler2DMS tex0; //color
layout(binding = 1) uniform sampler2DMS tex1; //depth

//! Unfortunately there is no textureSampleCount() like textureSize()
uniform int sampleCount;

in vec2 TexCoord;

layout(location = 0) out vec4 pixelColor;

/** could do funky MSAA based SSAO with the depth,
or some FXAA/MLAA thing that uses multi-sample depth fur teh lulz**/
void main()
{
    ivec2 integerTexCoord = ivec2(TexCoord*textureSize(tex0));

    //! or could do a fancier resolve
    vec4 outColor = texelFetch(tex0,integerTexCoord,0);
    for (int i=1; i<sampleCount; i++)
    {
        outColor += texelFetch(tex0,integerTexCoord,i);
    }
    pixelColor = outColor/float(sampleCount);
}

