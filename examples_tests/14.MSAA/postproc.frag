#version 330 core
uniform sampler2DMS tex0; //color
uniform sampler2DMS tex1; //depth

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

