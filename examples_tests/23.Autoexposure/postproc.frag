#version 420 core
uniform sampler2D tex0;

layout(binding = 0, std140) uniform PerFrameBlock
{
    vec4 dynamicResolutionScale;
    vec4 autoExposureInput;
	vec4 autoExposureParameters; //(exponentTerm,divisorTerm,avgLuma)
};


in vec2 TexCoord;


#define kLumaConvertCoeff vec3(0.299, 0.587, 0.114)

//! Future users.. DONT USE THIS, GO FOR ACES tonemapper instead
vec3 toneMapExp(in vec3 inCol)
{
    float luma = dot(inCol,kLumaConvertCoeff);
    if (luma<=0.0) return vec3(0.0);

    return inCol*(1.0-exp2(autoExposureParameters.x*luma))/(autoExposureParameters.y*luma);
    ///@TODO do some rescale to fit greatest channel so no hue shift
}


layout(location = 0) out vec4 pixelColor;

void main()
{
    vec3 hdrVal = textureLod(tex0,TexCoord*dynamicResolutionScale.xy,0.0).rgb;

    pixelColor = vec4(toneMapExp(hdrVal),1.0);
}

