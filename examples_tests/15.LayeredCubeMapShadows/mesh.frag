#version 420 core

layout(binding = 0) uniform sampler2D tex0;
layout(binding = 1) uniform samplerCube tex1; //shadow cubemap
// could do a shadow sampler, but would have to extend the engine to support hardware PCF
//these are the far and near values of the shadow camera... will just hardcode them here for now
const float near = 0.1;
const float far = 250.0;
const float farNear = far*near;
const float farNearDiff = far-near;

in vec3 Normal;
in vec2 TexCoord;
in vec3 lightDir;

layout(location = 0) out vec4 pixelColor;

const float lightStrength = 250.0;
float calcLighting(in vec3 posToLightVec, in vec3 normal)
{
    float lightDistSq = dot(posToLightVec,posToLightVec);
    float lightDist = sqrt(lightDistSq);
    return max(dot(normal,posToLightVec),0.0)*lightStrength/(lightDistSq*lightDist); //still faster than pow(lightDistSq,1.5)
}

float linearizeZBufferVal(in float nonLinearZBufferVal)
{
    //float NDC_z = 1.0-nonLinearZBufferVal;
    //return near*far/(far-NDC_z*(far-near));
    //return near*far/(near+nonLinearZBufferVal*(far-near));
    return farNear/(near+nonLinearZBufferVal*farNearDiff);
}

float chebyshevNorm(in vec3 dir)
{
    vec3 tmp = abs(dir);
    return max(max(tmp.x,tmp.y),tmp.z);
}

void main()
{
    float lightChebyshev = chebyshevNorm(lightDir);

    vec3 normal = normalize(Normal);
    float lightIntensity = calcLighting(lightDir,normal);

    if (lightIntensity>1.0/255.0) //only read from texture if could be lit
    {
        float zBufferVal = texture(tex1,-lightDir).x;
        const float epsilon = 0.0625;
        const float ramp = 8.0;
        float visibility = clamp(1.0+(linearizeZBufferVal(zBufferVal)+epsilon-lightChebyshev)*ramp,0.0,1.0);
        lightIntensity *= visibility;
    }

    pixelColor = texture(tex0,TexCoord)*vec4(lightIntensity);
}
