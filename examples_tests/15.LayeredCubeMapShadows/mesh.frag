#version 450 core

layout (location = 0) uniform float derivScaleFactor = 1000.0/10.0;
layout(binding = 0) uniform sampler2D tex0;
layout(binding = 1) uniform samplerCube tex1; //shadow cubemap
layout(binding = 4) uniform sampler2D derivativeMap;
// could do a shadow sampler, but would have to extend the engine to support hardware PCF
//these are the far and near values of the shadow camera... will just hardcode them here for now
const float near = 0.1;
const float far = 250.0;
const float farNear = far*near;
const float farNearDiff = far-near;

in vec3 Position;
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



vec3 calculateSurfaceGradient(in vec3 normal, in vec3 dpdx, in vec3 dpdy, in float dhdx, in float dhdy)
{
    vec3 r1 = cross(dpdy, normal);
    vec3 r2 = cross(normal, dpdx);
 
    return (r1*dhdx + r2*dhdy) / dot(dpdx, r1);
}

vec3 perturbNormal(in vec3 normal, in vec3 dpdx, in vec3 dpdy, in float dhdx, in float dhdy)
{
    return normalize(normal - calculateSurfaceGradient(normal, dpdx, dpdy, dhdx, dhdy));
}

float applyChainRule(in vec2 h_gradient, in vec2 dUVd_)
{
    return dot(h_gradient, dUVd_);
}

// Calculate the surface normal using the uv-space gradient (dhdu, dhdv)
vec3 calculateSurfaceNormal(in vec3 position, in vec2 uv, in vec3 normal, in vec2 h_gradient)
{
    vec3 dpdx = dFdx(position);
    vec3 dpdy = dFdy(position);
 
	vec2 dUVdx = dFdx(uv);
	vec2 dUVdy = dFdy(uv);
 
    float dhdx = applyChainRule(h_gradient, dUVdx);
    float dhdy = applyChainRule(h_gradient, dUVdy);
 
    return perturbNormal(normal, dpdx, dpdy, dhdx, dhdy);
}

void main()
{
    float lightChebyshev = chebyshevNorm(lightDir);

	vec2 h_gradient = texture(derivativeMap, TexCoord).xy * derivScaleFactor.xx;
	
    vec3 normal = normalize(Normal);
	normal = calculateSurfaceNormal(Position, TexCoord, normal, h_gradient);
	
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
