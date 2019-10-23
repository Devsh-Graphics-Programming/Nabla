#version 450 core

layout (std140, binding = 0) uniform UBO
{
    vec3 cameraPos;
	mat4 MVP;
};

layout(location = 0) in vec4 vPos; //only a 3d position is passed from irrlicht, but last (the W) coordinate gets filled with default 1.0
layout(location = 1) in vec4 vCol; //only a 3d position is passed from irrlicht, but last (the W) coordinate gets filled with default 1.0
layout(location = 3) in vec3 vNormal;

layout (location = 0) out vec4 Color; //per vertex output color, will be interpolated across the triangle
layout (location = 1) out vec3 Normal;
layout (location = 2) out vec3 lightDir;

void main()
{
    gl_Position = MVP*vPos; //only thing preventing the shader from being core-compliant
    Color = vec4(1.0);
    Normal = normalize(vNormal); //have to normalize twice because of normal quantization
    lightDir = cameraPos-vPos.xyz;
}
