#version 430 core
layout(location = 0) uniform mat4 MVP;
layout(location = 1) uniform mat3 NormalMatrix;

layout(location = 0) in vec3 vPos; //only a 3d position is passed from irrlicht, but last (the W) coordinate gets filled with default 1.0
layout(location = 2) in vec2 vUV;
layout(location = 3) in vec3 vNormal;

out vec2 uv;
out vec3 Normal;

void main()
{
    gl_Position = MVP*vec4(vPos.xyz,1.0);
	uv = vUV;
    Normal = NormalMatrix*normalize(vNormal); //have to normalize twice because of normal quantization
}
