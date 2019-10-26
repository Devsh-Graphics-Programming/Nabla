#version 430 core
layout(location = 3) uniform mat4x3 W;
layout(location = 0) uniform mat4 MVP;

layout(location = 0) in vec3 vPos; //only a 3d position is passed from irrlicht, but last (the W) coordinate gets filled with default 1.0
layout(location = 2) in vec2 vUV;
layout(location = 3) in vec3 vNormal;

out vec2 uv;
out vec3 Normal;

void main()
{
    gl_Position = MVP*vec4(W[0]*vPos.x+W[1]*vPos.y+W[2]*vPos.z+W[3],1.0);
	uv = vUV;
    Normal = normalize(vNormal); //have to normalize twice because of normal quantization
}
