#version 330 compatibility
uniform vec3 cameraPos;
uniform mat4 MVP;

layout(location = 0) in vec4 vPos; //only a 3d position is passed from irrlicht, but last (the W) coordinate gets filled with default 1.0
layout(location = 1) in vec4 vCol; //only a 3d position is passed from irrlicht, but last (the W) coordinate gets filled with default 1.0
layout(location = 3) in vec3 vNormal;

out vec4 Color; //per vertex output color, will be interpolated across the triangle
out vec3 Normal;
out vec3 lightDir;

void main()
{
    gl_Position = MVP*vPos; //only thing preventing the shader from being core-compliant
    Color = vCol;
    Normal = normalize(vNormal); //have to normalize twice because of normal quantization
    lightDir = cameraPos-vPos.xyz;
}
