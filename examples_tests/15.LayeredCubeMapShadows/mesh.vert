#version 420 core
uniform vec3 worldSpaceLightPos;
uniform mat4 MVP;
uniform mat4x3 worldMat;
uniform mat3 normalMat;

layout(location = 0) in vec3 vPos; //only a 3d position is passed from irrlicht, but last (the W) coordinate gets filled with default 1.0
layout(location = 2) in vec2 vTC;
layout(location = 3) in vec3 vNormal;

out vec3 Normal;
out vec2 TexCoord;
out vec3 lightDir;

void main()
{
    gl_Position = MVP*vec4(vPos,1.0);
    Normal = normalMat*normalize(vNormal); //have to normalize twice because of normal quantization
    lightDir = worldSpaceLightPos-worldMat*vec4(vPos,1.0);
    TexCoord = vTC;
}
