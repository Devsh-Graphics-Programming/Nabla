#version 430 core

#extension ARB_shader_draw_parameters : require

struct ModelData_t
{
    mat4 MVP;
    mat3 normalMat;
};

layout(std430, row_major, binding = 0) buffer PerObject
{
    ModelData_t modelData[];
};


layout(location = 0 ) in vec3 vPos; //only a 3d position is passed from irrlicht, but last (the W) coordinate gets filled with default 1.0
layout(location = 3 ) in vec3 vNormal;

out vec4 Color; //per vertex output color, will be interpolated across the triangle
flat out vec3 Normal;

void main()
{
    uint drawID = gl_DrawIDARB;

    gl_Position = modelData[drawID].MVP[0]*vPos.x+modelData[drawID].MVP[1]*vPos.y+modelData[drawID].MVP[2]*vPos.z+modelData[drawID].MVP[3];
    Color = vec4(0.4,0.4,1.0,1.0);
    Normal = normalize(modelData[drawID].normalMat*vNormal); //have to normalize twice because of normal quantization
}

