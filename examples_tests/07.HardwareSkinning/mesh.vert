// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#version 420 core
layout(binding = 3) uniform samplerBuffer tex3;

layout(location = 0) in vec3 vPos;
layout(location = 2) in vec2 vTC;
layout(location = 3) in vec3 vNormal;
//
layout(location = 5) in uvec4 vBoneIDs;
layout(location = 6) in vec4 vBoneWeights;

out vec3 Normal;
out vec2 TexCoord;
out vec3 lightDir;


void main()
{
    vec4 pos;
    vec3 nml;
    {
        const bool hasBones = nbl_glsl_scene_Node_isValidUID(vBoneIDs[0]);
        uint nodeID = hasBones ? vBoneIDs[0]:pc.nodeID;

        nbl_glsl_scene_Node_per_camera_data_t bone_camera_data = node_camera_data[(pc.cameraID,nodeID)];
        nbl_glsl_scene_Node_output_data_t bone_out_data = node_output_data[nodeID];

        nbl_glsl_scene_Node_initializeLinearSkin(pos,nml,vPos,vNormal,bone_camera_data.worldViewProj,nbl_glsl_scene_Node_output_data_t_getNormalMatrix(bone_out_data),hasBones ? vBoneWeights[0]:1.f);
        for (int i=1; i<4; i++)
        {
            nodeID = vBoneIDs[i];
            if (nbl_glsl_scene_Node_isValidUID(nodeID))
            {
                bone_camera_data = node_camera_data[(pc.cameraID,nodeID)];
                bone_out_data = node_output_data[nodeID];

                nbl_glsl_scene_Node_accumulateLinearSkin(pos,nml,vPos,vNormal,bone_camera_data.worldViewProj,nbl_glsl_scene_Node_output_data_t_getNormalMatrix(bone_out_data),vBoneWeights[i]);
            }
        }
    }

    gl_Position = pos;
    Normal = normalize(nml);
    lightDir = vec3(100.0,0.0,0.0)-vPos.xyz;
    TexCoord = vTC;
}
