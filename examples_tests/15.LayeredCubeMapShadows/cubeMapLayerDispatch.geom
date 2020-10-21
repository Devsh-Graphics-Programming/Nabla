// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#version 400 core
layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;
layout (invocations = 6) in; //use fixed function tessellation stage to amplify geometry

uniform mat4 ViewProjCubeMatrices[6];

void main()
{
    gl_Layer = gl_InvocationID;

    mat3x4 clipPos;
    for (int i=0; i<3; i++)
        clipPos[i] = ViewProjCubeMatrices[gl_InvocationID]*vec4(gl_in[i].gl_Position.xyz,1.0);

    // can comment this clip-space culling in and out, but my guess is that the rasterizer pipeline in most GPUs does this already!
    mat4x3 cullingMatrix = transpose(clipPos);
    for (int i=0; i<2; i++)
    {
        if (all(greaterThan(cullingMatrix[i],cullingMatrix[3])) || all(lessThan(cullingMatrix[i],-cullingMatrix[3])))
            return;
    }
	// realy optional stuff if you already have good CPU culling
	// I'm working in [0,1] or [1,0] depth range
    if (all(greaterThan(cullingMatrix[2],cullingMatrix[3])) || all(lessThan(cullingMatrix[2],vec3(0.0))))
		return;

    // emit
    for (int i=0; i<3; i++)
    {
        gl_Position = clipPos[i];
        EmitVertex();
    }

    EndPrimitive();
}

/* NAH
#version 330 core
layout (triangles) in;
layout (triangle_strip, max_vertices = 18) out;

uniform mat4 ViewProjCubeMatrices[6];

void main()
{
    for (int i=0; i<6; i++)
    {
        gl_Layer = i;

        vec4 clipPos[3];
        for (int j=0; j<3; j++)
            clipPos[j] = ViewProjCubeMatrices[i]*vec4(gl_in[j].gl_Position.xyz,1.0);

        if (clipPos[0].x>clipPos[0].w&&clipPos[1].x>clipPos[1].w&&clipPos[2].x>clipPos[2].w)
            continue;
        if (clipPos[0].y>clipPos[0].w&&clipPos[1].y>clipPos[1].w&&clipPos[2].y>clipPos[2].w)
            continue;
        if (clipPos[0].z>clipPos[0].w&&clipPos[1].z>clipPos[1].w&&clipPos[2].z>clipPos[2].w)
            continue;
        if (clipPos[0].x<-clipPos[0].w&&clipPos[1].x<-clipPos[1].w&&clipPos[2].x<-clipPos[2].w)
            continue;
        if (clipPos[0].y<-clipPos[0].w&&clipPos[1].y<-clipPos[1].w&&clipPos[2].y<-clipPos[2].w)
            continue;
        if (clipPos[0].z<-clipPos[0].w&&clipPos[1].z<-clipPos[1].w&&clipPos[2].z<-clipPos[2].w)
            continue;

        for (int j=0; j<3; j++)
        {
            gl_Position = clipPos[j];
            EmitVertex();
        }

        EndPrimitive();
    }
}
*/
