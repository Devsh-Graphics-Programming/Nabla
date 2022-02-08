// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_BARYCENTRIC_FRAG_INCLUDED_
#define _NBL_BUILTIN_GLSL_BARYCENTRIC_FRAG_INCLUDED_



// forward declaration is useful for offline compilers to not moan until the end
vec2 nbl_glsl_barycentric_frag_get();



// TODO: Check for Nvidia Pascal Barycentric extension and AMD Barycentric extension when SPIRV-Cross supports it and make Nabla provide NBL_GL_ cap macros
#ifdef NBL_GL_NV_fragment_shader_barycentric


vec2 nbl_glsl_barycentric_frag_get()
{
    return gl_BaryCoordNV.xy;
}


#else


#ifndef NBL_GLSL_BARYCENTRIC_FRAG_POS_INPUT
#define NBL_GLSL_BARYCENTRIC_FRAG_POS_INPUT
    #ifndef NBL_GLSL_BARYCENTRIC_FRAG_POS_INPUT_LOC
    #define NBL_GLSL_BARYCENTRIC_FRAG_POS_INPUT_LOC 0
    #endif

    #ifdef NBL_GL_AMD_shader_explicit_vertex_parameter
        #define INTERPOLATION_ATTRIBUTES __explicitInterpAMD
    #else
        #define INTERPOLATION_ATTRIBUTES smooth
    #endif
    layout(location = NBL_GLSL_BARYCENTRIC_FRAG_POS_INPUT_LOC) INTERPOLATION_ATTRIBUTES in vec3 nbl_glsl_barycentric_frag_pos;
    #undef INTERPOLATION_ATTRIBUTES
#endif

#ifndef NBL_GLSL_BARYCENTRIC_FRAG_PROVOKINGPOS_INPUT
#define NBL_GLSL_BARYCENTRIC_FRAG_PROVOKINGPOS_INPUT
    #ifndef NBL_GLSL_BARYCENTRIC_FRAG_PROVOKINGPOS_INPUT_LOC
    #define NBL_GLSL_BARYCENTRIC_FRAG_PROVOKINGPOS_INPUT_LOC 1
    #endif

    layout(location = NBL_GLSL_BARYCENTRIC_FRAG_PROVOKINGPOS_INPUT_LOC) flat in vec3 nbl_glsl_barycentric_frag_provokingPos;
#endif


#ifndef _NBL_BUILTIN_GLSL_BARYCENTRIC_FRAG_GET_DEFINED_
    #ifdef NBL_GL_AMD_shader_explicit_vertex_parameter
        vec2 nbl_glsl_barycentric_frag_get()
        {
            if (nbl_glsl_barycentric_frag_provokingPos==interpolateAtVertexAMD(nbl_glsl_barycentric_frag_pos,1))
                return gl_BaryCoordSmoothAMD.xy;
            
            const float lastBaryCoord = 1.f-gl_BaryCoordSmoothAMD.x-gl_BaryCoordSmoothAMD.y;
            const bool lastChosenAsAMDProvoking = nbl_glsl_barycentric_frag_provokingPos==interpolateAtVertexAMD(nbl_glsl_barycentric_frag_pos,0);
            if (lastChosenAsAMDProvoking) // are these even the right way round?
                return vec2(lastBaryCoord,gl_BaryCoordSmoothAMD.x);
            else
                return vec2(gl_BaryCoordSmoothAMD.y,lastBaryCoord);
        }
    #else
        // these will need to be defined by the user
        uint nbl_glsl_barycentric_frag_getDrawID();
        vec3 nbl_glsl_barycentric_frag_getVertexPos(in uint drawID, in uint primID, in uint primsVx);

        #include <nbl/builtin/glsl/barycentric/utils.glsl>
        vec2 nbl_glsl_barycentric_frag_get()
        {
            return nbl_glsl_barycentric_reconstructBarycentrics(nbl_glsl_barycentric_frag_pos,mat3(
                nbl_glsl_barycentric_frag_provokingPos,
                nbl_glsl_barycentric_frag_getVertexPos(nbl_glsl_barycentric_frag_getDrawID(),gl_PrimitiveID,1u),
                nbl_glsl_barycentric_frag_getVertexPos(nbl_glsl_barycentric_frag_getDrawID(),gl_PrimitiveID,2u)
            ));
        }
    #endif
#define _NBL_BUILTIN_GLSL_BARYCENTRIC_FRAG_GET_DEFINED_
#endif


#endif



#endif