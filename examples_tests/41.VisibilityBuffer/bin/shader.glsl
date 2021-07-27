#version 430 core

	#define NBL_IMPL_GL_ARB_shading_language_include
	#define NBL_IMPL_GL_ARB_enhanced_layouts
	#define NBL_IMPL_GL_ARB_shader_draw_parameters
	#define NBL_IMPL_GL_ARB_shader_group_vote
	#define NBL_IMPL_GL_ARB_cull_distance
	#define NBL_IMPL_GL_ARB_derivative_control
	#define NBL_IMPL_GL_ARB_shader_texture_image_samples
	#define NBL_IMPL_GL_KHR_blend_equation_advanced
	#define NBL_IMPL_GL_KHR_blend_equation_advanced_coherent
	#define NBL_IMPL_GL_ARB_fragment_shader_interlock
	#define NBL_IMPL_GL_ARB_post_depth_coverage
	#define NBL_IMPL_GL_ARB_shader_ballot
	#define NBL_IMPL_GL_ARB_shader_clock
	#define NBL_IMPL_GL_ARB_shader_viewport_layer_array
	#define NBL_IMPL_GL_ARB_gl_spirv
	#define NBL_IMPL_GL_ARB_spirv_extensions
	#define NBL_IMPL_GL_EXT_shader_integer_mix
	#define NBL_IMPL_GL_EXT_shader_image_load_formatted
	#define NBL_IMPL_GL_EXT_post_depth_coverage
	#define NBL_IMPL_GL_NBL_RUNNING_IN_RENDERDOC

#ifdef NBL_IMPL_GL_AMD_gpu_shader_half_float
#define NBL_GL_EXT_shader_explicit_arithmetic_types_float16
#endif

#ifdef NBL_IMPL_GL_NV_gpu_shader5
#define NBL_GL_EXT_shader_explicit_arithmetic_types_float16
#define NBL_GL_EXT_nonuniform_qualifier
#define NBL_GL_KHR_shader_subgroup_vote_subgroup_any_all_equal_bool
#endif

#ifdef NBL_IMPL_GL_AMD_gpu_shader_int16
#define NBL_GL_EXT_shader_explicit_arithmetic_types_int16
#endif

#ifdef NBL_IMPL_GL_NV_shader_thread_group
#define NBL_GL_KHR_shader_subgroup_ballot_subgroup_mask
#define NBL_GL_KHR_shader_subgroup_basic_subgroup_size
#define NBL_GL_KHR_shader_subgroup_basic_subgroup_invocation_id
#define NBL_GL_KHR_shader_subgroup_ballot_subgroup_ballot
#define NBL_GL_KHR_shader_subgroup_ballot_inverse_ballot_bit_count
#endif

#if defined(NBL_IMPL_GL_ARB_shader_ballot) && defined(NBL_IMPL_GL_ARB_shader_int64)
#define NBL_GL_KHR_shader_subgroup_ballot_subgroup_mask
#define NBL_GL_KHR_shader_subgroup_basic_subgroup_size
#define NBL_GL_KHR_shader_subgroup_basic_subgroup_invocation_id
#define NBL_GL_KHR_shader_subgroup_ballot_subgroup_broadcast_first
#define NBL_GL_KHR_shader_subgroup_ballot_subgroup_ballot
#define NBL_GL_KHR_shader_subgroup_ballot_inverse_ballot_bit_count
#endif

#if defined(NBL_IMPL_GL_AMD_gcn_shader) && (defined(NBL_IMPL_GL_AMD_gpu_shader_int64) || defined(NBL_IMPL_GL_NV_gpu_shader5))
#define NBL_GL_KHR_shader_subgroup_basic_subgroup_size
#define NBL_GL_KHR_shader_subgroup_vote_subgroup_any_all_equal_bool
#endif

#ifdef NBL_IMPL_GL_NV_shader_thread_shuffle
#define NBL_GL_KHR_shader_subgroup_ballot_subgroup_broadcast_first
#endif

#ifdef NBL_IMPL_GL_ARB_shader_group_vote
#define NBL_GL_KHR_shader_subgroup_vote_subgroup_any_all_equal_bool
#endif

#if defined(NBL_GL_KHR_shader_subgroup_ballot_subgroup_broadcast_first) && defined(NBL_GL_KHR_shader_subgroup_vote_subgroup_any_all_equal_bool)
#define NBL_GL_KHR_shader_subgroup_vote_subgroup_all_equal_T
#endif

#if defined(NBL_GL_KHR_shader_subgroup_ballot_subgroup_ballot) && defined(NBL_GL_KHR_shader_subgroup_basic_subgroup_invocation_id)
#define NBL_GL_KHR_shader_subgroup_basic_subgroup_elect
#endif

#ifdef NBL_GL_KHR_shader_subgroup_ballot_subgroup_mask
#define NBL_GL_KHR_shader_subgroup_ballot_inverse_ballot
#define NBL_GL_KHR_shader_subgroup_ballot_inclusive_bit_count
#define NBL_GL_KHR_shader_subgroup_ballot_exclusive_bit_count
#endif

#ifdef NBL_GL_KHR_shader_subgroup_ballot_subgroup_ballot
#define NBL_GL_KHR_shader_subgroup_ballot_bit_count
#endif

// the natural extensions TODO: @Crisspl implement support for https://www.khronos.org/registry/OpenGL/extensions/KHR/KHR_shader_subgroup.txt
#ifdef NBL_IMPL_GL_KHR_shader_subgroup_basic
#define NBL_GL_KHR_shader_subgroup_basic
#define NBL_GL_KHR_shader_subgroup_basic_subgroup_size
#define NBL_GL_KHR_shader_subgroup_basic_subgroup_invocation_id
#define NBL_GL_KHR_shader_subgroup_basic_subgroup_elect
#endif

#ifdef NBL_IMPL_GL_KHR_shader_subgroup_vote
#define NBL_GL_KHR_shader_subgroup_vote
#define NBL_GL_KHR_shader_subgroup_vote_subgroup_any_all_equal_bool
#define NBL_GL_KHR_shader_subgroup_vote_subgroup_all_equal_T
#endif

#ifdef NBL_IMPL_GL_KHR_shader_subgroup_ballot
#define NBL_GL_KHR_shader_subgroup_ballot
#define NBL_GL_KHR_shader_subgroup_ballot_bit_count
#define NBL_GL_KHR_shader_subgroup_ballot_subgroup_mask
#define NBL_GL_KHR_shader_subgroup_ballot_subgroup_ballot
#define NBL_GL_KHR_shader_subgroup_ballot_inclusive_bit_count
#define NBL_GL_KHR_shader_subgroup_ballot_exclusive_bit_count
#define NBL_GL_KHR_shader_subgroup_ballot_inverse_ballot_bit_count
#define NBL_GL_KHR_shader_subgroup_ballot_subgroup_broadcast_first
#endif

// TODO: do a SPIR-V Cross contribution to do all the fallbacks (later)
#ifdef NBL_IMPL_GL_KHR_shader_subgroup_shuffle
#define NBL_GL_KHR_shader_subgroup_shuffle
#endif

#ifdef NBL_IMPL_GL_KHR_shader_subgroup_shuffle_relative
#define NBL_GL_KHR_shader_subgroup_shuffle_relative
#endif

#ifdef NBL_IMPL_GL_KHR_shader_subgroup_arithmetic
#define NBL_GL_KHR_shader_subgroup_arithmetic
#endif

#ifdef NBL_IMPL_GL_KHR_shader_subgroup_clustered
#define NBL_GL_KHR_shader_subgroup_clustered
#endif

#ifdef NBL_IMPL_GL_KHR_shader_subgroup_quad
#define NBL_GL_KHR_shader_subgroup_quad
#endif
#line 2
#extension GL_EXT_shader_16bit_storage : require

#include "rasterizationCommon.h"
layout(local_size_x = WORKGROUP_SIZE) in;

layout(set=0, binding=0, std430, row_major) restrict readonly buffer PerInstanceCull
{
    CullData_t cullData[];
};
layout(set=0, binding=1, std430) restrict buffer MVPs
{
    mat4 mvps[];
} mvpBuff;

layout(set=0, binding=2, std430, column_major) restrict buffer CubeMVPs
{
    mat4 cubeMVPs[];
} cubeMvpBuff;

#define ENABLE_CUBE_COMMAND_BUFFER
#define ENABLE_FRUSTUM_CULLED_COMMAND_BUFFER
#define ENABLE_OCCLUSION_CULLED_COMMAND_BUFFER
#define ENABLE_CUBE_DRAW_GUID_BUFFER
#define ENABLE_OCCLUSION_DISPATCH_INDIRECT_BUFFER
#define ENABLE_VISIBLE_BUFFER

#define CUBE_COMMAND_BUFF_SET 0
#define CUBE_COMMAND_BUFF_BINDING 3
#define FRUSTUM_CULLED_COMMAND_BUFF_SET 0
#define FRUSTUM_CULLED_COMMAND_BUFF_BINDING 4
#define OCCLUSION_CULLED_COMMAND_BUFF_SET 0
#define OCCLUSION_CULLED_COMMAND_BUFF_BINDING 5
#define CUBE_DRAW_GUID_BUFF_SET 0
#define CUBE_DRAW_GUID_BUFF_BINDING 6
#define OCCLUSION_DISPATCH_INDIRECT_BUFF_SET 0
#define OCCLUSION_DISPATCH_INDIRECT_BUFF_BINDING 7
#define VISIBLE_BUFF_SET 0
#define VISIBLE_BUFF_BINDING 8
#include "occlusionCullingShaderCommon.glsl"

layout(push_constant, row_major) uniform PushConstants
{
    CullShaderData_t data;
} pc;

#include <nbl/builtin/glsl/utils/culling.glsl>
#include <nbl/builtin/glsl/utils/transform.glsl>


void main()
{
    if (gl_GlobalInvocationID.x >= pc.data.maxBatchCount)
        return;
    
    mvpBuff.mvps[gl_GlobalInvocationID.x] = pc.data.viewProjMatrix; // no model matrices

    const CullData_t batchCullData = cullData[gl_GlobalInvocationID.x];
    const uint drawCommandGUID = batchCullData.drawCommandGUID;
    occlusionCommandBuff.draws[drawCommandGUID].instanceCount = 0;

    if (bool(pc.data.freezeCulling))
        return;


    const mat2x3 bbox = mat2x3(batchCullData.aabbMinEdge,batchCullData.aabbMaxEdge);
    bool couldBeVisible = nbl_glsl_couldBeVisible(pc.data.viewProjMatrix,bbox);
    
    if (couldBeVisible)
    {
        const vec3 localCameraPos = pc.data.worldCamPos; // yep
        const bool cameraInsideAABB = all(greaterThanEqual(localCameraPos, batchCullData.aabbMinEdge)) && all(lessThanEqual(localCameraPos, batchCullData.aabbMaxEdge));
        const bool assumedVisible = uint(visibleBuff.visible[gl_GlobalInvocationID.x]) == 1u || cameraInsideAABB;
        frustumCommandBuff.draws[drawCommandGUID].instanceCount = assumedVisible ? 1u : 0u;
        // if not frustum culled and batch was not visible in the last frame, and it makes sense to test
        if(!assumedVisible)
        {
            const uint currCubeIdx = atomicAdd(cubeIndirectDraw.draw.instanceCount, 1);

            if(currCubeIdx % WORKGROUP_SIZE == 0) //TODO: distinct work group size for map shader and frustum cull shader
                atomicAdd(occlusionDispatchIndirect.di.num_groups_x, 1);

            // only works for a source geometry box which is [0,1]^2, the geometry creator box is [-0.5,0.5]^2, so either make your own box, or work out the math for this
            vec3 aabbExtent = batchCullData.aabbMaxEdge - batchCullData.aabbMinEdge;
            cubeMvpBuff.cubeMVPs[currCubeIdx] = mat4(
                pc.data.viewProjMatrix[0]*aabbExtent.x,
                pc.data.viewProjMatrix[1]*aabbExtent.y,
                pc.data.viewProjMatrix[2]*aabbExtent.z,
                pc.data.viewProjMatrix*vec4(batchCullData.aabbMinEdge,1)
            );

            cubeDrawGUIDBuffer.drawGUID[currCubeIdx] = drawCommandGUID;
        }
    }
    else
        frustumCommandBuff.draws[drawCommandGUID].instanceCount = 0;

    // does `freezeCulling` affect this negatively?
    visibleBuff.visible[gl_GlobalInvocationID.x] = uint16_t(0u);
} 