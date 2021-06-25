#version 440 core

	#define NBL_IMPL_GL_NV_viewport_array2
	#define NBL_IMPL_GL_NV_stereo_view_rendering
	#define NBL_IMPL_GL_NV_sample_mask_override_coverage
	#define NBL_IMPL_GL_NV_geometry_shader_passthrough
	#define NBL_IMPL_GL_NV_shader_subgroup_partitioned
	#define NBL_IMPL_GL_NV_compute_shader_derivatives
	#define NBL_IMPL_GL_NV_fragment_shader_barycentric
	#define NBL_IMPL_GL_NV_mesh_shader
	#define NBL_IMPL_GL_NV_shading_rate_image
	#define NBL_IMPL_GL_ARB_shading_language_include
	#define NBL_IMPL_GL_ARB_enhanced_layouts
	#define NBL_IMPL_GL_ARB_bindless_texture
	#define NBL_IMPL_GL_ARB_shader_draw_parameters
	#define NBL_IMPL_GL_ARB_shader_group_vote
	#define NBL_IMPL_GL_ARB_cull_distance
	#define NBL_IMPL_GL_ARB_derivative_control
	#define NBL_IMPL_GL_ARB_shader_texture_image_samples
	#define NBL_IMPL_GL_KHR_blend_equation_advanced
	#define NBL_IMPL_GL_KHR_blend_equation_advanced_coherent
	#define NBL_IMPL_GL_ARB_fragment_shader_interlock
	#define NBL_IMPL_GL_ARB_gpu_shader_int64
	#define NBL_IMPL_GL_ARB_post_depth_coverage
	#define NBL_IMPL_GL_ARB_shader_ballot
	#define NBL_IMPL_GL_ARB_shader_clock
	#define NBL_IMPL_GL_ARB_shader_viewport_layer_array
	#define NBL_IMPL_GL_ARB_sparse_texture2
	#define NBL_IMPL_GL_ARB_sparse_texture_clamp
	#define NBL_IMPL_GL_ARB_gl_spirv
	#define NBL_IMPL_GL_ARB_spirv_extensions
	#define NBL_IMPL_GL_AMD_vertex_shader_viewport_index
	#define NBL_IMPL_GL_AMD_vertex_shader_layer
	#define NBL_IMPL_GL_NV_bindless_texture
	#define NBL_IMPL_GL_NV_shader_atomic_float
	#define NBL_IMPL_GL_EXT_shader_integer_mix
	#define NBL_IMPL_GL_NV_shader_thread_group
	#define NBL_IMPL_GL_NV_shader_thread_shuffle
	#define NBL_IMPL_GL_EXT_shader_image_load_formatted
	#define NBL_IMPL_GL_NV_shader_atomic_int64
	#define NBL_IMPL_GL_EXT_post_depth_coverage
	#define NBL_IMPL_GL_EXT_sparse_texture2
	#define NBL_IMPL_GL_NV_fragment_shader_interlock
	#define NBL_IMPL_GL_NV_sample_locations
	#define NBL_IMPL_GL_NV_shader_atomic_fp16_vector
	#define NBL_IMPL_GL_NV_command_list
	#define NBL_IMPL_GL_OVR_multiview
	#define NBL_IMPL_GL_OVR_multiview2
	#define NBL_IMPL_GL_NV_shader_atomic_float64
	#define NBL_IMPL_GL_NV_conservative_raster_pre_snap
	#define NBL_IMPL_GL_NV_shader_texture_footprint
	#define NBL_IMPL_GL_NV_gpu_shader5

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
#define _NBL_BUILTIN_PROPERTY_COUNT_ 7
#define _NBL_BUILTIN_PROPERTY_COPY_GROUP_SIZE_ 256

layout(local_size_x=_NBL_BUILTIN_PROPERTY_COPY_GROUP_SIZE_) in;

layout(set=0,binding=0) readonly restrict buffer Indices
{
    uint elementCount[_NBL_BUILTIN_PROPERTY_COUNT_];
	int propertyDWORDsize_upDownFlag[_NBL_BUILTIN_PROPERTY_COUNT_];
    uint indexOffset[_NBL_BUILTIN_PROPERTY_COUNT_];
    uint indices[];
};


layout(set=0, binding=1) readonly restrict buffer InData
{
    uint data[];
} inBuff[_NBL_BUILTIN_PROPERTY_COUNT_];
layout(set=0, binding=2) writeonly restrict buffer OutData
{
    uint data[];
} outBuff[_NBL_BUILTIN_PROPERTY_COUNT_];


#if 0 // optimization
uint shared workgroupShared[_NBL_BUILTIN_PROPERTY_COPY_GROUP_SIZE_];
#endif


void main()
{
    const uint propID = gl_WorkGroupID.y;

	const int combinedFlag = propertyDWORDsize_upDownFlag[propID];
	const bool download = combinedFlag<0;

	const uint propDWORDs = uint(download ? (-combinedFlag):combinedFlag);
#if 0 // optimization
	const uint localIx = gl_LocalInvocationID.x;
	const uint MaxItemsToProcess = ;
	if (localIx<MaxItemsToProcess)
		workgroupShared[localIx] = indices[localIx+indexOffset[propID]];
	barrier();
	memoryBarrier();
#endif

    const uint index = gl_GlobalInvocationID.x/propDWORDs;
    if (index>=elementCount[propID])
        return;

	const uint redir = (
#if 0 //optimization
		workgroupShared[index]
#else 
		indices[index+indexOffset[propID]]
#endif
	// its equivalent to `indices[index]*propDWORDs+gl_GlobalInvocationID.x%propDWORDs`
    -index)*propDWORDs+gl_GlobalInvocationID.x;

    const uint inIndex = download ? redir:gl_GlobalInvocationID.x;
    const uint outIndex = download ? gl_GlobalInvocationID.x:redir;
	outBuff[propID].data[outIndex] = inBuff[propID].data[inIndex];
}
 