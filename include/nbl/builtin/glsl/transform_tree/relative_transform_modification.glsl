#ifndef _NBL_GLSL_TRANSFORM_TREE_RELATIVE_TRANSFORM_MODIFICATION_GLSL_INCLUDED_
#define _NBL_GLSL_TRANSFORM_TREE_RELATIVE_TRANSFORM_MODIFICATION_GLSL_INCLUDED_

// TODO: move this
#ifndef __cplusplus
#define NBL_INLINE inline
#else
#define NBL_INLINE inline
#endif

struct nbl_glsl_transform_tree_relative_transform_modification_t
{
	uvec4 data[3];
};
#define _NBL_BUILTIN_TRANSFORM_TREE_RELATIVE_TRANSFORM_MODIFICATION_T_E_TYPE_OVERWRITE_ 0
#define _NBL_BUILTIN_TRANSFORM_TREE_RELATIVE_TRANSFORM_MODIFICATION_T_E_TYPE_CONCATENATE_AFTER_ 1
#define _NBL_BUILTIN_TRANSFORM_TREE_RELATIVE_TRANSFORM_MODIFICATION_T_E_TYPE_CONCATENATE_BEFORE_ 2
#define _NBL_BUILTIN_TRANSFORM_TREE_RELATIVE_TRANSFORM_MODIFICATION_T_E_TYPE_WEIGHTED_ACCUMULATE_ 3
#define _NBL_BUILTIN_TRANSFORM_TREE_RELATIVE_TRANSFORM_MODIFICATION_T_E_TYPE_COUNT_ 4
#define _NBL_BUILTIN_TRANSFORM_TREE_RELATIVE_TRANSFORM_MODIFICATION_T_SIZE_ 16

// stuff the bits into x and z components of scale (without a rotation)
NBL_INLINE uint nbl_glsl_transform_tree_relative_transform_modification_t_getType(nbl_glsl_transform_tree_relative_transform_modification_t rtm)
{
	return (rtm.data[0][0]&0x1u)|((rtm.data[2][2]&0x1u)<<1u);
}
#ifndef __cplusplus
mat4x3 nbl_glsl_transform_tree_relative_transform_modification_t_getMatrix(in nbl_glsl_transform_tree_relative_transform_modification_t rtm)
{
	return transpose(mat3x4(uintBitsToFloat(rtm.data[0]),uintBitsToFloat(rtm.data[1]),uintBitsToFloat(rtm.data[2])));
}

mat4x3 nbl_glsl_transform_tree_relative_transform_modification_t_apply(in mat4x3 oldTform, in nbl_glsl_transform_tree_relative_transform_modification_t rtm)
{
	const mat4x3 delta = nbl_glsl_transform_tree_relative_transform_modification_t_getMatrix(rtm);
	switch (nbl_glsl_transform_tree_relative_transform_modification_t_getType(rtm))
	{
		case _NBL_BUILTIN_TRANSFORM_TREE_RELATIVE_TRANSFORM_MODIFICATION_T_E_TYPE_CONCATENATE_AFTER_:
			return delta*oldTform;
			break;
		case _NBL_BUILTIN_TRANSFORM_TREE_RELATIVE_TRANSFORM_MODIFICATION_T_E_TYPE_CONCATENATE_BEFORE_:
			return oldTform*delta;
			break;
		case _NBL_BUILTIN_TRANSFORM_TREE_RELATIVE_TRANSFORM_MODIFICATION_T_E_TYPE_WEIGHTED_ACCUMULATE_:
			return oldTform+delta;
			break;
		default:
			break;
	}
	return delta; // overwrite
}
#endif

#endif