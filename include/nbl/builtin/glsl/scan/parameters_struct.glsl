#ifndef _NBL_GLSL_SCAN_PARAMETERS_STRUCT_INCLUDED_
#define _NBL_GLSL_SCAN_PARAMETERS_STRUCT_INCLUDED_


#define NBL_BUILTIN_MAX_SCAN_LEVELS 7

#ifdef __cplusplus
#define uint uint32_t
#endif
struct nbl_glsl_scan_Parameters_t
{
	uint topLevel;
	uint elementCount[NBL_BUILTIN_MAX_SCAN_LEVELS/2+1];
	uint temporaryStorageOffset[NBL_BUILTIN_MAX_SCAN_LEVELS/2];
};
#ifdef __cplusplus
#undef uint
#endif

#define _NBL_GLSL_SCAN_OP_AND_ 0
#define _NBL_GLSL_SCAN_OP_XOR_ 1
#define _NBL_GLSL_SCAN_OP_OR_ 2
#define _NBL_GLSL_SCAN_OP_ADD_ 3
#define _NBL_GLSL_SCAN_OP_MUL_ 4
#define _NBL_GLSL_SCAN_OP_MIN_ 5
#define _NBL_GLSL_SCAN_OP_MAX_ 6
#define _NBL_GLSL_SCAN_OP_COUNT_ 7

#define _NBL_GLSL_SCAN_TYPE_INCLUSIVE_ 0
#define _NBL_GLSL_SCAN_TYPE_EXCLUSIVE_ 1

#endif