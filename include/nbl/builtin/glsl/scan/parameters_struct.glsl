#ifndef _NBL_GLSL_SCAN_PARAMETERS_STRUCT_INCLUDED_
#define _NBL_GLSL_SCAN_PARAMETERS_STRUCT_INCLUDED_

#ifdef __cplusplus
#define uint uint32_t
#endif
struct nbl_glsl_scan_Parameters_t
{
	uint elementCount;
	//uint stride;
	//uint element_count_pass;
	//uint element_count_total;
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

#endif