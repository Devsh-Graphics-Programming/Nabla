#ifndef _NBL_BUILTIN_GLSL_WORKGROUP_SHARED_BALLOT_INCLUDED_
#define _NBL_BUILTIN_GLSL_WORKGROUP_SHARED_BALLOT_INCLUDED_



#include <nbl/builtin/glsl/workgroup/basic.glsl>
#include <nbl/builtin/glsl/subgroup/shared_arithmetic_portability.glsl>



#define nbl_glsl_workgroupBallot_impl_getDWORD(IX) (IX>>5)
#define nbl_glsl_workgroupBallot_impl_BitfieldDWORDs nbl_glsl_workgroupBallot_impl_getDWORD(_NBL_GLSL_WORKGROUP_SIZE_+31)

// TODO: test if it actually works
//#ifdef NBL_GLSL_SUBGROUP_EMULATION_SCRATCH_BOUND
#define NBL_GLSL_WORKGROUP_REDUCTION_SCRATCH_BOUND(LAST_ITEM) NBL_GLSL_EVAL(NBL_GLSL_SUBGROUP_EMULATION_SCRATCH_BOUND(LAST_ITEM))
//#else
//#define NBL_GLSL_WORKGROUP_REDUCTION_SCRATCH_BOUND(LAST_ITEM) 0
//#endif

#define NBL_GLSL_WORKGROUP_SCAN_SCRATCH_BOUND_IMPL(LAST_ITEM,SUBGROUP_SIZE_LOG2) (NBL_GLSL_WORKGROUP_REDUCTION_SCRATCH_BOUND(LAST_ITEM)+ \
	(LAST_ITEM>>(SUBGROUP_SIZE_LOG2))+ \
	(LAST_ITEM>>(SUBGROUP_SIZE_LOG2*2))+ \
	(LAST_ITEM>>(SUBGROUP_SIZE_LOG2*3))+ \
	(LAST_ITEM>>(SUBGROUP_SIZE_LOG2*4))+ \
	(LAST_ITEM>>(SUBGROUP_SIZE_LOG2*5))+ \
5)

#if defined(NBL_GLSL_SUBGROUP_SIZE_IS_CONSTEXPR)
	#if (nbl_glsl_MaxWorkgroupSizeLog2-nbl_glsl_SubgroupSizeLog2*6)>0
		#error "Someone updated nbl_glsl_MaxWorkgroupSizeLog2 without letting us know" 
	#endif
	#define NBL_GLSL_WORKGROUP_SCAN_SCRATCH_BOUND(LAST_ITEM)  NBL_GLSL_WORKGROUP_SCAN_SCRATCH_BOUND_IMPL(LAST_ITEM,nbl_glsl_SubgroupSizeLog2)
#else
	#define NBL_GLSL_WORKGROUP_SCAN_SCRATCH_BOUND(LAST_ITEM)  NBL_GLSL_WORKGROUP_SCAN_SCRATCH_BOUND_IMPL(LAST_ITEM,nbl_glsl_MinSubgroupSizeLog2)
#endif



#if NBL_GLSL_WORKGROUP_REDUCTION_SCRATCH_BOUND(nbl_glsl_workgroupBallot_impl_BitfieldDWORDs-1)>nbl_glsl_workgroupBallot_impl_BitfieldDWORDs
	#define _NBL_GLSL_WORKGROUP_BALLOT_SHARED_SIZE_NEEDED_  NBL_GLSL_WORKGROUP_REDUCTION_SCRATCH_BOUND(nbl_glsl_workgroupBallot_impl_BitfieldDWORDs-1)
#else
	#define _NBL_GLSL_WORKGROUP_BALLOT_SHARED_SIZE_NEEDED_  (nbl_glsl_workgroupBallot_impl_BitfieldDWORDs+1)
#endif



#endif
