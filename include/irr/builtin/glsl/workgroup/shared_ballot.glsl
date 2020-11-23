#ifndef _IRR_BUILTIN_GLSL_WORKGROUP_SHARED_BALLOT_INCLUDED_
#define _IRR_BUILTIN_GLSL_WORKGROUP_SHARED_BALLOT_INCLUDED_



#include <irr/builtin/glsl/workgroup/basic.glsl>
#include <irr/builtin/glsl/subgroup/shared_arithmetic_portability.glsl>



#define irr_glsl_workgroupBallot_impl_getDWORD(IX) (IX>>5)
#define irr_glsl_workgroupBallot_impl_BitfieldDWORDs irr_glsl_workgroupBallot_impl_getDWORD(_IRR_GLSL_WORKGROUP_SIZE_+31)



#define IRR_GLSL_WORKGROUP_REDUCTION_SCRATCH_BOUND(LAST_ITEM) IRR_GLSL_EVAL(IRR_GLSL_SUBGROUP_EMULATION_SCRATCH_BOUND(LAST_ITEM))


#define IRR_GLSL_WORKGROUP_SCAN_SCRATCH_BOUND_IMPL(LAST_ITEM,SUBGROUP_SIZE_LOG2) (IRR_GLSL_WORKGROUP_REDUCTION_SCRATCH_BOUND(LAST_ITEM)+ \
	(LAST_ITEM>>(SUBGROUP_SIZE_LOG2))+ \
	(LAST_ITEM>>(SUBGROUP_SIZE_LOG2*2))+ \
	(LAST_ITEM>>(SUBGROUP_SIZE_LOG2*3))+ \
	(LAST_ITEM>>(SUBGROUP_SIZE_LOG2*4))+ \
	(LAST_ITEM>>(SUBGROUP_SIZE_LOG2*5))+ \
5)

#if defined(IRR_GLSL_SUBGROUP_SIZE_IS_CONSTEXPR)
	#if (irr_glsl_MaxWorkgroupSizeLog2-irr_glsl_SubgroupSizeLog2*6)>0
		#error "Someone updated irr_glsl_MaxWorkgroupSizeLog2 without letting us know" 
	#endif
	#define IRR_GLSL_WORKGROUP_SCAN_SCRATCH_BOUND(LAST_ITEM)  IRR_GLSL_WORKGROUP_SCAN_SCRATCH_BOUND_IMPL(LAST_ITEM,irr_glsl_SubgroupSizeLog2)
#else
	#define IRR_GLSL_WORKGROUP_SCAN_SCRATCH_BOUND(LAST_ITEM)  IRR_GLSL_WORKGROUP_SCAN_SCRATCH_BOUND_IMPL(LAST_ITEM,irr_glsl_MinSubgroupSizeLog2)
#endif



#if IRR_GLSL_WORKGROUP_REDUCTION_SCRATCH_BOUND(irr_glsl_workgroupBallot_impl_BitfieldDWORDs-1)>irr_glsl_workgroupBallot_impl_BitfieldDWORDs
	#define _IRR_GLSL_WORKGROUP_BALLOT_SHARED_SIZE_NEEDED_  IRR_GLSL_WORKGROUP_REDUCTION_SCRATCH_BOUND(irr_glsl_workgroupBallot_impl_BitfieldDWORDs-1)
#else
	#define _IRR_GLSL_WORKGROUP_BALLOT_SHARED_SIZE_NEEDED_  (irr_glsl_workgroupBallot_impl_BitfieldDWORDs+1)
#endif



#endif
