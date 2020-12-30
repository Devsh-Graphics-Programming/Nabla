#ifndef _NBL_BUILTIN_GLSL_SUBGROUP_ARITHMETIC_PORTABILITY_INCLUDED_
#define _NBL_BUILTIN_GLSL_SUBGROUP_ARITHMETIC_PORTABILITY_INCLUDED_


#include <nbl/builtin/glsl/subgroup/shared_arithmetic_portability.glsl>


/*
#ifdef GL_KHR_subgroup_arithmetic



#define nbl_glsl_subgroupAdd subgroupAnd

#define nbl_glsl_subgroupAdd subgroupXor

#define nbl_glsl_subgroupAdd subgroupOr


#define nbl_glsl_subgroupAdd subgroupAdd

#define nbl_glsl_subgroupAdd subgroupMul

#define nbl_glsl_subgroupAdd subgroupMin

#define nbl_glsl_subgroupAdd subgroupMax


#define nbl_glsl_subgroupExclusiveAdd subgroupExclusiveAnd
#define nbl_glsl_subgroupInclusiveAdd subgroupInclusiveAnd

#define nbl_glsl_subgroupExclusiveAdd subgroupExclusiveXor
#define nbl_glsl_subgroupInclusiveAdd subgroupInclusiveXor

#define nbl_glsl_subgroupExclusiveAdd subgroupExclusiveOr
#define nbl_glsl_subgroupInclusiveAdd subgroupInclusiveOr


#define nbl_glsl_subgroupExclusiveAdd subgroupExclusiveAdd
#define nbl_glsl_subgroupInclusiveAdd subgroupInclusiveAdd

#define nbl_glsl_subgroupExclusiveAdd subgroupExclusiveMul
#define nbl_glsl_subgroupInclusiveAdd subgroupInclusiveMul

#define nbl_glsl_subgroupExclusiveAdd subgroupExclusiveMin
#define nbl_glsl_subgroupInclusiveAdd subgroupInclusiveMin

#define nbl_glsl_subgroupExclusiveAdd subgroupExclusiveMax
#define nbl_glsl_subgroupInclusiveAdd subgroupInclusiveMax


#else
*/


// If you're planning to use the emulated `nbl_glsl_subgroup` with workgroup sizes not divisible by subgroup size, you should clear the _NBL_GLSL_SCRATCH_SHARED_DEFINED_ to the identity value yourself.
#define nbl_glsl_subgroupAnd(VALUE) nbl_glsl_subgroupAnd_impl(true,VALUE)

#define nbl_glsl_subgroupXor(VALUE) nbl_glsl_subgroupXor_impl(true,VALUE)

#define nbl_glsl_subgroupOr(VALUE) nbl_glsl_subgroupOr_impl(true,VALUE)


#define nbl_glsl_subgroupAdd(VALUE) nbl_glsl_subgroupAdd_impl(true,VALUE)

#define nbl_glsl_subgroupMul(VALUE) nbl_glsl_subgroupMul_impl(true,VALUE)

#define nbl_glsl_subgroupMin(VALUE) nbl_glsl_subgroupMin_impl(true,VALUE)

#define nbl_glsl_subgroupMax(VALUE) nbl_glsl_subgroupMax_impl(true,VALUE)


#define nbl_glsl_subgroupExclusiveAnd(VALUE) nbl_glsl_subgroupExclusiveAnd_impl(true,VALUE)
#define nbl_glsl_subgroupInclusiveAnd(VALUE) nbl_glsl_subgroupInclusiveAnd_impl(true,VALUE)

#define nbl_glsl_subgroupExclusiveXor(VALUE) nbl_glsl_subgroupExclusiveXor_impl(true,VALUE)
#define nbl_glsl_subgroupInclusiveXor(VALUE) nbl_glsl_subgroupInclusiveXor_impl(true,VALUE)

#define nbl_glsl_subgroupExclusiveOr(VALUE) nbl_glsl_subgroupExclusiveOr_impl(true,VALUE)
#define nbl_glsl_subgroupInclusiveOr(VALUE) nbl_glsl_subgroupInclusiveOr_impl(true,VALUE)


#define nbl_glsl_subgroupExclusiveAdd(VALUE) nbl_glsl_subgroupExclusiveAdd_impl(true,VALUE)
#define nbl_glsl_subgroupInclusiveAdd(VALUE) nbl_glsl_subgroupInclusiveAdd_impl(true,VALUE)

#define nbl_glsl_subgroupExclusiveMul(VALUE) nbl_glsl_subgroupExclusiveMul_impl(true,VALUE)
#define nbl_glsl_subgroupInclusiveMul(VALUE) nbl_glsl_subgroupInclusiveMul_impl(true,VALUE)

#define nbl_glsl_subgroupExclusiveMin(VALUE) nbl_glsl_subgroupExclusiveMin_impl(true,VALUE)
#define nbl_glsl_subgroupInclusiveMin(VALUE) nbl_glsl_subgroupInclusiveMin_impl(true,VALUE)

#define nbl_glsl_subgroupExclusiveMax(VALUE) nbl_glsl_subgroupExclusiveMax_impl(true,VALUE)
#define nbl_glsl_subgroupInclusiveMax(VALUE) nbl_glsl_subgroupInclusiveMax_impl(true,VALUE)



#ifdef _NBL_GLSL_SCRATCH_SHARED_DEFINED_
	#if NBL_GLSL_LESS(_NBL_GLSL_SCRATCH_SHARED_SIZE_DEFINED_,_NBL_GLSL_SUBGROUP_ARITHMETIC_EMULATION_SHARED_SIZE_NEEDED_)
		#error "Not enough shared memory declared"
	#endif
#else
	#if NBL_GLSL_GREATER(_NBL_GLSL_SUBGROUP_ARITHMETIC_EMULATION_SHARED_SIZE_NEEDED_,0)
		#define _NBL_GLSL_SCRATCH_SHARED_DEFINED_ nbl_glsl_subgroupArithmeticEmulationScratchShared
		#define _NBL_GLSL_SCRATCH_SHARED_SIZE_DEFINED_ _NBL_GLSL_SUBGROUP_ARITHMETIC_EMULATION_SHARED_SIZE_NEEDED_
		shared uint _NBL_GLSL_SCRATCH_SHARED_DEFINED_[_NBL_GLSL_SUBGROUP_ARITHMETIC_EMULATION_SHARED_SIZE_NEEDED_];
	#endif
#endif


#include <nbl/builtin/glsl/subgroup/arithmetic_portability_impl.glsl>


//#endif // GL_KHR_subgroup_arithmetic



#endif
