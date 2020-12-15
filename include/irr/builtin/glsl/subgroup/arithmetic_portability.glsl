#ifndef _IRR_BUILTIN_GLSL_SUBGROUP_ARITHMETIC_PORTABILITY_INCLUDED_
#define _IRR_BUILTIN_GLSL_SUBGROUP_ARITHMETIC_PORTABILITY_INCLUDED_


#include <irr/builtin/glsl/subgroup/shared_arithmetic_portability.glsl>


/*
#ifdef GL_KHR_subgroup_arithmetic



#define irr_glsl_subgroupAdd subgroupAnd

#define irr_glsl_subgroupAdd subgroupXor

#define irr_glsl_subgroupAdd subgroupOr


#define irr_glsl_subgroupAdd subgroupAdd

#define irr_glsl_subgroupAdd subgroupMul

#define irr_glsl_subgroupAdd subgroupMin

#define irr_glsl_subgroupAdd subgroupMax


#define irr_glsl_subgroupExclusiveAdd subgroupExclusiveAnd
#define irr_glsl_subgroupInclusiveAdd subgroupInclusiveAnd

#define irr_glsl_subgroupExclusiveAdd subgroupExclusiveXor
#define irr_glsl_subgroupInclusiveAdd subgroupInclusiveXor

#define irr_glsl_subgroupExclusiveAdd subgroupExclusiveOr
#define irr_glsl_subgroupInclusiveAdd subgroupInclusiveOr


#define irr_glsl_subgroupExclusiveAdd subgroupExclusiveAdd
#define irr_glsl_subgroupInclusiveAdd subgroupInclusiveAdd

#define irr_glsl_subgroupExclusiveAdd subgroupExclusiveMul
#define irr_glsl_subgroupInclusiveAdd subgroupInclusiveMul

#define irr_glsl_subgroupExclusiveAdd subgroupExclusiveMin
#define irr_glsl_subgroupInclusiveAdd subgroupInclusiveMin

#define irr_glsl_subgroupExclusiveAdd subgroupExclusiveMax
#define irr_glsl_subgroupInclusiveAdd subgroupInclusiveMax


#else
*/


// If you're planning to use the emulated `irr_glsl_subgroup` with workgroup sizes not divisible by subgroup size, you should clear the _IRR_GLSL_SCRATCH_SHARED_DEFINED_ to the identity value yourself.
#define irr_glsl_subgroupAnd(VALUE) irr_glsl_subgroupAnd_impl(true,VALUE)

#define irr_glsl_subgroupXor(VALUE) irr_glsl_subgroupXor_impl(true,VALUE)

#define irr_glsl_subgroupOr(VALUE) irr_glsl_subgroupOr_impl(true,VALUE)


#define irr_glsl_subgroupAdd(VALUE) irr_glsl_subgroupAdd_impl(true,VALUE)

#define irr_glsl_subgroupMul(VALUE) irr_glsl_subgroupMul_impl(true,VALUE)

#define irr_glsl_subgroupMin(VALUE) irr_glsl_subgroupMin_impl(true,VALUE)

#define irr_glsl_subgroupMax(VALUE) irr_glsl_subgroupMax_impl(true,VALUE)


#define irr_glsl_subgroupExclusiveAnd(VALUE) irr_glsl_subgroupExclusiveAnd_impl(true,VALUE)
#define irr_glsl_subgroupInclusiveAnd(VALUE) irr_glsl_subgroupInclusiveAnd_impl(true,VALUE)

#define irr_glsl_subgroupExclusiveXor(VALUE) irr_glsl_subgroupExclusiveXor_impl(true,VALUE)
#define irr_glsl_subgroupInclusiveXor(VALUE) irr_glsl_subgroupInclusiveXor_impl(true,VALUE)

#define irr_glsl_subgroupExclusiveOr(VALUE) irr_glsl_subgroupExclusiveOr_impl(true,VALUE)
#define irr_glsl_subgroupInclusiveOr(VALUE) irr_glsl_subgroupInclusiveOr_impl(true,VALUE)


#define irr_glsl_subgroupExclusiveAdd(VALUE) irr_glsl_subgroupExclusiveAdd_impl(true,VALUE)
#define irr_glsl_subgroupInclusiveAdd(VALUE) irr_glsl_subgroupInclusiveAdd_impl(true,VALUE)

#define irr_glsl_subgroupExclusiveMul(VALUE) irr_glsl_subgroupExclusiveMul_impl(true,VALUE)
#define irr_glsl_subgroupInclusiveMul(VALUE) irr_glsl_subgroupInclusiveMul_impl(true,VALUE)

#define irr_glsl_subgroupExclusiveMin(VALUE) irr_glsl_subgroupExclusiveMin_impl(true,VALUE)
#define irr_glsl_subgroupInclusiveMin(VALUE) irr_glsl_subgroupInclusiveMin_impl(true,VALUE)

#define irr_glsl_subgroupExclusiveMax(VALUE) irr_glsl_subgroupExclusiveMax_impl(true,VALUE)
#define irr_glsl_subgroupInclusiveMax(VALUE) irr_glsl_subgroupInclusiveMax_impl(true,VALUE)



#ifdef _IRR_GLSL_SCRATCH_SHARED_DEFINED_
	#if IRR_GLSL_LESS(_IRR_GLSL_SCRATCH_SHARED_SIZE_DEFINED_,_IRR_GLSL_SUBGROUP_ARITHMETIC_EMULATION_SHARED_SIZE_NEEDED_)
		#error "Not enough shared memory declared"
	#endif
#else
	#if IRR_GLSL_GREATER(_IRR_GLSL_SUBGROUP_ARITHMETIC_EMULATION_SHARED_SIZE_NEEDED_,0)
		#define _IRR_GLSL_SCRATCH_SHARED_DEFINED_ irr_glsl_subgroupArithmeticEmulationScratchShared
		#define _IRR_GLSL_SCRATCH_SHARED_SIZE_DEFINED_ _IRR_GLSL_SUBGROUP_ARITHMETIC_EMULATION_SHARED_SIZE_NEEDED_
		shared uint _IRR_GLSL_SCRATCH_SHARED_DEFINED_[_IRR_GLSL_SUBGROUP_ARITHMETIC_EMULATION_SHARED_SIZE_NEEDED_];
	#endif
#endif


#include <irr/builtin/glsl/subgroup/arithmetic_portability_impl.glsl>


//#endif // GL_KHR_subgroup_arithmetic



#endif
