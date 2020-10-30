#ifndef _IRR_BUILTIN_GLSL_SUBGROUP_ARITHMETIC_PORTABILITY_INCLUDED_
#define _IRR_BUILTIN_GLSL_SUBGROUP_ARITHMETIC_PORTABILITY_INCLUDED_


#include <irr/builtin/glsl/limits/numeric.glsl>
#include <irr/builtin/glsl/math/typeless_arithmetic.glsl>
#include <irr/builtin/glsl/subgroup/basic_portability.glsl>


/* TODO: @Hazardu or someone finish the definitions as soon as Nabla can report Vulkan GLSL equivalent caps
#ifdef GL_KHR_subgroup_basic

	#define SUBGROUP_BARRIERS subgroupBarrier(); \
	subgroupBarrierShared()

#else
*/

#define SUBGROUP_BARRIERS

//#endif


/*
#ifdef GL_KHR_subgroup_arithmetic


#define _IRR_GLSL_SUBGROUP_ARITHMETIC_EMULATION_SHARED_SIZE_NEEDED_ 0


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



#define _IRR_GLSL_SUBGROUP_ARITHMETIC_EMULATION_SHARED_SIZE_NEEDED_  (IRR_GLSL_EVAL(_IRR_GLSL_WORKGROUP_SIZE_<<1))

#ifdef _IRR_GLSL_SCRATCH_SHARED_DEFINED_
	#if IRR_GLSL_LESS(_IRR_GLSL_SCRATCH_SHARED_SIZE_DEFINED_,_IRR_GLSL_SUBGROUP_ARITHMETIC_EMULATION_SHARED_SIZE_NEEDED_)
		#error "Not enough shared memory declared"
	#endif
#else
	#if IRR_GLSL_GREATER(_IRR_GLSL_SUBGROUP_ARITHMETIC_EMULATION_SHARED_SIZE_NEEDED_,0)
		#define _IRR_GLSL_SCRATCH_SHARED_DEFINED_ irr_glsl_subgroupArithmeticEmulationScratchShared
		shared uint _IRR_GLSL_SCRATCH_SHARED_DEFINED_[_IRR_GLSL_SUBGROUP_ARITHMETIC_EMULATION_SHARED_SIZE_NEEDED_];
	#endif
#endif


/*
How to avoid bank conflicts:
read:	00,01,02,03,    08,09,10,11,	16,17,18,19,    24,25,26,27,    04,05,06,07,    12,13,14,15,    20,21,22,23,    28,29,30,31
write:	30,31,00,01,    06,07,08,09,    14,15,16,17,    22,23,24,25,    02,03,04,05,    10,11,12,13,    18,19,20,21,    26,27,28,29

This design should also work for workgroups that are not divisible by subgroup size, but only if we could clear the [0,SubgroupSize/2) range of _IRR_GLSL_SCRATCH_SHARED_DEFINED_ to the IDENTITY ELEMENT reliably.

TODO: Keep the pseudo subgroup and offset code DRY, move to a function.
*/
#define IRR_GLSL_SUBGROUP_ARITHMETIC_IMPL(OP,VALUE,CLEAR,IDENTITY) const uint loMask = irr_glsl_SubgroupSize-1u; \
	const uint pseudoSubgroupInvocation = gl_LocalInvocationIndex&loMask; \
	const uint hiMask = ~loMask; \
	const uint pseudoSubgroupID = gl_LocalInvocationIndex&hiMask; \
	const uint scratchOffset = (pseudoSubgroupID<<1u)|pseudoSubgroupInvocation; \
	const uint primaryOffset = scratchOffset+irr_glsl_HalfSubgroupSize; \
	_IRR_GLSL_SCRATCH_SHARED_DEFINED_[primaryOffset] = VALUE; \
	if (CLEAR && pseudoSubgroupInvocation<irr_glsl_HalfSubgroupSize) \
		_IRR_GLSL_SCRATCH_SHARED_DEFINED_[scratchOffset] = IDENTITY; \
	SUBGROUP_BARRIERS; \
	uint self = _IRR_GLSL_SCRATCH_SHARED_DEFINED_[primaryOffset]; \
	uint other = _IRR_GLSL_SCRATCH_SHARED_DEFINED_[primaryOffset-1u]; \
	_IRR_GLSL_SCRATCH_SHARED_DEFINED_[primaryOffset] = OP (self,other); \
	SUBGROUP_BARRIERS; \
	self = _IRR_GLSL_SCRATCH_SHARED_DEFINED_[primaryOffset]; \
	other = _IRR_GLSL_SCRATCH_SHARED_DEFINED_[primaryOffset-2u]; \
	_IRR_GLSL_SCRATCH_SHARED_DEFINED_[primaryOffset] = OP (self,other); \
	for (uint stp=irr_glsl_MinSubgroupSize; stp<irr_glsl_SubgroupSize; stp<<=1u) \
	{ \
		SUBGROUP_BARRIERS; \
		self = _IRR_GLSL_SCRATCH_SHARED_DEFINED_[primaryOffset]; \
		other = _IRR_GLSL_SCRATCH_SHARED_DEFINED_[primaryOffset-stp]; \
		_IRR_GLSL_SCRATCH_SHARED_DEFINED_[primaryOffset] = OP (self,other); \
	}


#define IRR_GLSL_SUBGROUP_REDUCE(CONV,OP,VALUE,CLEAR,IDENTITY) IRR_GLSL_SUBGROUP_ARITHMETIC_IMPL(OP,VALUE,CLEAR,IDENTITY) \
	SUBGROUP_BARRIERS; \
	const uint maxPseudoSubgroupInvocation = (_IRR_GLSL_WORKGROUP_SIZE_-1u)&loMask; \
	const uint maxPseudoSubgroupID = (_IRR_GLSL_WORKGROUP_SIZE_-1u)&hiMask; \
	return CONV(_IRR_GLSL_SCRATCH_SHARED_DEFINED_[((maxPseudoSubgroupID<<1u)|maxPseudoSubgroupInvocation)+irr_glsl_HalfSubgroupSize])



uint irr_glsl_subgroupAnd_impl(in bool clearScratchToIdentity, in uint value)
{
	IRR_GLSL_SUBGROUP_REDUCE(irr_glsl_identityFunction,irr_glsl_and,value,clearScratchToIdentity,0xffFFffFFu);
}
int irr_glsl_subgroupAnd_impl(in bool clearScratchToIdentity, in int value)
{
	return int(irr_glsl_subgroupAnd_impl(clearScratchToIdentity,int(value)));
}
float irr_glsl_subgroupAnd_impl(in bool clearScratchToIdentity, in float value)
{
	return uintBitsToFloat(irr_glsl_subgroupAnd_impl(clearScratchToIdentity,floatBitsToUint(value)));
}

uint irr_glsl_subgroupXor_impl(in bool clearScratchToIdentity, in uint value)
{
	IRR_GLSL_SUBGROUP_REDUCE(irr_glsl_identityFunction,irr_glsl_xor,value,clearScratchToIdentity,0u);
}
int irr_glsl_subgroupXor_impl(in bool clearScratchToIdentity, in int value)
{
	return int(irr_glsl_subgroupXor_impl(clearScratchToIdentity,int(value)));
}
float irr_glsl_subgroupXor_impl(in bool clearScratchToIdentity, in float value)
{
	return uintBitsToFloat(irr_glsl_subgroupXor_impl(clearScratchToIdentity,floatBitsToUint(value)));
}

uint irr_glsl_subgroupOr_impl(in bool clearScratchToIdentity, in uint value)
{
	IRR_GLSL_SUBGROUP_REDUCE(irr_glsl_identityFunction,irr_glsl_or,value,clearScratchToIdentity,0u);
}
int irr_glsl_subgroupOr_impl(in bool clearScratchToIdentity, in int value)
{
	return int(irr_glsl_subgroupOr_impl(clearScratchToIdentity,int(value)));
}
float irr_glsl_subgroupOr_impl(in bool clearScratchToIdentity, in float value)
{
	return uintBitsToFloat(irr_glsl_subgroupOr_impl(clearScratchToIdentity,floatBitsToUint(value)));
}


uint irr_glsl_subgroupAdd_impl(in bool clearScratchToIdentity, in uint value)
{
	IRR_GLSL_SUBGROUP_REDUCE(irr_glsl_identityFunction,irr_glsl_add,value,clearScratchToIdentity,0u);
}
int irr_glsl_subgroupAdd_impl(in bool clearScratchToIdentity, in int value)
{
	return int(irr_glsl_subgroupAdd_impl(clearScratchToIdentity,int(value)));
}
float irr_glsl_subgroupAdd_impl(in bool clearScratchToIdentity, in float value)
{
	IRR_GLSL_SUBGROUP_REDUCE(uintBitsToFloat,irr_glsl_addAsFloat,floatBitsToUint(value),clearScratchToIdentity,floatBitsToUint(0.0));
}

uint irr_glsl_subgroupMul_impl(in bool clearScratchToIdentity, in uint value)
{
	IRR_GLSL_SUBGROUP_REDUCE(irr_glsl_identityFunction,irr_glsl_mul,value,clearScratchToIdentity,1u);
}
int irr_glsl_subgroupMul_impl(in bool clearScratchToIdentity, in int value)
{
	return int(irr_glsl_subgroupMul_impl(clearScratchToIdentity,int(value)));
}
float irr_glsl_subgroupMul_impl(in bool clearScratchToIdentity, in float value)
{
	IRR_GLSL_SUBGROUP_REDUCE(uintBitsToFloat,irr_glsl_mulAsFloat,floatBitsToUint(value),clearScratchToIdentity,floatBitsToUint(1.0)); 
}

uint irr_glsl_subgroupMin_impl(in bool clearScratchToIdentity, in uint value)
{
	IRR_GLSL_SUBGROUP_REDUCE(irr_glsl_identityFunction,min,value,clearScratchToIdentity,UINT_MAX);
}
int irr_glsl_subgroupMin_impl(in bool clearScratchToIdentity, in int value)
{
	IRR_GLSL_SUBGROUP_REDUCE(int,irr_glsl_minAsInt,uint(value),clearScratchToIdentity,uint(INT_MAX));
}
float irr_glsl_subgroupMin_impl(in bool clearScratchToIdentity, in float value)
{
	IRR_GLSL_SUBGROUP_REDUCE(uintBitsToFloat,irr_glsl_minAsFloat,floatBitsToUint(value),clearScratchToIdentity,floatBitsToUint(FLT_INF)); 
}

uint irr_glsl_subgroupMax_impl(in bool clearScratchToIdentity, in uint value)
{
	IRR_GLSL_SUBGROUP_REDUCE(irr_glsl_identityFunction,max,value,clearScratchToIdentity,UINT_MIN);
}
int irr_glsl_subgroupMax_impl(in bool clearScratchToIdentity, in int value)
{
	IRR_GLSL_SUBGROUP_REDUCE(int,irr_glsl_maxAsInt,uint(value),clearScratchToIdentity,uint(INT_MIN));
}
float irr_glsl_subgroupMax_impl(in bool clearScratchToIdentity, in float value)
{
	IRR_GLSL_SUBGROUP_REDUCE(uintBitsToFloat,irr_glsl_maxAsFloat,floatBitsToUint(value),clearScratchToIdentity,floatBitsToUint(-FLT_INF)); 
}



#define IRR_GLSL_SUBGROUP_INCLUSIVE_SCAN(CONV,OP,VALUE,CLEAR,IDENTITY) IRR_GLSL_SUBGROUP_ARITHMETIC_IMPL(OP,VALUE,CLEAR,IDENTITY) \
	return CONV (_IRR_GLSL_SCRATCH_SHARED_DEFINED_[primaryOffset])

#define IRR_GLSL_SUBGROUP_EXCLUSIVE_SCAN(CONV,OP,VALUE,CLEAR,IDENTITY) IRR_GLSL_SUBGROUP_ARITHMETIC_IMPL(OP,VALUE,CLEAR,IDENTITY) \
	return CONV (_IRR_GLSL_SCRATCH_SHARED_DEFINED_[primaryOffset-1u])


uint irr_glsl_subgroupInclusiveAnd_impl(in bool clearScratchToIdentity, in uint value)
{
	IRR_GLSL_SUBGROUP_INCLUSIVE_SCAN(irr_glsl_identityFunction,irr_glsl_and,value,clearScratchToIdentity,0xffFFffFFu);
}
int irr_glsl_subgroupInclusiveAnd_impl(in bool clearScratchToIdentity, in int value)
{
	return int(irr_glsl_subgroupInclusiveAnd_impl(clearScratchToIdentity,int(value)));
}
float irr_glsl_subgroupInclusiveAnd_impl(in bool clearScratchToIdentity, in float value)
{
	return uintBitsToFloat(irr_glsl_subgroupInclusiveAnd_impl(clearScratchToIdentity,floatBitsToUint(value)));
}
uint irr_glsl_subgroupExclusiveAnd_impl(in bool clearScratchToIdentity, in uint value)
{
	IRR_GLSL_SUBGROUP_EXCLUSIVE_SCAN(irr_glsl_identityFunction,irr_glsl_and,value,clearScratchToIdentity,0xffFFffFFu);
}
int irr_glsl_subgroupExclusiveAnd_impl(in bool clearScratchToIdentity, in int value)
{
	return int(irr_glsl_subgroupExclusiveAnd_impl(clearScratchToIdentity,int(value)));
}
float irr_glsl_subgroupExclusiveAnd_impl(in bool clearScratchToIdentity, in float value)
{
	return uintBitsToFloat(irr_glsl_subgroupExclusiveAnd_impl(clearScratchToIdentity,floatBitsToUint(value)));
}

uint irr_glsl_subgroupInclusiveXor_impl(in bool clearScratchToIdentity, in uint value)
{
	IRR_GLSL_SUBGROUP_INCLUSIVE_SCAN(irr_glsl_identityFunction,irr_glsl_xor,value,clearScratchToIdentity,0u);
}
int irr_glsl_subgroupInclusiveXor_impl(in bool clearScratchToIdentity, in int value)
{
	return int(irr_glsl_subgroupInclusiveXor_impl(clearScratchToIdentity,int(value)));
}
float irr_glsl_subgroupInclusiveXor_impl(in bool clearScratchToIdentity, in float value)
{
	return uintBitsToFloat(irr_glsl_subgroupInclusiveXor_impl(clearScratchToIdentity,floatBitsToUint(value)));
}
uint irr_glsl_subgroupExclusiveXor_impl(in bool clearScratchToIdentity, in uint value)
{
	IRR_GLSL_SUBGROUP_EXCLUSIVE_SCAN(irr_glsl_identityFunction,irr_glsl_xor,value,clearScratchToIdentity,0u);
}
int irr_glsl_subgroupExclusiveXor_impl(in bool clearScratchToIdentity, in int value)
{
	return int(irr_glsl_subgroupExclusiveXor_impl(clearScratchToIdentity,int(value)));
}
float irr_glsl_subgroupExclusiveXor_impl(in bool clearScratchToIdentity, in float value)
{
	return uintBitsToFloat(irr_glsl_subgroupExclusiveXor_impl(clearScratchToIdentity,floatBitsToUint(value)));
}

uint irr_glsl_subgroupInclusiveOr_impl(in bool clearScratchToIdentity, in uint value)
{
	IRR_GLSL_SUBGROUP_INCLUSIVE_SCAN(irr_glsl_identityFunction,irr_glsl_or,value,clearScratchToIdentity,0u);
}
int irr_glsl_subgroupInclusiveOr_impl(in bool clearScratchToIdentity, in int value)
{
	return int(irr_glsl_subgroupInclusiveOr_impl(clearScratchToIdentity,int(value)));
}
float irr_glsl_subgroupInclusiveOr_impl(in bool clearScratchToIdentity, in float value)
{
	return uintBitsToFloat(irr_glsl_subgroupInclusiveOr_impl(clearScratchToIdentity,floatBitsToUint(value)));
}
uint irr_glsl_subgroupExclusiveOr_impl(in bool clearScratchToIdentity, in uint value)
{
	IRR_GLSL_SUBGROUP_EXCLUSIVE_SCAN(irr_glsl_identityFunction,irr_glsl_or,value,clearScratchToIdentity,0u);
}
int irr_glsl_subgroupExclusiveOr_impl(in bool clearScratchToIdentity, in int value)
{
	return int(irr_glsl_subgroupExclusiveOr_impl(clearScratchToIdentity,int(value)));
}
float irr_glsl_subgroupExclusiveOr_impl(in bool clearScratchToIdentity, in float value)
{
	return uintBitsToFloat(irr_glsl_subgroupExclusiveOr_impl(clearScratchToIdentity,floatBitsToUint(value)));
}


uint irr_glsl_subgroupInclusiveAdd_impl(in bool clearScratchToIdentity, in uint value)
{
	IRR_GLSL_SUBGROUP_INCLUSIVE_SCAN(irr_glsl_identityFunction,irr_glsl_add,value,clearScratchToIdentity,0u);
}
int irr_glsl_subgroupInclusiveAdd_impl(in bool clearScratchToIdentity, in int value)
{
	return int(irr_glsl_subgroupInclusiveAdd_impl(clearScratchToIdentity,int(value)));
}
float irr_glsl_subgroupInclusiveAdd_impl(in bool clearScratchToIdentity, in float value)
{
	IRR_GLSL_SUBGROUP_INCLUSIVE_SCAN(uintBitsToFloat,irr_glsl_addAsFloat,floatBitsToUint(value),clearScratchToIdentity,floatBitsToUint(0.0));
}
uint irr_glsl_subgroupExclusiveAdd_impl(in bool clearScratchToIdentity, in uint value)
{
	IRR_GLSL_SUBGROUP_EXCLUSIVE_SCAN(irr_glsl_identityFunction,irr_glsl_add,value,clearScratchToIdentity,0u);
}
int irr_glsl_subgroupExclusiveAdd_impl(in bool clearScratchToIdentity, in int value)
{
	return int(irr_glsl_subgroupExclusiveAdd_impl(clearScratchToIdentity,int(value)));
}
float irr_glsl_subgroupExclusiveAdd_impl(in bool clearScratchToIdentity, in float value)
{
	IRR_GLSL_SUBGROUP_EXCLUSIVE_SCAN(uintBitsToFloat,irr_glsl_addAsFloat,floatBitsToUint(value),clearScratchToIdentity,floatBitsToUint(0.0));
}

uint irr_glsl_subgroupInclusiveMul_impl(in bool clearScratchToIdentity, in uint value)
{
	IRR_GLSL_SUBGROUP_INCLUSIVE_SCAN(irr_glsl_identityFunction,irr_glsl_mul,value,clearScratchToIdentity,1u);
}
int irr_glsl_subgroupInclusiveMul_impl(in bool clearScratchToIdentity, in int value)
{
	return int(irr_glsl_subgroupInclusiveMul_impl(clearScratchToIdentity,int(value)));
}
float irr_glsl_subgroupInclusiveMul_impl(in bool clearScratchToIdentity, in float value)
{
	IRR_GLSL_SUBGROUP_INCLUSIVE_SCAN(uintBitsToFloat,irr_glsl_mulAsFloat,floatBitsToUint(value),clearScratchToIdentity,floatBitsToUint(1.0));
}
uint irr_glsl_subgroupExclusiveMul_impl(in bool clearScratchToIdentity, in uint value)
{
	IRR_GLSL_SUBGROUP_EXCLUSIVE_SCAN(irr_glsl_identityFunction,irr_glsl_mul,value,clearScratchToIdentity,1u);
}
int irr_glsl_subgroupExclusiveMul_impl(in bool clearScratchToIdentity, in int value)
{
	return int(irr_glsl_subgroupExclusiveMul_impl(clearScratchToIdentity,int(value)));
}
float irr_glsl_subgroupExclusiveMul_impl(in bool clearScratchToIdentity, in float value)
{
	IRR_GLSL_SUBGROUP_EXCLUSIVE_SCAN(uintBitsToFloat,irr_glsl_mulAsFloat,floatBitsToUint(value),clearScratchToIdentity,floatBitsToUint(1.0));
}

uint irr_glsl_subgroupInclusiveMin_impl(in bool clearScratchToIdentity, in uint value)
{
	IRR_GLSL_SUBGROUP_INCLUSIVE_SCAN(irr_glsl_identityFunction,min,value,clearScratchToIdentity,UINT_MAX);
}
int irr_glsl_subgroupInclusiveMin_impl(in bool clearScratchToIdentity, in int value)
{
	IRR_GLSL_SUBGROUP_INCLUSIVE_SCAN(int,irr_glsl_minAsInt,uint(value),clearScratchToIdentity,uint(INT_MAX));
}
float irr_glsl_subgroupInclusiveMin_impl(in bool clearScratchToIdentity, in float value)
{
	IRR_GLSL_SUBGROUP_INCLUSIVE_SCAN(uintBitsToFloat,irr_glsl_minAsFloat,floatBitsToUint(value),clearScratchToIdentity,floatBitsToUint(FLT_INF));
}
uint irr_glsl_subgroupExclusiveMin_impl(in bool clearScratchToIdentity, in uint value)
{
	IRR_GLSL_SUBGROUP_EXCLUSIVE_SCAN(irr_glsl_identityFunction,min,value,clearScratchToIdentity,UINT_MAX);
}
int irr_glsl_subgroupExclusiveMin_impl(in bool clearScratchToIdentity, in int value)
{
	IRR_GLSL_SUBGROUP_EXCLUSIVE_SCAN(int,irr_glsl_minAsInt,uint(value),clearScratchToIdentity,uint(INT_MAX));
}
float irr_glsl_subgroupExclusiveMin_impl(in bool clearScratchToIdentity, in float value)
{
	IRR_GLSL_SUBGROUP_EXCLUSIVE_SCAN(uintBitsToFloat,irr_glsl_minAsFloat,floatBitsToUint(value),clearScratchToIdentity,floatBitsToUint(FLT_INF));
}

uint irr_glsl_subgroupInclusiveMax_impl(in bool clearScratchToIdentity, in uint value)
{
	IRR_GLSL_SUBGROUP_INCLUSIVE_SCAN(irr_glsl_identityFunction,max,value,clearScratchToIdentity,UINT_MIN);
}
int irr_glsl_subgroupInclusiveMax_impl(in bool clearScratchToIdentity, in int value)
{
	IRR_GLSL_SUBGROUP_INCLUSIVE_SCAN(int,irr_glsl_maxAsInt,uint(value),clearScratchToIdentity,uint(INT_MIN));
}
float irr_glsl_subgroupInclusiveMax_impl(in bool clearScratchToIdentity, in float value)
{
	IRR_GLSL_SUBGROUP_INCLUSIVE_SCAN(uintBitsToFloat,irr_glsl_maxAsFloat,floatBitsToUint(value),clearScratchToIdentity,floatBitsToUint(-FLT_INF));
}
uint irr_glsl_subgroupExclusiveMax_impl(in bool clearScratchToIdentity, in uint value)
{
	IRR_GLSL_SUBGROUP_EXCLUSIVE_SCAN(irr_glsl_identityFunction,max,value,clearScratchToIdentity,UINT_MIN);
}
int irr_glsl_subgroupExclusiveMax_impl(in bool clearScratchToIdentity, in int value)
{
	IRR_GLSL_SUBGROUP_EXCLUSIVE_SCAN(int,irr_glsl_maxAsInt,uint(value),clearScratchToIdentity,uint(INT_MIN));
}
float irr_glsl_subgroupExclusiveMax_impl(in bool clearScratchToIdentity, in float value)
{
	IRR_GLSL_SUBGROUP_EXCLUSIVE_SCAN(uintBitsToFloat,irr_glsl_maxAsFloat,floatBitsToUint(value),clearScratchToIdentity,floatBitsToUint(-FLT_INF));
}



//#endif //GL_KHR_subgroup_arithmetic



#undef SUBGROUP_BARRIERS



#endif
