#ifndef _IRR_BUILTIN_GLSL_SUBGROUP_ARITHMETIC_PORTABILITY_INCLUDED_
#define _IRR_BUILTIN_GLSL_SUBGROUP_ARITHMETIC_PORTABILITY_INCLUDED_


#include <irr/builtin/glsl/math/typeless_arithmetic.glsl>
#include <irr/builtin/glsl/subgroup/basic_portability.glsl>


/* TODO: @Hazardu or someone finish the definitions as soon as Nabla can report Vulkan GLSL equivalent caps
#ifdef GL_KHR_subgroup_basic
	#define SUBGROUP_BARRIERS subgroupBarrier(); \
	subgroupBarrierShared();
#else
*/
#define SUBGROUP_BARRIERS
//#endif

//#ifndef GL_KHR_subgroup_arithmetic
//#endif


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

#define IRR_GLSL_SUBGROUP_ARITHMETIC_GET_SHARED_OFFSET(IX,SUBGROUP_SIZE) ((((IX)&(~((SUBGROUP_SIZE)-1u)))<<1u)|((IX)&((SUBGROUP_SIZE)-1u)))
#if IRR_GLSL_SUBGROUP_SIZE_IS_CONSTEXPR
	#define _IRR_GLSL_SUBGROUP_ARITHMETIC_EMULATION_SHARED_SIZE_NEEDED_  (IRR_GLSL_SUBGROUP_ARITHMETIC_GET_SHARED_OFFSET(_IRR_GLSL_WORKGROUP_SIZE_-1u,irr_glsl_SubgroupSize)+irr_glsl_HalfSubgroupSize+1u)
#else
	#define _IRR_GLSL_SUBGROUP_ARITHMETIC_EMULATION_SHARED_SIZE_NEEDED_  (IRR_GLSL_SUBGROUP_ARITHMETIC_GET_SHARED_OFFSET(_IRR_GLSL_WORKGROUP_SIZE_-1u,irr_glsl_MinSubgroupSize)+(irr_glsl_MaxSubgroupSize>>1u)+1u)
#endif

#ifdef _IRR_GLSL_SCRATCH_SHARED_DEFINED_
/* can't get this to work either
	#if _IRR_GLSL_SCRATCH_SHARED_SIZE_DEFINED_<_IRR_GLSL_SUBGROUP_ARITHMETIC_EMULATION_SHARED_SIZE_NEEDED_
		#error "Not enough shared memory declared"
	#endif
*/
#else
	#if _IRR_GLSL_SUBGROUP_ARITHMETIC_EMULATION_SHARED_SIZE_NEEDED_>0
		#define _IRR_GLSL_SCRATCH_SHARED_DEFINED_ irr_glsl_subgroupArithmeticEmulationScratchShared
		shared uint _IRR_GLSL_SCRATCH_SHARED_DEFINED_[_IRR_GLSL_SUBGROUP_ARITHMETIC_EMULATION_SHARED_SIZE_NEEDED_];
	#endif
#endif

/*
How to avoid bank conflicts:
read:	00,01,02,03,    08,09,10,11,	16,17,18,19,    24,25,26,27,    04,05,06,07,    12,13,14,15,    20,21,22,23,    28,29,30,31
write:	30,31,00,01,    06,07,08,09,    14,15,16,17,    22,23,24,25,    02,03,04,05,    10,11,12,13,    18,19,20,21,    26,27,28,29

This design should also work for workgroups that are not divisible by subgroup size, which is neat as ****
*/
#define IRR_GLSL_SUBGROUP_ARITHMETIC_IMPL(OP,IDENTITY,VALUE) const uint scratchOffset = IRR_GLSL_SUBGROUP_ARITHMETIC_GET_SHARED_OFFSET(gl_LocalInvocationIndex,irr_glsl_SubgroupSize);
	const uint primaryOffset = scratchOffset+irr_glsl_HalfSubgroupSize; \
	_IRR_GLSL_SCRATCH_SHARED_DEFINED_[primaryOffset] = VALUE; \
	_IRR_GLSL_SCRATCH_SHARED_DEFINED_[scratchOffset] = IDENTITY; \
	SUBGROUP_BARRIERS \
	uint self = _IRR_GLSL_SCRATCH_SHARED_DEFINED_[primaryOffset]; \
	uint other = _IRR_GLSL_SCRATCH_SHARED_DEFINED_[primaryOffset-1u]; \
	_IRR_GLSL_SCRATCH_SHARED_DEFINED_[primaryOffset] = OP (self,other); \
	SUBGROUP_BARRIERS \
	self = _IRR_GLSL_SCRATCH_SHARED_DEFINED_[primaryOffset]; \
	other = _IRR_GLSL_SCRATCH_SHARED_DEFINED_[primaryOffset-2u]; \
	_IRR_GLSL_SCRATCH_SHARED_DEFINED_[primaryOffset] = OP (self,other); \
	for (uint stp=irr_glsl_MinSubgroupSize; stp<irr_glsl_SubgroupSize; stp<<=1u) \
	{ \
		SUBGROUP_BARRIERS \
		self = _IRR_GLSL_SCRATCH_SHARED_DEFINED_[primaryOffset]; \
		other = _IRR_GLSL_SCRATCH_SHARED_DEFINED_[primaryOffset-stp]; \
		_IRR_GLSL_SCRATCH_SHARED_DEFINED_[primaryOffset] = OP (self,other); \
	}



#define IRR_GLSL_SUBGROUP_REDUCE(CONV,OP,IDENTITY,VALUE) IRR_GLSL_SUBGROUP_ARITHMETIC_IMPL(OP,IDENTITY,VALUE) \
	SUBGROUP_BARRIERS \
	const uint loMask = irr_glsl_SubgroupSize-1u; \
	const uint hiMask = ~loMask; \
	const uint maxPseudoSubgroup = (_IRR_GLSL_WORKGROUP_SIZE_-1u)&hiMask; \
	const uint pseudoSubgroup = gl_LocalInvocationIndex&hiMask; \
	return CONV(_IRR_GLSL_SCRATCH_SHARED_DEFINED_[primaryOffset|((pseudoSubgroup!=maxPseudoSubgroup ? (_IRR_GLSL_WORKGROUP_SIZE_-1u):0xffffu)&loMask)])



uint irr_glsl_subgroupAnd(in uint value)
{
	IRR_GLSL_SUBGROUP_REDUCE(irr_glsl_identityFunction,irr_glsl_and,0xffFFffFFu,value);
}
int irr_glsl_subgroupAnd(in int value)
{
	return int(irr_glsl_subgroupAnd(int(value)));
}
float irr_glsl_subgroupAnd(in float value)
{
	return uintBitsToFloat(irr_glsl_subgroupAnd(floatBitsToUint(value)));
}

uint irr_glsl_subgroupXor(in uint value)
{
	IRR_GLSL_SUBGROUP_REDUCE(irr_glsl_identityFunction,irr_glsl_xor,0u,value);
}
int irr_glsl_subgroupXor(in int value)
{
	return int(irr_glsl_subgroupXor(int(value)));
}
float irr_glsl_subgroupXor(in float value)
{
	return uintBitsToFloat(irr_glsl_subgroupXor(floatBitsToUint(value)));
}

uint irr_glsl_subgroupOr(in uint value)
{
	IRR_GLSL_SUBGROUP_REDUCE(irr_glsl_identityFunction,irr_glsl_or,0u,value);
}
int irr_glsl_subgroupOr(in int value)
{
	return int(irr_glsl_subgroupOr(int(value)));
}
float irr_glsl_subgroupOr(in float value)
{
	return uintBitsToFloat(irr_glsl_subgroupOr(floatBitsToUint(value)));
}


uint irr_glsl_subgroupAdd(in uint value)
{
	IRR_GLSL_SUBGROUP_REDUCE(irr_glsl_identityFunction,irr_glsl_add,0u,value);
}
int irr_glsl_subgroupAdd(in int value)
{
	return int(irr_glsl_subgroupAdd(int(value)));
}
float irr_glsl_subgroupAdd(in float value)
{
	IRR_GLSL_SUBGROUP_REDUCE(uintBitsToFloat,irr_glsl_addAsFloat,floatBitsToUint(0.0),floatBitsToUint(value));
}

uint irr_glsl_subgroupMul(in uint value)
{
	IRR_GLSL_SUBGROUP_REDUCE(irr_glsl_identityFunction,irr_glsl_mul,1u,value);
}
int irr_glsl_subgroupMul(in int value)
{
	return int(irr_glsl_subgroupMul(int(value)));
}
float irr_glsl_subgroupMul(in float value)
{
	IRR_GLSL_SUBGROUP_REDUCE(uintBitsToFloat,irr_glsl_mulAsFloat,floatBitsToUint(1.0),floatBitsToUint(value)); 
}

uint irr_glsl_subgroupMin(in uint value)
{
	IRR_GLSL_SUBGROUP_REDUCE(irr_glsl_identityFunction,irr_glsl_min,UINT_MAX,value);
}
int irr_glsl_subgroupMin(in int value)
{
	IRR_GLSL_SUBGROUP_REDUCE(int,irr_glsl_minAsInt,uint(INT_MAX),uint(value));
}
float irr_glsl_subgroupMin(in float value)
{
	IRR_GLSL_SUBGROUP_REDUCE(uintBitsToFloat,irr_glsl_minAsFloat,floatBitsToUint(FLT_INF),floatBitsToUint(value)); 
}

uint irr_glsl_subgroupMax(in uint value)
{
	IRR_GLSL_SUBGROUP_REDUCE(irr_glsl_identityFunction,irr_glsl_max,UINT_MIN,value);
}
int irr_glsl_subgroupMax(in int value)
{
	IRR_GLSL_SUBGROUP_REDUCE(int,irr_glsl_maxAsInt,uint(INT_MIN),uint(value));
}
float irr_glsl_subgroupMax(in float value)
{
	IRR_GLSL_SUBGROUP_REDUCE(uintBitsToFloat,irr_glsl_maxAsFloat,floatBitsToUint(-FLT_INF),floatBitsToUint(value)); 
}



#define IRR_GLSL_SUBGROUP_INCLUSIVE_SCAN(CONV,OP,IDENTITY,VALUE) IRR_GLSL_SUBGROUP_ARITHMETIC_IMPL(OP,IDENTITY,VALUE) \
	return CONV (_IRR_GLSL_SCRATCH_SHARED_DEFINED_[primaryOffset])

#define IRR_GLSL_SUBGROUP_EXCLUSIVE_SCAN(CONV,OP,IDENTITY,VALUE) IRR_GLSL_SUBGROUP_ARITHMETIC_IMPL(OP,IDENTITY,VALUE) \
	return CONV (_IRR_GLSL_SCRATCH_SHARED_DEFINED_[primaryOffset-1u])


uint irr_glsl_subgroupInclusiveAnd(in uint value)
{
	IRR_GLSL_SUBGROUP_SCAN_INCLUSIVE(irr_glsl_identityFunction,irr_glsl_and,0xffFFffFFu,value);
}
int irr_glsl_subgroupInclusiveAnd(in int value)
{
	return int(irr_glsl_subgroupInclusiveAnd(int(value)));
}
float irr_glsl_subgroupInclusiveAnd(in float value)
{
	return uintBitsToFloat(irr_glsl_subgroupInclusiveAnd(floatBitsToUint(value)));
}
uint irr_glsl_subgroupExclusiveAnd(in uint value)
{
	IRR_GLSL_SUBGROUP_SCAN_EXCLUSIVE(irr_glsl_identityFunction,irr_glsl_and,0xffFFffFFu,value);
}
int irr_glsl_subgroupExclusiveAnd(in int value)
{
	return int(irr_glsl_subgroupExclusiveAnd(int(value)));
}
float irr_glsl_subgroupExclusiveAnd(in float value)
{
	return uintBitsToFloat(irr_glsl_subgroupExclusiveAnd(floatBitsToUint(value)));
}

uint irr_glsl_subgroupInclusiveXor(in uint value)
{
	IRR_GLSL_SUBGROUP_SCAN_INCLUSIVE(irr_glsl_identityFunction,irr_glsl_xor,0u,value);
}
int irr_glsl_subgroupInclusiveXor(in int value)
{
	return int(irr_glsl_subgroupInclusiveXor(int(value)));
}
float irr_glsl_subgroupInclusiveXor(in float value)
{
	return uintBitsToFloat(irr_glsl_subgroupInclusiveXor(floatBitsToUint(value)));
}
uint irr_glsl_subgroupExclusiveXor(in uint value)
{
	IRR_GLSL_SUBGROUP_SCAN_EXCLUSIVE(irr_glsl_identityFunction,irr_glsl_xor,0u,value);
}
int irr_glsl_subgroupExclusiveXor(in int value)
{
	return int(irr_glsl_subgroupExclusiveXor(int(value)));
}
float irr_glsl_subgroupExclusiveXor(in float value)
{
	return uintBitsToFloat(irr_glsl_subgroupExclusiveXor(floatBitsToUint(value)));
}

uint irr_glsl_subgroupInclusiveOr(in uint value)
{
	IRR_GLSL_SUBGROUP_SCAN_INCLUSIVE(irr_glsl_identityFunction,irr_glsl_or,0u,value);
}
int irr_glsl_subgroupInclusiveOr(in int value)
{
	return int(irr_glsl_subgroupInclusiveOr(int(value)));
}
float irr_glsl_subgroupInclusiveOr(in float value)
{
	return uintBitsToFloat(irr_glsl_subgroupInclusiveOr(floatBitsToUint(value)));
}
uint irr_glsl_subgroupExclusiveOr(in uint value)
{
	IRR_GLSL_SUBGROUP_SCAN_EXCLUSIVE(irr_glsl_identityFunction,irr_glsl_or,0u,value);
}
int irr_glsl_subgroupExclusiveOr(in int value)
{
	return int(irr_glsl_subgroupExclusiveOr(int(value)));
}
float irr_glsl_subgroupExclusiveOr(in float value)
{
	return uintBitsToFloat(irr_glsl_subgroupExclusiveOr(floatBitsToUint(value)));
}


uint irr_glsl_subgroupInclusiveAdd(in uint value)
{
	IRR_GLSL_SUBGROUP_SCAN_INCLUSIVE(irr_glsl_identityFunction,irr_glsl_add,0u,value);
}
int irr_glsl_subgroupInclusiveAdd(in int value)
{
	return int(irr_glsl_subgroupInclusiveAdd(int(value)));
}
float irr_glsl_subgroupInclusiveAdd(in float value)
{
	IRR_GLSL_SUBGROUP_SCAN_INCLUSIVE(uintBitsToFloat,irr_glsl_addAsFloat,floatBitsToUint(0.0),floatBitsToUint(value));
}
uint irr_glsl_subgroupExclusiveAdd(in uint value)
{
	IRR_GLSL_SUBGROUP_SCAN_EXCLUSIVE(irr_glsl_identityFunction,irr_glsl_add,0u,value);
}
int irr_glsl_subgroupExclusiveAdd(in int value)
{
	return int(irr_glsl_subgroupExclusiveAdd(int(value)));
}
float irr_glsl_subgroupExclusiveAdd(in float value)
{
	IRR_GLSL_SUBGROUP_SCAN_EXCLUSIVE(uintBitsToFloat,irr_glsl_addAsFloat,floatBitsToUint(0.0),floatBitsToUint(value));
}

uint irr_glsl_subgroupInclusiveMul(in uint value)
{
	IRR_GLSL_SUBGROUP_SCAN_INCLUSIVE(irr_glsl_identityFunction,irr_glsl_mul,1u,value);
}
int irr_glsl_subgroupInclusiveMul(in int value)
{
	return int(irr_glsl_subgroupInclusiveMul(int(value)));
}
float irr_glsl_subgroupInclusiveMul(in float value)
{
	IRR_GLSL_SUBGROUP_SCAN_INCLUSIVE(uintBitsToFloat,irr_glsl_mulAsFloat,floatBitsToUint(1.0),floatBitsToUint(value));
}
uint irr_glsl_subgroupExclusiveMul(in uint value)
{
	IRR_GLSL_SUBGROUP_SCAN_EXCLUSIVE(irr_glsl_identityFunction,irr_glsl_mul,1u,value);
}
int irr_glsl_subgroupExclusiveMul(in int value)
{
	return int(irr_glsl_subgroupExclusiveMul(int(value)));
}
float irr_glsl_subgroupExclusiveMul(in float value)
{
	IRR_GLSL_SUBGROUP_SCAN_EXCLUSIVE(uintBitsToFloat,irr_glsl_mulAsFloat,floatBitsToUint(1.0),floatBitsToUint(value));
}

uint irr_glsl_subgroupInclusiveMin(in uint value)
{
	IRR_GLSL_SUBGROUP_SCAN_INCLUSIVE(irr_glsl_identityFunction,irr_glsl_min,UINT_MAX,value);
}
int irr_glsl_subgroupInclusiveMin(in int value)
{
	IRR_GLSL_SUBGROUP_SCAN_INCLUSIVE(int,irr_glsl_minAsInt,uint(INT_MAX),uint(value));
}
float irr_glsl_subgroupInclusiveMin(in float value)
{
	IRR_GLSL_SUBGROUP_SCAN_INCLUSIVE(uintBitsToFloat,irr_glsl_minAsFloat,floatBitsToUint(FLT_INF),floatBitsToUint(value));
}
uint irr_glsl_subgroupExclusiveMin(in uint value)
{
	IRR_GLSL_SUBGROUP_SCAN_EXCLUSIVE(irr_glsl_identityFunction,irr_glsl_min,UINT_MAX,value);
}
int irr_glsl_subgroupExclusiveMin(in int value)
{
	IRR_GLSL_SUBGROUP_SCAN_EXCLUSIVE(int,irr_glsl_minAsInt,uint(INT_MAX),uint(value));
}
float irr_glsl_subgroupExclusiveMin(in float value)
{
	IRR_GLSL_SUBGROUP_SCAN_EXCLUSIVE(uintBitsToFloat,irr_glsl_minAsFloat,floatBitsToUint(FLT_INF),floatBitsToUint(value));
}

uint irr_glsl_subgroupInclusiveMax(in uint value)
{
	IRR_GLSL_SUBGROUP_SCAN_INCLUSIVE(irr_glsl_identityFunction,irr_glsl_max,UINT_MIN,value);
}
int irr_glsl_subgroupInclusiveMax(in int value)
{
	IRR_GLSL_SUBGROUP_SCAN_INCLUSIVE(int,irr_glsl_maxAsInt,uint(INT_MIN),uint(value));
}
float irr_glsl_subgroupInclusiveMax(in float value)
{
	IRR_GLSL_SUBGROUP_SCAN_INCLUSIVE(uintBitsToFloat,irr_glsl_maxAsFloat,floatBitsToUint(-FLT_INF),floatBitsToUint(value));
}
uint irr_glsl_subgroupExclusiveMax(in uint value)
{
	IRR_GLSL_SUBGROUP_SCAN_EXCLUSIVE(irr_glsl_identityFunction,irr_glsl_max,UINT_MIN,value);
}
int irr_glsl_subgroupExclusiveMax(in int value)
{
	IRR_GLSL_SUBGROUP_SCAN_EXCLUSIVE(int,irr_glsl_maxAsInt,uint(INT_MIN),uint(value));
}
float irr_glsl_subgroupExclusiveMax(in float value)
{
	IRR_GLSL_SUBGROUP_SCAN_EXCLUSIVE(uintBitsToFloat,irr_glsl_maxAsFloat,floatBitsToUint(-FLT_INF),floatBitsToUint(value));
}



#undef IRR_GLSL_SUBGROUP_ARITHMETIC_GET_SHARED_OFFSET



//#endif //GL_KHR_subgroup_arithmetic



#undef SUBGROUP_BARRIERS



#endif
