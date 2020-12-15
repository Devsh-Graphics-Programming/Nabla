#ifndef _IRR_BUILTIN_GLSL_SUBGROUP_ARITHMETIC_PORTABILITY_IMPLEMENTATION_INCLUDED_
#define _IRR_BUILTIN_GLSL_SUBGROUP_ARITHMETIC_PORTABILITY_IMPLEMENTATION_INCLUDED_


/* TODO: @Hazardu or someone finish the definitions as soon as Nabla can report Vulkan GLSL equivalent caps
#ifdef GL_KHR_subgroup_basic

	#define SUBGROUP_BARRIERS subgroupBarrier(); \
	subgroupBarrierShared()

#else
*/

#define SUBGROUP_BARRIERS memoryBarrierShared()

//#endif




/*
How to avoid bank conflicts:
read:	00,01,02,03,    08,09,10,11,	16,17,18,19,    24,25,26,27,    04,05,06,07,    12,13,14,15,    20,21,22,23,    28,29,30,31
write:	30,31,00,01,    06,07,08,09,    14,15,16,17,    22,23,24,25,    02,03,04,05,    10,11,12,13,    18,19,20,21,    26,27,28,29

This design should also work for workgroups that are not divisible by subgroup size, but only if we could initialize the _IRR_GLSL_SCRATCH_SHARED_DEFINED_ reliably.

TODO: add [[unroll]] to all loops
*/
uint irr_glsl_subgroup_impl_pseudoSubgroupElectedInvocation(in uint loMask, in uint invocationIndex)
{
	return invocationIndex&(~loMask);
}
uint irr_glsl_subgroup_impl_pseudoSubgroupInvocation(in uint loMask, in uint invocationIndex)
{
	return invocationIndex&loMask;
}
uint irr_glsl_subgroup_impl_getSubgroupEmulationMemoryStart(in uint pseudoSubgroupElectedInvocation)
{
	return pseudoSubgroupElectedInvocation<<1u;
}
uint irr_glsl_subgroup_impl_getSubgroupEmulationMemoryStoreOffset(in uint subgroupMemoryStart, in uint pseudoSubgroupInvocation, out uint lastLoadOffset)
{
	lastLoadOffset = (subgroupMemoryStart|pseudoSubgroupInvocation);
	return lastLoadOffset+irr_glsl_HalfSubgroupSize;
}
uint irr_glsl_subgroup_impl_getSubgroupEmulationMemoryStoreOffset(in uint subgroupMemoryStart, in uint pseudoSubgroupInvocation)
{
	uint dummy;
	return irr_glsl_subgroup_impl_getSubgroupEmulationMemoryStoreOffset(subgroupMemoryStart,pseudoSubgroupInvocation,dummy);
}
uint irr_glsl_subgroup_getSubgroupEmulationMemoryStoreOffset(in uint loMask, in uint invocationIndex)
{
	return irr_glsl_subgroup_impl_getSubgroupEmulationMemoryStoreOffset(
		irr_glsl_subgroup_impl_getSubgroupEmulationMemoryStart(
			irr_glsl_subgroup_impl_pseudoSubgroupElectedInvocation(loMask,invocationIndex)
		),
		irr_glsl_subgroup_impl_pseudoSubgroupInvocation(loMask,invocationIndex)
	);
}

#define SUBGROUP_SCRATCH_OFFSETS_AND_MASKS const uint loMask = irr_glsl_SubgroupSize-1u; \
	const uint pseudoSubgroupElectedInvocation = irr_glsl_subgroup_impl_pseudoSubgroupElectedInvocation(loMask,gl_LocalInvocationIndex); \
	const uint pseudoSubgroupInvocation = irr_glsl_subgroup_impl_pseudoSubgroupInvocation(loMask,gl_LocalInvocationIndex); \
	const uint subgroupMemoryStart = irr_glsl_subgroup_impl_getSubgroupEmulationMemoryStart(pseudoSubgroupElectedInvocation); \
	uint lastLoadOffset = 0xdeadbeefu; \
	const uint subgroupScanStoreOffset = irr_glsl_subgroup_impl_getSubgroupEmulationMemoryStoreOffset(subgroupMemoryStart,pseudoSubgroupInvocation,lastLoadOffset)


#define SUBGROUP_SCRATCH_INITIALIZE_IMPL_CLEAR_INDEX(PSEUDO_INVOCATION) ((((PSEUDO_INVOCATION)&(~halfMask))<<2u)|((PSEUDO_INVOCATION)&halfMask))

#define SUBGROUP_SCRATCH_INITIALIZE(VALUE,ACTIVE_INVOCATION_INDEX_UPPER_BOUND,IDENTITY,INVCONV) SUBGROUP_SCRATCH_OFFSETS_AND_MASKS; \
	{ \
		_IRR_GLSL_SCRATCH_SHARED_DEFINED_[subgroupScanStoreOffset] = INVCONV (VALUE); \
		const uint halfMask = loMask>>1u; \
		_IRR_GLSL_SCRATCH_SHARED_DEFINED_[SUBGROUP_SCRATCH_INITIALIZE_IMPL_CLEAR_INDEX(gl_LocalInvocationIndex)] = INVCONV (IDENTITY); \
		if (_IRR_GLSL_WORKGROUP_SIZE_<irr_glsl_HalfSubgroupSize) \
		{ \
			const uint maxItemsToClear = (irr_glsl_subgroup_impl_pseudoSubgroupElectedInvocation(loMask,ACTIVE_INVOCATION_INDEX_UPPER_BOUND-1u)>>1u)+irr_glsl_HalfSubgroupSize; \
			for (uint ix=gl_LocalInvocationIndex+_IRR_GLSL_WORKGROUP_SIZE_; ix<maxItemsToClear; ix+=_IRR_GLSL_WORKGROUP_SIZE_) \
				_IRR_GLSL_SCRATCH_SHARED_DEFINED_[SUBGROUP_SCRATCH_INITIALIZE_IMPL_CLEAR_INDEX(ix)] = INVCONV (IDENTITY); \
		} \
		barrier(); \
	}


#define IRR_GLSL_SUBGROUP_ARITHMETIC_IMPL(CONV,OP,VALUE,INITIALIZE,IDENTITY,INVCONV) SUBGROUP_SCRATCH_OFFSETS_AND_MASKS; \
	if (INITIALIZE) \
	{ \
		SUBGROUP_BARRIERS; \
		_IRR_GLSL_SCRATCH_SHARED_DEFINED_[subgroupScanStoreOffset] = INVCONV (VALUE); \
		if (pseudoSubgroupInvocation<irr_glsl_HalfSubgroupSize) \
			_IRR_GLSL_SCRATCH_SHARED_DEFINED_[lastLoadOffset] = INVCONV (IDENTITY); \
	} \
	SUBGROUP_BARRIERS; \
	VALUE = OP (VALUE,CONV (_IRR_GLSL_SCRATCH_SHARED_DEFINED_[subgroupScanStoreOffset-1u])); \
	for (uint stp=irr_glsl_MinSubgroupSize; stp<irr_glsl_HalfSubgroupSize; stp<<=1u) \
	{ \
		SUBGROUP_BARRIERS; \
		_IRR_GLSL_SCRATCH_SHARED_DEFINED_[subgroupScanStoreOffset] = INVCONV (VALUE); \
		SUBGROUP_BARRIERS; \
		VALUE = OP (VALUE,CONV (_IRR_GLSL_SCRATCH_SHARED_DEFINED_[subgroupScanStoreOffset-stp])); \
	} \
	SUBGROUP_BARRIERS; \
	_IRR_GLSL_SCRATCH_SHARED_DEFINED_[subgroupScanStoreOffset] = INVCONV (VALUE); \
	SUBGROUP_BARRIERS; \
	VALUE = OP (VALUE,CONV (_IRR_GLSL_SCRATCH_SHARED_DEFINED_[lastLoadOffset])); \
	SUBGROUP_BARRIERS;


#define IRR_GLSL_SUBGROUP_REDUCE(CONV,OP,VALUE,INITIALIZE,IDENTITY,INVCONV) IRR_GLSL_SUBGROUP_ARITHMETIC_IMPL(CONV,OP,VALUE,INITIALIZE,IDENTITY,INVCONV) \
	_IRR_GLSL_SCRATCH_SHARED_DEFINED_[subgroupScanStoreOffset] = INVCONV (VALUE); \
	SUBGROUP_BARRIERS; \
	uint lastSubgroupInvocation = loMask; \
	if (pseudoSubgroupElectedInvocation==irr_glsl_subgroup_impl_pseudoSubgroupElectedInvocation(loMask,_IRR_GLSL_WORKGROUP_SIZE_-1u)) \
		lastSubgroupInvocation &= _IRR_GLSL_WORKGROUP_SIZE_-1u;\
	const uint lastItem = _IRR_GLSL_SCRATCH_SHARED_DEFINED_[irr_glsl_subgroup_impl_getSubgroupEmulationMemoryStoreOffset(subgroupMemoryStart,lastSubgroupInvocation)]; \
	SUBGROUP_BARRIERS; \
	return CONV (lastItem);



uint irr_glsl_subgroupAnd_impl(in bool clearScratchToIdentity, uint value)
{
	IRR_GLSL_SUBGROUP_REDUCE(irr_glsl_identityFunction,irr_glsl_and,value,clearScratchToIdentity,0xffFFffFFu,irr_glsl_identityFunction);
}
int irr_glsl_subgroupAnd_impl(in bool clearScratchToIdentity, int value)
{
	return int(irr_glsl_subgroupAnd_impl(clearScratchToIdentity,uint(value)));
}
float irr_glsl_subgroupAnd_impl(in bool clearScratchToIdentity, float value)
{
	return uintBitsToFloat(irr_glsl_subgroupAnd_impl(clearScratchToIdentity,floatBitsToUint(value)));
}

uint irr_glsl_subgroupXor_impl(in bool clearScratchToIdentity, uint value)
{
	IRR_GLSL_SUBGROUP_REDUCE(irr_glsl_identityFunction,irr_glsl_xor,value,clearScratchToIdentity,0u,irr_glsl_identityFunction);
}
int irr_glsl_subgroupXor_impl(in bool clearScratchToIdentity, int value)
{
	return int(irr_glsl_subgroupXor_impl(clearScratchToIdentity,uint(value)));
}
float irr_glsl_subgroupXor_impl(in bool clearScratchToIdentity, float value)
{
	return uintBitsToFloat(irr_glsl_subgroupXor_impl(clearScratchToIdentity,floatBitsToUint(value)));
}

uint irr_glsl_subgroupOr_impl(in bool clearScratchToIdentity, uint value)
{
	IRR_GLSL_SUBGROUP_REDUCE(irr_glsl_identityFunction,irr_glsl_or,value,clearScratchToIdentity,0u,irr_glsl_identityFunction);
}
int irr_glsl_subgroupOr_impl(in bool clearScratchToIdentity, int value)
{
	return int(irr_glsl_subgroupOr_impl(clearScratchToIdentity,uint(value)));
}
float irr_glsl_subgroupOr_impl(in bool clearScratchToIdentity, float value)
{
	return uintBitsToFloat(irr_glsl_subgroupOr_impl(clearScratchToIdentity,floatBitsToUint(value)));
}


uint irr_glsl_subgroupAdd_impl(in bool clearScratchToIdentity, uint value)
{
	IRR_GLSL_SUBGROUP_REDUCE(irr_glsl_identityFunction,irr_glsl_add,value,clearScratchToIdentity,0u,irr_glsl_identityFunction);
}
int irr_glsl_subgroupAdd_impl(in bool clearScratchToIdentity, int value)
{
	return int(irr_glsl_subgroupAdd_impl(clearScratchToIdentity,uint(value)));
}
float irr_glsl_subgroupAdd_impl(in bool clearScratchToIdentity, float value)
{
	IRR_GLSL_SUBGROUP_REDUCE(uintBitsToFloat,irr_glsl_add,value,clearScratchToIdentity,0.0,floatBitsToUint);
}

uint irr_glsl_subgroupMul_impl(in bool clearScratchToIdentity, uint value)
{
	IRR_GLSL_SUBGROUP_REDUCE(irr_glsl_identityFunction,irr_glsl_mul,value,clearScratchToIdentity,1u,irr_glsl_identityFunction);
}
int irr_glsl_subgroupMul_impl(in bool clearScratchToIdentity, int value)
{
	return int(irr_glsl_subgroupMul_impl(clearScratchToIdentity,uint(value)));
}
float irr_glsl_subgroupMul_impl(in bool clearScratchToIdentity, float value)
{
	IRR_GLSL_SUBGROUP_REDUCE(uintBitsToFloat,irr_glsl_mul,value,clearScratchToIdentity,1.0,floatBitsToUint); 
}

uint irr_glsl_subgroupMin_impl(in bool clearScratchToIdentity, uint value)
{
	IRR_GLSL_SUBGROUP_REDUCE(irr_glsl_identityFunction,min,value,clearScratchToIdentity,UINT_MAX,irr_glsl_identityFunction);
}
int irr_glsl_subgroupMin_impl(in bool clearScratchToIdentity, int value)
{
	IRR_GLSL_SUBGROUP_REDUCE(int,min,value,clearScratchToIdentity,INT_MAX,uint);
}
float irr_glsl_subgroupMin_impl(in bool clearScratchToIdentity, float value)
{
	IRR_GLSL_SUBGROUP_REDUCE(uintBitsToFloat,min,value,clearScratchToIdentity,FLT_INF,floatBitsToUint); 
}

uint irr_glsl_subgroupMax_impl(in bool clearScratchToIdentity, uint value)
{
	IRR_GLSL_SUBGROUP_REDUCE(irr_glsl_identityFunction,max,value,clearScratchToIdentity,UINT_MIN,irr_glsl_identityFunction);
}
int irr_glsl_subgroupMax_impl(in bool clearScratchToIdentity, int value)
{
	IRR_GLSL_SUBGROUP_REDUCE(int,max,value,clearScratchToIdentity,INT_MIN,uint);
}
float irr_glsl_subgroupMax_impl(in bool clearScratchToIdentity, float value)
{
	IRR_GLSL_SUBGROUP_REDUCE(uintBitsToFloat,max,value,clearScratchToIdentity,-FLT_INF,floatBitsToUint); 
}

#undef IRR_GLSL_SUBGROUP_REDUCE


#define IRR_GLSL_SUBGROUP_INCLUSIVE_SCAN(CONV,OP,VALUE,INITIALIZE,IDENTITY,INVCONV) IRR_GLSL_SUBGROUP_ARITHMETIC_IMPL(CONV,OP,VALUE,INITIALIZE,IDENTITY,INVCONV) \
	return VALUE

#define IRR_GLSL_SUBGROUP_EXCLUSIVE_SCAN(CONV,OP,VALUE,INITIALIZE,IDENTITY,INVCONV) IRR_GLSL_SUBGROUP_ARITHMETIC_IMPL(CONV,OP,VALUE,INITIALIZE,IDENTITY,INVCONV) \
	_IRR_GLSL_SCRATCH_SHARED_DEFINED_[subgroupScanStoreOffset] = INVCONV (VALUE); \
	SUBGROUP_BARRIERS; \
	const uint prevItem = _IRR_GLSL_SCRATCH_SHARED_DEFINED_[subgroupScanStoreOffset-1u]; \
	SUBGROUP_BARRIERS; \
	return CONV (prevItem);


uint irr_glsl_subgroupInclusiveAnd_impl(in bool clearScratchToIdentity, uint value)
{
	IRR_GLSL_SUBGROUP_INCLUSIVE_SCAN(irr_glsl_identityFunction,irr_glsl_and,value,clearScratchToIdentity,0xffFFffFFu,irr_glsl_identityFunction);
}
int irr_glsl_subgroupInclusiveAnd_impl(in bool clearScratchToIdentity, int value)
{
	return int(irr_glsl_subgroupInclusiveAnd_impl(clearScratchToIdentity,uint(value)));
}
float irr_glsl_subgroupInclusiveAnd_impl(in bool clearScratchToIdentity, float value)
{
	return uintBitsToFloat(irr_glsl_subgroupInclusiveAnd_impl(clearScratchToIdentity,floatBitsToUint(value)));
}
uint irr_glsl_subgroupExclusiveAnd_impl(in bool clearScratchToIdentity, uint value)
{
	IRR_GLSL_SUBGROUP_EXCLUSIVE_SCAN(irr_glsl_identityFunction,irr_glsl_and,value,clearScratchToIdentity,0xffFFffFFu,irr_glsl_identityFunction);
}
int irr_glsl_subgroupExclusiveAnd_impl(in bool clearScratchToIdentity, int value)
{
	return int(irr_glsl_subgroupExclusiveAnd_impl(clearScratchToIdentity,uint(value)));
}
float irr_glsl_subgroupExclusiveAnd_impl(in bool clearScratchToIdentity, float value)
{
	return uintBitsToFloat(irr_glsl_subgroupExclusiveAnd_impl(clearScratchToIdentity,floatBitsToUint(value)));
}

uint irr_glsl_subgroupInclusiveXor_impl(in bool clearScratchToIdentity, uint value)
{
	IRR_GLSL_SUBGROUP_INCLUSIVE_SCAN(irr_glsl_identityFunction,irr_glsl_xor,value,clearScratchToIdentity,0u,irr_glsl_identityFunction);
}
int irr_glsl_subgroupInclusiveXor_impl(in bool clearScratchToIdentity, int value)
{
	return int(irr_glsl_subgroupInclusiveXor_impl(clearScratchToIdentity,uint(value)));
}
float irr_glsl_subgroupInclusiveXor_impl(in bool clearScratchToIdentity, float value)
{
	return uintBitsToFloat(irr_glsl_subgroupInclusiveXor_impl(clearScratchToIdentity,floatBitsToUint(value)));
}
uint irr_glsl_subgroupExclusiveXor_impl(in bool clearScratchToIdentity, uint value)
{
	IRR_GLSL_SUBGROUP_EXCLUSIVE_SCAN(irr_glsl_identityFunction,irr_glsl_xor,value,clearScratchToIdentity,0u,irr_glsl_identityFunction);
}
int irr_glsl_subgroupExclusiveXor_impl(in bool clearScratchToIdentity, int value)
{
	return int(irr_glsl_subgroupExclusiveXor_impl(clearScratchToIdentity,uint(value)));
}
float irr_glsl_subgroupExclusiveXor_impl(in bool clearScratchToIdentity, float value)
{
	return uintBitsToFloat(irr_glsl_subgroupExclusiveXor_impl(clearScratchToIdentity,floatBitsToUint(value)));
}

uint irr_glsl_subgroupInclusiveOr_impl(in bool clearScratchToIdentity, uint value)
{
	IRR_GLSL_SUBGROUP_INCLUSIVE_SCAN(irr_glsl_identityFunction,irr_glsl_or,value,clearScratchToIdentity,0u,irr_glsl_identityFunction);
}
int irr_glsl_subgroupInclusiveOr_impl(in bool clearScratchToIdentity, int value)
{
	return int(irr_glsl_subgroupInclusiveOr_impl(clearScratchToIdentity,uint(value)));
}
float irr_glsl_subgroupInclusiveOr_impl(in bool clearScratchToIdentity, float value)
{
	return uintBitsToFloat(irr_glsl_subgroupInclusiveOr_impl(clearScratchToIdentity,floatBitsToUint(value)));
}
uint irr_glsl_subgroupExclusiveOr_impl(in bool clearScratchToIdentity, uint value)
{
	IRR_GLSL_SUBGROUP_EXCLUSIVE_SCAN(irr_glsl_identityFunction,irr_glsl_or,value,clearScratchToIdentity,0u, irr_glsl_identityFunction);
}
int irr_glsl_subgroupExclusiveOr_impl(in bool clearScratchToIdentity, int value)
{
	return int(irr_glsl_subgroupExclusiveOr_impl(clearScratchToIdentity,uint(value)));
}
float irr_glsl_subgroupExclusiveOr_impl(in bool clearScratchToIdentity, float value)
{
	return uintBitsToFloat(irr_glsl_subgroupExclusiveOr_impl(clearScratchToIdentity,floatBitsToUint(value)));
}


uint irr_glsl_subgroupInclusiveAdd_impl(in bool clearScratchToIdentity, uint value)
{
	IRR_GLSL_SUBGROUP_INCLUSIVE_SCAN(irr_glsl_identityFunction,irr_glsl_add,value,clearScratchToIdentity,0u,irr_glsl_identityFunction);
}
int irr_glsl_subgroupInclusiveAdd_impl(in bool clearScratchToIdentity, int value)
{
	return int(irr_glsl_subgroupInclusiveAdd_impl(clearScratchToIdentity,uint(value)));
}
float irr_glsl_subgroupInclusiveAdd_impl(in bool clearScratchToIdentity, float value)
{
	IRR_GLSL_SUBGROUP_INCLUSIVE_SCAN(uintBitsToFloat,irr_glsl_add,value,clearScratchToIdentity,0.0,floatBitsToUint);
}
uint irr_glsl_subgroupExclusiveAdd_impl(in bool clearScratchToIdentity, uint value)
{
	IRR_GLSL_SUBGROUP_EXCLUSIVE_SCAN(irr_glsl_identityFunction,irr_glsl_add,value,clearScratchToIdentity,0u,irr_glsl_identityFunction);
}
int irr_glsl_subgroupExclusiveAdd_impl(in bool clearScratchToIdentity, int value)
{
	return int(irr_glsl_subgroupExclusiveAdd_impl(clearScratchToIdentity,uint(value)));
}
float irr_glsl_subgroupExclusiveAdd_impl(in bool clearScratchToIdentity, float value)
{
	IRR_GLSL_SUBGROUP_EXCLUSIVE_SCAN(uintBitsToFloat,irr_glsl_add,value,clearScratchToIdentity,0.0,floatBitsToUint);
}

uint irr_glsl_subgroupInclusiveMul_impl(in bool clearScratchToIdentity, uint value)
{
	IRR_GLSL_SUBGROUP_INCLUSIVE_SCAN(irr_glsl_identityFunction,irr_glsl_mul,value,clearScratchToIdentity,1u,irr_glsl_identityFunction);
}
int irr_glsl_subgroupInclusiveMul_impl(in bool clearScratchToIdentity, int value)
{
	return int(irr_glsl_subgroupInclusiveMul_impl(clearScratchToIdentity,uint(value)));
}
float irr_glsl_subgroupInclusiveMul_impl(in bool clearScratchToIdentity, float value)
{
	IRR_GLSL_SUBGROUP_INCLUSIVE_SCAN(uintBitsToFloat,irr_glsl_mul,value,clearScratchToIdentity,1.0,floatBitsToUint);
}
uint irr_glsl_subgroupExclusiveMul_impl(in bool clearScratchToIdentity, uint value)
{
	IRR_GLSL_SUBGROUP_EXCLUSIVE_SCAN(irr_glsl_identityFunction,irr_glsl_mul,value,clearScratchToIdentity,1u,irr_glsl_identityFunction);
}
int irr_glsl_subgroupExclusiveMul_impl(in bool clearScratchToIdentity, int value)
{
	return int(irr_glsl_subgroupExclusiveMul_impl(clearScratchToIdentity,uint(value)));
}
float irr_glsl_subgroupExclusiveMul_impl(in bool clearScratchToIdentity, float value)
{
	IRR_GLSL_SUBGROUP_EXCLUSIVE_SCAN(uintBitsToFloat,irr_glsl_mul,value,clearScratchToIdentity,1.0,floatBitsToUint);
}

uint irr_glsl_subgroupInclusiveMin_impl(in bool clearScratchToIdentity, uint value)
{
	IRR_GLSL_SUBGROUP_INCLUSIVE_SCAN(irr_glsl_identityFunction,min,value,clearScratchToIdentity,UINT_MAX,irr_glsl_identityFunction);
}
int irr_glsl_subgroupInclusiveMin_impl(in bool clearScratchToIdentity, int value)
{
	IRR_GLSL_SUBGROUP_INCLUSIVE_SCAN(int,min,value,clearScratchToIdentity,INT_MAX,uint);
}
float irr_glsl_subgroupInclusiveMin_impl(in bool clearScratchToIdentity, float value)
{
	IRR_GLSL_SUBGROUP_INCLUSIVE_SCAN(uintBitsToFloat,min,value,clearScratchToIdentity,FLT_INF,floatBitsToUint);
}
uint irr_glsl_subgroupExclusiveMin_impl(in bool clearScratchToIdentity, uint value)
{
	IRR_GLSL_SUBGROUP_EXCLUSIVE_SCAN(irr_glsl_identityFunction,min,value,clearScratchToIdentity,UINT_MAX,irr_glsl_identityFunction);
}
int irr_glsl_subgroupExclusiveMin_impl(in bool clearScratchToIdentity, int value)
{
	IRR_GLSL_SUBGROUP_EXCLUSIVE_SCAN(int,min,value,clearScratchToIdentity,INT_MAX,uint);
}
float irr_glsl_subgroupExclusiveMin_impl(in bool clearScratchToIdentity, float value)
{
	IRR_GLSL_SUBGROUP_EXCLUSIVE_SCAN(uintBitsToFloat,min,value,clearScratchToIdentity,FLT_INF,floatBitsToUint);
}

uint irr_glsl_subgroupInclusiveMax_impl(in bool clearScratchToIdentity, uint value)
{
	IRR_GLSL_SUBGROUP_INCLUSIVE_SCAN(irr_glsl_identityFunction,max,value,clearScratchToIdentity,UINT_MIN,irr_glsl_identityFunction);
}
int irr_glsl_subgroupInclusiveMax_impl(in bool clearScratchToIdentity, int value)
{
	IRR_GLSL_SUBGROUP_INCLUSIVE_SCAN(int,max,value,clearScratchToIdentity,INT_MIN,uint);
}
float irr_glsl_subgroupInclusiveMax_impl(in bool clearScratchToIdentity, float value)
{
	IRR_GLSL_SUBGROUP_INCLUSIVE_SCAN(uintBitsToFloat,max,value,clearScratchToIdentity,-FLT_INF,floatBitsToUint);
}
uint irr_glsl_subgroupExclusiveMax_impl(in bool clearScratchToIdentity, uint value)
{
	IRR_GLSL_SUBGROUP_EXCLUSIVE_SCAN(irr_glsl_identityFunction,max,value,clearScratchToIdentity,UINT_MIN,irr_glsl_identityFunction);
}
int irr_glsl_subgroupExclusiveMax_impl(in bool clearScratchToIdentity, int value)
{
	IRR_GLSL_SUBGROUP_EXCLUSIVE_SCAN(int,max,value,clearScratchToIdentity,INT_MIN,uint);
}
float irr_glsl_subgroupExclusiveMax_impl(in bool clearScratchToIdentity, float value)
{
	IRR_GLSL_SUBGROUP_EXCLUSIVE_SCAN(uintBitsToFloat,max,value,clearScratchToIdentity,-FLT_INF,floatBitsToUint);
}

#undef IRR_GLSL_SUBGROUP_INCLUSIVE_SCAN
#undef IRR_GLSL_SUBGROUP_EXCLUSIVE_SCAN


#undef IRR_GLSL_SUBGROUP_ARITHMETIC_IMPL



#undef SUBGROUP_BARRIERS



#endif
