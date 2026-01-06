#ifndef _NBL_BUILTIN_GLSL_SUBGROUP_ARITHMETIC_PORTABILITY_IMPLEMENTATION_INCLUDED_
#define _NBL_BUILTIN_GLSL_SUBGROUP_ARITHMETIC_PORTABILITY_IMPLEMENTATION_INCLUDED_


/*
How to avoid bank conflicts:
read:	00,01,02,03,    08,09,10,11,	16,17,18,19,    24,25,26,27,    04,05,06,07,    12,13,14,15,    20,21,22,23,    28,29,30,31
write:	30,31,00,01,    06,07,08,09,    14,15,16,17,    22,23,24,25,    02,03,04,05,    10,11,12,13,    18,19,20,21,    26,27,28,29

This design should also work for workgroups that are not divisible by subgroup size, but only if we could initialize the _NBL_GLSL_SCRATCH_SHARED_DEFINED_ reliably.

TODO: add [[unroll]] to all loops
*/
uint nbl_glsl_subgroup_impl_pseudoSubgroupElectedInvocation(in uint loMask, in uint invocationIndex)
{
	return invocationIndex&(~loMask);
}
uint nbl_glsl_subgroup_impl_pseudoSubgroupInvocation(in uint loMask, in uint invocationIndex)
{
	return invocationIndex&loMask;
}
uint nbl_glsl_subgroup_impl_getSubgroupEmulationMemoryStart(in uint pseudoSubgroupElectedInvocation)
{
	return pseudoSubgroupElectedInvocation<<1u;
}
uint nbl_glsl_subgroup_impl_getSubgroupEmulationMemoryStoreOffset(in uint subgroupMemoryStart, in uint pseudoSubgroupInvocation, out uint lastLoadOffset)
{
	lastLoadOffset = (subgroupMemoryStart|pseudoSubgroupInvocation);
	return lastLoadOffset+nbl_glsl_HalfSubgroupSize;
}
uint nbl_glsl_subgroup_impl_getSubgroupEmulationMemoryStoreOffset(in uint subgroupMemoryStart, in uint pseudoSubgroupInvocation)
{
	uint dummy;
	return nbl_glsl_subgroup_impl_getSubgroupEmulationMemoryStoreOffset(subgroupMemoryStart,pseudoSubgroupInvocation,dummy);
}
uint nbl_glsl_subgroup_getSubgroupEmulationMemoryStoreOffset(in uint loMask, in uint invocationIndex)
{
	return nbl_glsl_subgroup_impl_getSubgroupEmulationMemoryStoreOffset(
		nbl_glsl_subgroup_impl_getSubgroupEmulationMemoryStart(
			nbl_glsl_subgroup_impl_pseudoSubgroupElectedInvocation(loMask,invocationIndex)
		),
		nbl_glsl_subgroup_impl_pseudoSubgroupInvocation(loMask,invocationIndex)
	);
}

#define SUBGROUP_SCRATCH_OFFSETS_AND_MASKS const uint loMask = nbl_glsl_SubgroupSize-1u; \
	const uint pseudoSubgroupElectedInvocation = nbl_glsl_subgroup_impl_pseudoSubgroupElectedInvocation(loMask,gl_LocalInvocationIndex); \
	const uint pseudoSubgroupInvocation = nbl_glsl_subgroup_impl_pseudoSubgroupInvocation(loMask,gl_LocalInvocationIndex); \
	const uint subgroupMemoryStart = nbl_glsl_subgroup_impl_getSubgroupEmulationMemoryStart(pseudoSubgroupElectedInvocation); \
	uint lastLoadOffset = 0xdeadbeefu; \
	const uint subgroupScanStoreOffset = nbl_glsl_subgroup_impl_getSubgroupEmulationMemoryStoreOffset(subgroupMemoryStart,pseudoSubgroupInvocation,lastLoadOffset)


#define SUBGROUP_SCRATCH_INITIALIZE_IMPL_CLEAR_INDEX(PSEUDO_INVOCATION) ((((PSEUDO_INVOCATION)&(~halfMask))<<2u)|((PSEUDO_INVOCATION)&halfMask))

#define SUBGROUP_SCRATCH_INITIALIZE(VALUE,ACTIVE_INVOCATION_INDEX_UPPER_BOUND,IDENTITY,INVCONV) SUBGROUP_SCRATCH_OFFSETS_AND_MASKS; \
	{ \
		_NBL_GLSL_SCRATCH_SHARED_DEFINED_[subgroupScanStoreOffset] = INVCONV (VALUE); \
		const uint halfMask = loMask>>1u; \
		_NBL_GLSL_SCRATCH_SHARED_DEFINED_[SUBGROUP_SCRATCH_INITIALIZE_IMPL_CLEAR_INDEX(gl_LocalInvocationIndex)] = INVCONV (IDENTITY); \
		if (_NBL_GLSL_WORKGROUP_SIZE_<nbl_glsl_HalfSubgroupSize) \
		{ \
			const uint maxItemsToClear = (nbl_glsl_subgroup_impl_pseudoSubgroupElectedInvocation(loMask,ACTIVE_INVOCATION_INDEX_UPPER_BOUND-1u)>>1u)+nbl_glsl_HalfSubgroupSize; \
			for (uint ix=gl_LocalInvocationIndex+_NBL_GLSL_WORKGROUP_SIZE_; ix<maxItemsToClear; ix+=_NBL_GLSL_WORKGROUP_SIZE_) \
				_NBL_GLSL_SCRATCH_SHARED_DEFINED_[SUBGROUP_SCRATCH_INITIALIZE_IMPL_CLEAR_INDEX(ix)] = INVCONV (IDENTITY); \
		} \
		barrier(); \
	}


#define NBL_GLSL_SUBGROUP_ARITHMETIC_IMPL(CONV,OP,VALUE,INITIALIZE,IDENTITY,INVCONV) SUBGROUP_SCRATCH_OFFSETS_AND_MASKS; \
	if (INITIALIZE) \
	{ \
		nbl_glsl_subgroupBarrier(); \
		nbl_glsl_subgroupMemoryBarrierShared(); \
		_NBL_GLSL_SCRATCH_SHARED_DEFINED_[subgroupScanStoreOffset] = INVCONV (VALUE); \
		if (pseudoSubgroupInvocation<nbl_glsl_HalfSubgroupSize) \
			_NBL_GLSL_SCRATCH_SHARED_DEFINED_[lastLoadOffset] = INVCONV (IDENTITY); \
	} \
	nbl_glsl_subgroupBarrier(); \
	nbl_glsl_subgroupMemoryBarrierShared(); \
	VALUE = OP (VALUE,CONV (_NBL_GLSL_SCRATCH_SHARED_DEFINED_[subgroupScanStoreOffset-1u])); \
	for (uint stp=2u; stp<nbl_glsl_HalfSubgroupSize; stp<<=1u) \
	{ \
		nbl_glsl_subgroupBarrier(); \
		nbl_glsl_subgroupMemoryBarrierShared(); \
		_NBL_GLSL_SCRATCH_SHARED_DEFINED_[subgroupScanStoreOffset] = INVCONV (VALUE); \
		nbl_glsl_subgroupBarrier(); \
		nbl_glsl_subgroupMemoryBarrierShared(); \
		VALUE = OP (VALUE,CONV (_NBL_GLSL_SCRATCH_SHARED_DEFINED_[subgroupScanStoreOffset-stp])); \
	} \
	nbl_glsl_subgroupBarrier(); \
	nbl_glsl_subgroupMemoryBarrierShared(); \
	_NBL_GLSL_SCRATCH_SHARED_DEFINED_[subgroupScanStoreOffset] = INVCONV (VALUE); \
	nbl_glsl_subgroupBarrier(); \
	nbl_glsl_subgroupMemoryBarrierShared(); \
	VALUE = OP (VALUE,CONV (_NBL_GLSL_SCRATCH_SHARED_DEFINED_[lastLoadOffset])); \
	nbl_glsl_subgroupBarrier(); \
	nbl_glsl_subgroupMemoryBarrierShared();


#define NBL_GLSL_SUBGROUP_REDUCE(CONV,OP,VALUE,INITIALIZE,IDENTITY,INVCONV) NBL_GLSL_SUBGROUP_ARITHMETIC_IMPL(CONV,OP,VALUE,INITIALIZE,IDENTITY,INVCONV) \
	_NBL_GLSL_SCRATCH_SHARED_DEFINED_[subgroupScanStoreOffset] = INVCONV (VALUE); \
	nbl_glsl_subgroupBarrier(); \
	nbl_glsl_subgroupMemoryBarrierShared(); \
	uint lastSubgroupInvocation = loMask; \
	if (pseudoSubgroupElectedInvocation==nbl_glsl_subgroup_impl_pseudoSubgroupElectedInvocation(loMask,_NBL_GLSL_WORKGROUP_SIZE_-1u)) \
		lastSubgroupInvocation &= _NBL_GLSL_WORKGROUP_SIZE_-1u;\
	const uint lastItem = _NBL_GLSL_SCRATCH_SHARED_DEFINED_[nbl_glsl_subgroup_impl_getSubgroupEmulationMemoryStoreOffset(subgroupMemoryStart,lastSubgroupInvocation)]; \
	nbl_glsl_subgroupBarrier(); \
	nbl_glsl_subgroupMemoryBarrierShared(); \
	return CONV (lastItem);



uint nbl_glsl_subgroupAnd_impl(in bool clearScratchToIdentity, uint value)
{
	NBL_GLSL_SUBGROUP_REDUCE(nbl_glsl_identityFunction,nbl_glsl_and,value,clearScratchToIdentity,0xffFFffFFu,nbl_glsl_identityFunction);
}
int nbl_glsl_subgroupAnd_impl(in bool clearScratchToIdentity, int value)
{
	return int(nbl_glsl_subgroupAnd_impl(clearScratchToIdentity,uint(value)));
}
float nbl_glsl_subgroupAnd_impl(in bool clearScratchToIdentity, float value)
{
	return uintBitsToFloat(nbl_glsl_subgroupAnd_impl(clearScratchToIdentity,floatBitsToUint(value)));
}

uint nbl_glsl_subgroupXor_impl(in bool clearScratchToIdentity, uint value)
{
	NBL_GLSL_SUBGROUP_REDUCE(nbl_glsl_identityFunction,nbl_glsl_xor,value,clearScratchToIdentity,0u,nbl_glsl_identityFunction);
}
int nbl_glsl_subgroupXor_impl(in bool clearScratchToIdentity, int value)
{
	return int(nbl_glsl_subgroupXor_impl(clearScratchToIdentity,uint(value)));
}
float nbl_glsl_subgroupXor_impl(in bool clearScratchToIdentity, float value)
{
	return uintBitsToFloat(nbl_glsl_subgroupXor_impl(clearScratchToIdentity,floatBitsToUint(value)));
}

uint nbl_glsl_subgroupOr_impl(in bool clearScratchToIdentity, uint value)
{
	NBL_GLSL_SUBGROUP_REDUCE(nbl_glsl_identityFunction,nbl_glsl_or,value,clearScratchToIdentity,0u,nbl_glsl_identityFunction);
}
int nbl_glsl_subgroupOr_impl(in bool clearScratchToIdentity, int value)
{
	return int(nbl_glsl_subgroupOr_impl(clearScratchToIdentity,uint(value)));
}
float nbl_glsl_subgroupOr_impl(in bool clearScratchToIdentity, float value)
{
	return uintBitsToFloat(nbl_glsl_subgroupOr_impl(clearScratchToIdentity,floatBitsToUint(value)));
}


uint nbl_glsl_subgroupAdd_impl(in bool clearScratchToIdentity, uint value)
{
	NBL_GLSL_SUBGROUP_REDUCE(nbl_glsl_identityFunction,nbl_glsl_add,value,clearScratchToIdentity,0u,nbl_glsl_identityFunction);
}
int nbl_glsl_subgroupAdd_impl(in bool clearScratchToIdentity, int value)
{
	return int(nbl_glsl_subgroupAdd_impl(clearScratchToIdentity,uint(value)));
}
float nbl_glsl_subgroupAdd_impl(in bool clearScratchToIdentity, float value)
{
	NBL_GLSL_SUBGROUP_REDUCE(uintBitsToFloat,nbl_glsl_add,value,clearScratchToIdentity,0.0,floatBitsToUint);
}

uint nbl_glsl_subgroupMul_impl(in bool clearScratchToIdentity, uint value)
{
	NBL_GLSL_SUBGROUP_REDUCE(nbl_glsl_identityFunction,nbl_glsl_mul,value,clearScratchToIdentity,1u,nbl_glsl_identityFunction);
}
int nbl_glsl_subgroupMul_impl(in bool clearScratchToIdentity, int value)
{
	return int(nbl_glsl_subgroupMul_impl(clearScratchToIdentity,uint(value)));
}
float nbl_glsl_subgroupMul_impl(in bool clearScratchToIdentity, float value)
{
	NBL_GLSL_SUBGROUP_REDUCE(uintBitsToFloat,nbl_glsl_mul,value,clearScratchToIdentity,1.0,floatBitsToUint); 
}

uint nbl_glsl_subgroupMin_impl(in bool clearScratchToIdentity, uint value)
{
	NBL_GLSL_SUBGROUP_REDUCE(nbl_glsl_identityFunction,min,value,clearScratchToIdentity,nbl_glsl_UINT_MAX,nbl_glsl_identityFunction);
}
int nbl_glsl_subgroupMin_impl(in bool clearScratchToIdentity, int value)
{
	NBL_GLSL_SUBGROUP_REDUCE(int,min,value,clearScratchToIdentity,nbl_glsl_INT_MAX,uint);
}
float nbl_glsl_subgroupMin_impl(in bool clearScratchToIdentity, float value)
{
	NBL_GLSL_SUBGROUP_REDUCE(uintBitsToFloat,min,value,clearScratchToIdentity,nbl_glsl_FLT_INF,floatBitsToUint); 
}

uint nbl_glsl_subgroupMax_impl(in bool clearScratchToIdentity, uint value)
{
	NBL_GLSL_SUBGROUP_REDUCE(nbl_glsl_identityFunction,max,value,clearScratchToIdentity,nbl_glsl_UINT_MIN,nbl_glsl_identityFunction);
}
int nbl_glsl_subgroupMax_impl(in bool clearScratchToIdentity, int value)
{
	NBL_GLSL_SUBGROUP_REDUCE(int,max,value,clearScratchToIdentity,nbl_glsl_INT_MIN,uint);
}
float nbl_glsl_subgroupMax_impl(in bool clearScratchToIdentity, float value)
{
	NBL_GLSL_SUBGROUP_REDUCE(uintBitsToFloat,max,value,clearScratchToIdentity,-nbl_glsl_FLT_INF,floatBitsToUint); 
}

#undef NBL_GLSL_SUBGROUP_REDUCE


#define NBL_GLSL_SUBGROUP_INCLUSIVE_SCAN(CONV,OP,VALUE,INITIALIZE,IDENTITY,INVCONV) NBL_GLSL_SUBGROUP_ARITHMETIC_IMPL(CONV,OP,VALUE,INITIALIZE,IDENTITY,INVCONV) \
	return VALUE

#define NBL_GLSL_SUBGROUP_EXCLUSIVE_SCAN(CONV,OP,VALUE,INITIALIZE,IDENTITY,INVCONV) NBL_GLSL_SUBGROUP_ARITHMETIC_IMPL(CONV,OP,VALUE,INITIALIZE,IDENTITY,INVCONV) \
	_NBL_GLSL_SCRATCH_SHARED_DEFINED_[subgroupScanStoreOffset] = INVCONV (VALUE); \
	nbl_glsl_subgroupBarrier(); \
	nbl_glsl_subgroupMemoryBarrierShared(); \
	const uint prevItem = _NBL_GLSL_SCRATCH_SHARED_DEFINED_[subgroupScanStoreOffset-1u]; \
	nbl_glsl_subgroupBarrier(); \
	nbl_glsl_subgroupMemoryBarrierShared(); \
	return CONV (prevItem);


uint nbl_glsl_subgroupInclusiveAnd_impl(in bool clearScratchToIdentity, uint value)
{
	NBL_GLSL_SUBGROUP_INCLUSIVE_SCAN(nbl_glsl_identityFunction,nbl_glsl_and,value,clearScratchToIdentity,0xffFFffFFu,nbl_glsl_identityFunction);
}
int nbl_glsl_subgroupInclusiveAnd_impl(in bool clearScratchToIdentity, int value)
{
	return int(nbl_glsl_subgroupInclusiveAnd_impl(clearScratchToIdentity,uint(value)));
}
float nbl_glsl_subgroupInclusiveAnd_impl(in bool clearScratchToIdentity, float value)
{
	return uintBitsToFloat(nbl_glsl_subgroupInclusiveAnd_impl(clearScratchToIdentity,floatBitsToUint(value)));
}
uint nbl_glsl_subgroupExclusiveAnd_impl(in bool clearScratchToIdentity, uint value)
{
	NBL_GLSL_SUBGROUP_EXCLUSIVE_SCAN(nbl_glsl_identityFunction,nbl_glsl_and,value,clearScratchToIdentity,0xffFFffFFu,nbl_glsl_identityFunction);
}
int nbl_glsl_subgroupExclusiveAnd_impl(in bool clearScratchToIdentity, int value)
{
	return int(nbl_glsl_subgroupExclusiveAnd_impl(clearScratchToIdentity,uint(value)));
}
float nbl_glsl_subgroupExclusiveAnd_impl(in bool clearScratchToIdentity, float value)
{
	return uintBitsToFloat(nbl_glsl_subgroupExclusiveAnd_impl(clearScratchToIdentity,floatBitsToUint(value)));
}

uint nbl_glsl_subgroupInclusiveXor_impl(in bool clearScratchToIdentity, uint value)
{
	NBL_GLSL_SUBGROUP_INCLUSIVE_SCAN(nbl_glsl_identityFunction,nbl_glsl_xor,value,clearScratchToIdentity,0u,nbl_glsl_identityFunction);
}
int nbl_glsl_subgroupInclusiveXor_impl(in bool clearScratchToIdentity, int value)
{
	return int(nbl_glsl_subgroupInclusiveXor_impl(clearScratchToIdentity,uint(value)));
}
float nbl_glsl_subgroupInclusiveXor_impl(in bool clearScratchToIdentity, float value)
{
	return uintBitsToFloat(nbl_glsl_subgroupInclusiveXor_impl(clearScratchToIdentity,floatBitsToUint(value)));
}
uint nbl_glsl_subgroupExclusiveXor_impl(in bool clearScratchToIdentity, uint value)
{
	NBL_GLSL_SUBGROUP_EXCLUSIVE_SCAN(nbl_glsl_identityFunction,nbl_glsl_xor,value,clearScratchToIdentity,0u,nbl_glsl_identityFunction);
}
int nbl_glsl_subgroupExclusiveXor_impl(in bool clearScratchToIdentity, int value)
{
	return int(nbl_glsl_subgroupExclusiveXor_impl(clearScratchToIdentity,uint(value)));
}
float nbl_glsl_subgroupExclusiveXor_impl(in bool clearScratchToIdentity, float value)
{
	return uintBitsToFloat(nbl_glsl_subgroupExclusiveXor_impl(clearScratchToIdentity,floatBitsToUint(value)));
}

uint nbl_glsl_subgroupInclusiveOr_impl(in bool clearScratchToIdentity, uint value)
{
	NBL_GLSL_SUBGROUP_INCLUSIVE_SCAN(nbl_glsl_identityFunction,nbl_glsl_or,value,clearScratchToIdentity,0u,nbl_glsl_identityFunction);
}
int nbl_glsl_subgroupInclusiveOr_impl(in bool clearScratchToIdentity, int value)
{
	return int(nbl_glsl_subgroupInclusiveOr_impl(clearScratchToIdentity,uint(value)));
}
float nbl_glsl_subgroupInclusiveOr_impl(in bool clearScratchToIdentity, float value)
{
	return uintBitsToFloat(nbl_glsl_subgroupInclusiveOr_impl(clearScratchToIdentity,floatBitsToUint(value)));
}
uint nbl_glsl_subgroupExclusiveOr_impl(in bool clearScratchToIdentity, uint value)
{
	NBL_GLSL_SUBGROUP_EXCLUSIVE_SCAN(nbl_glsl_identityFunction,nbl_glsl_or,value,clearScratchToIdentity,0u, nbl_glsl_identityFunction);
}
int nbl_glsl_subgroupExclusiveOr_impl(in bool clearScratchToIdentity, int value)
{
	return int(nbl_glsl_subgroupExclusiveOr_impl(clearScratchToIdentity,uint(value)));
}
float nbl_glsl_subgroupExclusiveOr_impl(in bool clearScratchToIdentity, float value)
{
	return uintBitsToFloat(nbl_glsl_subgroupExclusiveOr_impl(clearScratchToIdentity,floatBitsToUint(value)));
}


uint nbl_glsl_subgroupInclusiveAdd_impl(in bool clearScratchToIdentity, uint value)
{
	NBL_GLSL_SUBGROUP_INCLUSIVE_SCAN(nbl_glsl_identityFunction,nbl_glsl_add,value,clearScratchToIdentity,0u,nbl_glsl_identityFunction);
}
int nbl_glsl_subgroupInclusiveAdd_impl(in bool clearScratchToIdentity, int value)
{
	return int(nbl_glsl_subgroupInclusiveAdd_impl(clearScratchToIdentity,uint(value)));
}
float nbl_glsl_subgroupInclusiveAdd_impl(in bool clearScratchToIdentity, float value)
{
	NBL_GLSL_SUBGROUP_INCLUSIVE_SCAN(uintBitsToFloat,nbl_glsl_add,value,clearScratchToIdentity,0.0,floatBitsToUint);
}
uint nbl_glsl_subgroupExclusiveAdd_impl(in bool clearScratchToIdentity, uint value)
{
	NBL_GLSL_SUBGROUP_EXCLUSIVE_SCAN(nbl_glsl_identityFunction,nbl_glsl_add,value,clearScratchToIdentity,0u,nbl_glsl_identityFunction);
}
int nbl_glsl_subgroupExclusiveAdd_impl(in bool clearScratchToIdentity, int value)
{
	return int(nbl_glsl_subgroupExclusiveAdd_impl(clearScratchToIdentity,uint(value)));
}
float nbl_glsl_subgroupExclusiveAdd_impl(in bool clearScratchToIdentity, float value)
{
	NBL_GLSL_SUBGROUP_EXCLUSIVE_SCAN(uintBitsToFloat,nbl_glsl_add,value,clearScratchToIdentity,0.0,floatBitsToUint);
}

uint nbl_glsl_subgroupInclusiveMul_impl(in bool clearScratchToIdentity, uint value)
{
	NBL_GLSL_SUBGROUP_INCLUSIVE_SCAN(nbl_glsl_identityFunction,nbl_glsl_mul,value,clearScratchToIdentity,1u,nbl_glsl_identityFunction);
}
int nbl_glsl_subgroupInclusiveMul_impl(in bool clearScratchToIdentity, int value)
{
	return int(nbl_glsl_subgroupInclusiveMul_impl(clearScratchToIdentity,uint(value)));
}
float nbl_glsl_subgroupInclusiveMul_impl(in bool clearScratchToIdentity, float value)
{
	NBL_GLSL_SUBGROUP_INCLUSIVE_SCAN(uintBitsToFloat,nbl_glsl_mul,value,clearScratchToIdentity,1.0,floatBitsToUint);
}
uint nbl_glsl_subgroupExclusiveMul_impl(in bool clearScratchToIdentity, uint value)
{
	NBL_GLSL_SUBGROUP_EXCLUSIVE_SCAN(nbl_glsl_identityFunction,nbl_glsl_mul,value,clearScratchToIdentity,1u,nbl_glsl_identityFunction);
}
int nbl_glsl_subgroupExclusiveMul_impl(in bool clearScratchToIdentity, int value)
{
	return int(nbl_glsl_subgroupExclusiveMul_impl(clearScratchToIdentity,uint(value)));
}
float nbl_glsl_subgroupExclusiveMul_impl(in bool clearScratchToIdentity, float value)
{
	NBL_GLSL_SUBGROUP_EXCLUSIVE_SCAN(uintBitsToFloat,nbl_glsl_mul,value,clearScratchToIdentity,1.0,floatBitsToUint);
}

uint nbl_glsl_subgroupInclusiveMin_impl(in bool clearScratchToIdentity, uint value)
{
	NBL_GLSL_SUBGROUP_INCLUSIVE_SCAN(nbl_glsl_identityFunction,min,value,clearScratchToIdentity,nbl_glsl_UINT_MAX,nbl_glsl_identityFunction);
}
int nbl_glsl_subgroupInclusiveMin_impl(in bool clearScratchToIdentity, int value)
{
	NBL_GLSL_SUBGROUP_INCLUSIVE_SCAN(int,min,value,clearScratchToIdentity,nbl_glsl_INT_MAX,uint);
}
float nbl_glsl_subgroupInclusiveMin_impl(in bool clearScratchToIdentity, float value)
{
	NBL_GLSL_SUBGROUP_INCLUSIVE_SCAN(uintBitsToFloat,min,value,clearScratchToIdentity,nbl_glsl_FLT_INF,floatBitsToUint);
}
uint nbl_glsl_subgroupExclusiveMin_impl(in bool clearScratchToIdentity, uint value)
{
	NBL_GLSL_SUBGROUP_EXCLUSIVE_SCAN(nbl_glsl_identityFunction,min,value,clearScratchToIdentity,nbl_glsl_UINT_MAX,nbl_glsl_identityFunction);
}
int nbl_glsl_subgroupExclusiveMin_impl(in bool clearScratchToIdentity, int value)
{
	NBL_GLSL_SUBGROUP_EXCLUSIVE_SCAN(int,min,value,clearScratchToIdentity,nbl_glsl_INT_MAX,uint);
}
float nbl_glsl_subgroupExclusiveMin_impl(in bool clearScratchToIdentity, float value)
{
	NBL_GLSL_SUBGROUP_EXCLUSIVE_SCAN(uintBitsToFloat,min,value,clearScratchToIdentity,nbl_glsl_FLT_INF,floatBitsToUint);
}

uint nbl_glsl_subgroupInclusiveMax_impl(in bool clearScratchToIdentity, uint value)
{
	NBL_GLSL_SUBGROUP_INCLUSIVE_SCAN(nbl_glsl_identityFunction,max,value,clearScratchToIdentity,nbl_glsl_UINT_MIN,nbl_glsl_identityFunction);
}
int nbl_glsl_subgroupInclusiveMax_impl(in bool clearScratchToIdentity, int value)
{
	NBL_GLSL_SUBGROUP_INCLUSIVE_SCAN(int,max,value,clearScratchToIdentity,nbl_glsl_INT_MIN,uint);
}
float nbl_glsl_subgroupInclusiveMax_impl(in bool clearScratchToIdentity, float value)
{
	NBL_GLSL_SUBGROUP_INCLUSIVE_SCAN(uintBitsToFloat,max,value,clearScratchToIdentity,-nbl_glsl_FLT_INF,floatBitsToUint);
}
uint nbl_glsl_subgroupExclusiveMax_impl(in bool clearScratchToIdentity, uint value)
{
	NBL_GLSL_SUBGROUP_EXCLUSIVE_SCAN(nbl_glsl_identityFunction,max,value,clearScratchToIdentity,nbl_glsl_UINT_MIN,nbl_glsl_identityFunction);
}
int nbl_glsl_subgroupExclusiveMax_impl(in bool clearScratchToIdentity, int value)
{
	NBL_GLSL_SUBGROUP_EXCLUSIVE_SCAN(int,max,value,clearScratchToIdentity,nbl_glsl_INT_MIN,uint);
}
float nbl_glsl_subgroupExclusiveMax_impl(in bool clearScratchToIdentity, float value)
{
	NBL_GLSL_SUBGROUP_EXCLUSIVE_SCAN(uintBitsToFloat,max,value,clearScratchToIdentity,-nbl_glsl_FLT_INF,floatBitsToUint);
}

#undef NBL_GLSL_SUBGROUP_INCLUSIVE_SCAN
#undef NBL_GLSL_SUBGROUP_EXCLUSIVE_SCAN


#undef NBL_GLSL_SUBGROUP_ARITHMETIC_IMPL



#endif
