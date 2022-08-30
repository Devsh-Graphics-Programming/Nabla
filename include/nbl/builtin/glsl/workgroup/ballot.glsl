#ifndef _NBL_BUILTIN_GLSL_WORKGROUP_BALLOT_INCLUDED_
#define _NBL_BUILTIN_GLSL_WORKGROUP_BALLOT_INCLUDED_



#include <nbl/builtin/glsl/workgroup/shared_ballot.glsl>


/*
#ifdef GL_KHR_subgroup_arithmetic


#define CONDITIONAL_BARRIER

// just do nothing here
#define SUBGROUP_SCRATCH_INITIALIZE(IDENTITY) ;


#else
*/

#define CONDITIONAL_BARRIER barrier();

/*
If `GL_KHR_subgroup_arithmetic` is not available then these functions require emulated subgroup operations, which in turn means that if you're using the
`nbl_glsl_workgroupOp`s then the workgroup size must not be smaller than half a subgroup but having workgroups smaller than a subgroup is extremely bad practice.
*/

//#endif



#ifdef _NBL_GLSL_SCRATCH_SHARED_DEFINED_
	#if NBL_GLSL_LESS(_NBL_GLSL_SCRATCH_SHARED_SIZE_DEFINED_,_NBL_GLSL_WORKGROUP_BALLOT_SHARED_SIZE_NEEDED_)
		#error "Not enough shared memory declared for workgroup ballot!"
	#endif
#else
	#define _NBL_GLSL_SCRATCH_SHARED_DEFINED_ nbl_glsl_workgroupBallotScratchShared
	#define _NBL_GLSL_SCRATCH_SHARED_SIZE_DEFINED_ _NBL_GLSL_WORKGROUP_BALLOT_SHARED_SIZE_NEEDED_
	shared uint _NBL_GLSL_SCRATCH_SHARED_DEFINED_[_NBL_GLSL_WORKGROUP_BALLOT_SHARED_SIZE_NEEDED_];
#endif



#include <nbl/builtin/glsl/subgroup/arithmetic_portability_impl.glsl>



// puts the result into shared memory at offsets [0,_NBL_GLSL_WORKGROUP_SIZE_/32)
void nbl_glsl_workgroupBallot_noBarriers(in bool value)
{
	// TODO: Optimization using subgroupBallot in an ifdef NBL_GL_something (need to do feature mapping first)
	if (gl_LocalInvocationIndex<nbl_glsl_workgroupBallot_impl_BitfieldDWORDs)
		_NBL_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex] = 0u;
	barrier();
	if (value)
		atomicOr(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[nbl_glsl_workgroupBallot_impl_getDWORD(gl_LocalInvocationIndex)],1u<<(gl_LocalInvocationIndex&31u));
}
void nbl_glsl_workgroupBallot(in bool value)
{
	barrier();
	nbl_glsl_workgroupBallot_noBarriers(value);
	barrier();
}

// the ballot is expected to be in _NBL_GLSL_SCRATCH_SHARED_DEFINED_ at offsets [0,_NBL_GLSL_WORKGROUP_SIZE_/32)
bool nbl_glsl_workgroupBallotBitExtract_noEndBarriers(in uint index)
{
	return (_NBL_GLSL_SCRATCH_SHARED_DEFINED_[nbl_glsl_workgroupBallot_impl_getDWORD(index)]&(1u<<(index&31u)))!=0u;
}
bool nbl_glsl_workgroupBallotBitExtract(in uint index)
{
	barrier();
	const bool retval = nbl_glsl_workgroupBallotBitExtract_noEndBarriers(index);
	barrier();
	return retval;
}

bool nbl_glsl_workgroupInverseBallot_noEndBarriers()
{
	return nbl_glsl_workgroupBallotBitExtract_noEndBarriers(gl_LocalInvocationIndex);
}
bool nbl_glsl_workgroupInverseBallot()
{
	return nbl_glsl_workgroupBallotBitExtract(gl_LocalInvocationIndex);
}


uint nbl_glsl_workgroupBallotBitCount_noEndBarriers()
{
	_NBL_GLSL_SCRATCH_SHARED_DEFINED_[nbl_glsl_workgroupBallot_impl_BitfieldDWORDs] = 0u;
	barrier();
	if (gl_LocalInvocationIndex<nbl_glsl_workgroupBallot_impl_BitfieldDWORDs)
	{
		const uint localBallot = _NBL_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex];
		const uint localBallotBitCount = bitCount(localBallot);
		atomicAdd(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[nbl_glsl_workgroupBallot_impl_BitfieldDWORDs],localBallotBitCount);
	}
	barrier();

	return _NBL_GLSL_SCRATCH_SHARED_DEFINED_[nbl_glsl_workgroupBallot_impl_BitfieldDWORDs];
}
uint nbl_glsl_workgroupBallotBitCount()
{
	barrier();
	const uint retval = nbl_glsl_workgroupBallotBitCount_noEndBarriers();
	barrier();
	return retval;
}

#define NBL_GLSL_WORKGROUP_BROADCAST(CONV, INVCONV) if (gl_LocalInvocationIndex==id)\
	_NBL_GLSL_SCRATCH_SHARED_DEFINED_[nbl_glsl_workgroupBallot_impl_BitfieldDWORDs] = CONV(val);\
	barrier();\
	return INVCONV(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[nbl_glsl_workgroupBallot_impl_BitfieldDWORDs]);

uint nbl_glsl_workgroupBroadcast_noBarriers(in uint val, in uint id)
{
	NBL_GLSL_WORKGROUP_BROADCAST(nbl_glsl_identityFunction, nbl_glsl_identityFunction)
}

bool nbl_glsl_workgroupBroadcast_noBarriers(in bool val, in uint id)
{
	NBL_GLSL_WORKGROUP_BROADCAST(uint, bool)
}

float nbl_glsl_workgroupBroadcast_noBarriers(in float val, in uint id)
{
	NBL_GLSL_WORKGROUP_BROADCAST(floatBitsToUint, uintBitsToFloat)
}

int nbl_glsl_workgroupBroadcast_noBarriers(in int val, in uint id)
{
	NBL_GLSL_WORKGROUP_BROADCAST(uint, int)
}

#define DECLARE_WORKGROUP_BROADCAST_OVERLOAD_WITH_BARRIERS(TYPE,FUNC_NAME) TYPE nbl_glsl_##FUNC_NAME (in TYPE val, in uint id) \
{ \
	barrier(); \
	const TYPE retval = nbl_glsl_##FUNC_NAME##_noBarriers (val, id); \
	barrier(); \
	return retval; \
}

DECLARE_WORKGROUP_BROADCAST_OVERLOAD_WITH_BARRIERS(uint, workgroupBroadcast)
DECLARE_WORKGROUP_BROADCAST_OVERLOAD_WITH_BARRIERS(bool, workgroupBroadcast)
DECLARE_WORKGROUP_BROADCAST_OVERLOAD_WITH_BARRIERS(float, workgroupBroadcast)
DECLARE_WORKGROUP_BROADCAST_OVERLOAD_WITH_BARRIERS(int, workgroupBroadcast)

uint nbl_glsl_workgroupBroadcastFirst_noBarriers(in uint val)
{
	if (nbl_glsl_workgroupElect())
		_NBL_GLSL_SCRATCH_SHARED_DEFINED_[nbl_glsl_workgroupBallot_impl_BitfieldDWORDs] = val;
	barrier();
	return _NBL_GLSL_SCRATCH_SHARED_DEFINED_[nbl_glsl_workgroupBallot_impl_BitfieldDWORDs];
}
uint nbl_glsl_workgroupBroadcastFirst(in uint val)
{
	barrier();
	const uint retval = nbl_glsl_workgroupBroadcastFirst_noBarriers(val);
	barrier();
	return retval;
}


bool nbl_glsl_workgroupBroadcastFirst(in bool val) {return nbl_glsl_workgroupBroadcast(val,0u);}
float nbl_glsl_workgroupBroadcastFirst(in float val) {return nbl_glsl_workgroupBroadcast(val,0u);}
int nbl_glsl_workgroupBroadcastFirst(in int val) {return nbl_glsl_workgroupBroadcast(val,0u);}

/** TODO @Hazardu, @Przemog or @Anastazluk
// these could use optimization from `bitcount` on shared memory, then a part-sized arithmetic scan
uint nbl_glsl_workgroupBallotFindLSB();
uint nbl_glsl_workgroupBallotFindMSB();
**/


// TODO: [[unroll]] the while 5-times ?
#define NBL_GLSL_WORKGROUP_COMMON_IMPL_HEAD(CONV,INCLUSIVE_SUBGROUP_OP,VALUE,IDENTITY,INVCONV,ITEM_COUNT,SCAN) SUBGROUP_SCRATCH_INITIALIZE(VALUE,ITEM_COUNT,IDENTITY,INVCONV) \
	const uint lastInvocation = ITEM_COUNT-1u; \
	uint lastInvocationInLevel = lastInvocation; \
	uint firstLevelScan = INVCONV(INCLUSIVE_SUBGROUP_OP(false,VALUE)); \
	uint scan = firstLevelScan; \
	const bool possibleProp = pseudoSubgroupInvocation==loMask; \
	const uint pseudoSubgroupID = gl_LocalInvocationIndex>>nbl_glsl_SubgroupSizeLog2; \
	const uint nextStoreIndex = nbl_glsl_subgroup_getSubgroupEmulationMemoryStoreOffset(loMask,pseudoSubgroupID); \
	uint scanStoreIndex = nbl_glsl_subgroup_getSubgroupEmulationMemoryStoreOffset(loMask,lastInvocation)+gl_LocalInvocationIndex+1u; \
	bool participate = gl_LocalInvocationIndex<=lastInvocationInLevel; \
	while (lastInvocationInLevel>=nbl_glsl_SubgroupSize*nbl_glsl_SubgroupSize) \
	{ \
		CONDITIONAL_BARRIER \
		if (participate) \
		{ \
			if (any(bvec2(gl_LocalInvocationIndex==lastInvocationInLevel,possibleProp))) \
				_NBL_GLSL_SCRATCH_SHARED_DEFINED_[nextStoreIndex] = scan; \
		} \
		barrier(); \
		participate = gl_LocalInvocationIndex<=(lastInvocationInLevel>>=nbl_glsl_SubgroupSizeLog2); \
		if (participate) \
		{ \
			const uint prevLevelScan = _NBL_GLSL_SCRATCH_SHARED_DEFINED_[subgroupScanStoreOffset]; \
			scan = INVCONV(INCLUSIVE_SUBGROUP_OP(false,CONV(prevLevelScan))); \
			if (SCAN) _NBL_GLSL_SCRATCH_SHARED_DEFINED_[scanStoreIndex] = scan; \
		} \
		if (SCAN) scanStoreIndex += lastInvocationInLevel+1u; \
	} \
	if (lastInvocationInLevel>=nbl_glsl_SubgroupSize) \
	{ \
		CONDITIONAL_BARRIER \
		if (participate) \
		{ \
			if (any(bvec2(gl_LocalInvocationIndex==lastInvocationInLevel,possibleProp))) \
				_NBL_GLSL_SCRATCH_SHARED_DEFINED_[nextStoreIndex] = scan; \
		} \
		barrier(); \
		participate = gl_LocalInvocationIndex<=(lastInvocationInLevel>>=nbl_glsl_SubgroupSizeLog2); \
		if (participate) \
		{ \
			const uint prevLevelScan = _NBL_GLSL_SCRATCH_SHARED_DEFINED_[subgroupScanStoreOffset]; \
			scan = INVCONV(INCLUSIVE_SUBGROUP_OP(false,CONV(prevLevelScan))); \
			if (SCAN) _NBL_GLSL_SCRATCH_SHARED_DEFINED_[scanStoreIndex] = scan; \
		} \
	}

#define NBL_GLSL_WORKGROUP_SCAN_IMPL_TAIL(EXCLUSIVE,CONV,INCLUSIVE_SUBGROUP_OP,IDENTITY,INVCONV,OP) CONDITIONAL_BARRIER \
	if (lastInvocation>=nbl_glsl_SubgroupSize) \
	{ \
		uint scanLoadIndex = scanStoreIndex+nbl_glsl_SubgroupSize; \
		const uint shiftedInvocationIndex = gl_LocalInvocationIndex+nbl_glsl_SubgroupSize; \
		const uint currentToHighLevel = pseudoSubgroupID-shiftedInvocationIndex; \
		for (uint logShift=(findMSB(lastInvocation)/nbl_glsl_SubgroupSizeLog2-1u)*nbl_glsl_SubgroupSizeLog2; logShift>0u; logShift-=nbl_glsl_SubgroupSizeLog2) \
		{ \
			lastInvocationInLevel = lastInvocation>>logShift; \
			barrier(); \
			const uint currentLevelIndex = scanLoadIndex-(lastInvocationInLevel+1u); \
			if (shiftedInvocationIndex<=lastInvocationInLevel) \
				_NBL_GLSL_SCRATCH_SHARED_DEFINED_[currentLevelIndex] = INVCONV(OP (CONV(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[scanLoadIndex+currentToHighLevel]),CONV(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[currentLevelIndex]))); \
			scanLoadIndex = currentLevelIndex; \
		} \
		barrier(); \
		if (gl_LocalInvocationIndex<=lastInvocation && pseudoSubgroupID!=0u) \ 
		{ \
			const uint higherLevelExclusive = _NBL_GLSL_SCRATCH_SHARED_DEFINED_[scanLoadIndex+currentToHighLevel-1u]; \
			firstLevelScan = INVCONV(OP(CONV(higherLevelExclusive), CONV(firstLevelScan))); \
		} \
	} \
	if (EXCLUSIVE) \
	{ \
			if (gl_LocalInvocationIndex<lastInvocation) \
				_NBL_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex+1u] = firstLevelScan; \
			barrier(); \
			return any(bvec2(gl_LocalInvocationIndex!=0u,gl_LocalInvocationIndex<=lastInvocation)) ? CONV(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex]):IDENTITY; \
	} \
	else \
		return CONV(firstLevelScan);


uint nbl_glsl_workgroupBallotScanBitCount_impl(in bool exclusive);

uint nbl_glsl_workgroupBallotInclusiveBitCount()
{
	return nbl_glsl_workgroupBallotScanBitCount_impl(false);
}
uint nbl_glsl_workgroupBallotExclusiveBitCount()
{
	return nbl_glsl_workgroupBallotScanBitCount_impl(true);
}

uint nbl_glsl_workgroupBallotScanBitCount_impl_impl(in uint localBitCount)
{
	barrier();
	NBL_GLSL_WORKGROUP_COMMON_IMPL_HEAD(nbl_glsl_identityFunction,nbl_glsl_subgroupInclusiveAdd_impl,localBitCount,0u,nbl_glsl_identityFunction,nbl_glsl_workgroupBallot_impl_BitfieldDWORDs,true)
	NBL_GLSL_WORKGROUP_SCAN_IMPL_TAIL(true,nbl_glsl_identityFunction,nbl_glsl_subgroupInclusiveAdd_impl,0u,nbl_glsl_identityFunction,nbl_glsl_add)
}
uint nbl_glsl_workgroupBallotScanBitCount_impl(in bool exclusive)
{
	const uint _dword = nbl_glsl_workgroupBallot_impl_getDWORD(gl_LocalInvocationIndex);
	const uint localBitfield = _NBL_GLSL_SCRATCH_SHARED_DEFINED_[_dword];

	uint globalCount;
	{
		uint localBitfieldBackup;
		if (gl_LocalInvocationIndex<nbl_glsl_workgroupBallot_impl_BitfieldDWORDs)
			localBitfieldBackup = _NBL_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex];
		// scan hierarchically, invocations with `gl_LocalInvocationIndex>=nbl_glsl_workgroupBallot_impl_BitfieldDWORDs` will have garbage here
		nbl_glsl_workgroupBallotScanBitCount_impl_impl(bitCount(localBitfieldBackup));
		// fix it (abuse the fact memory is left over)
		globalCount = _dword!=0u ? _NBL_GLSL_SCRATCH_SHARED_DEFINED_[_dword]:0u;
		barrier();

		// restore
		if (gl_LocalInvocationIndex<nbl_glsl_workgroupBallot_impl_BitfieldDWORDs)
			_NBL_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex] = localBitfieldBackup;
		barrier();
	}

	const uint mask = (exclusive ? 0x7fffffffu:0xffffffffu)>>(31u-(gl_LocalInvocationIndex&31u));
	return globalCount+bitCount(localBitfield&mask);
}

#undef CONDITIONAL_BARRIER

#endif
