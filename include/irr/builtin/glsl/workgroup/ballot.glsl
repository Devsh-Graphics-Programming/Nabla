#ifndef _IRR_BUILTIN_GLSL_WORKGROUP_BALLOT_INCLUDED_
#define _IRR_BUILTIN_GLSL_WORKGROUP_BALLOT_INCLUDED_



#include <irr/builtin/glsl/workgroup/shared_ballot.glsl>


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
`irr_glsl_workgroupOp`s then the workgroup size must not be smaller than half a subgroup but having workgroups smaller than a subgroup is extremely bad practice.
*/

//#endif



#ifdef _IRR_GLSL_SCRATCH_SHARED_DEFINED_
	#if IRR_GLSL_LESS(_IRR_GLSL_SCRATCH_SHARED_SIZE_DEFINED_,_IRR_GLSL_WORKGROUP_BALLOT_SHARED_SIZE_NEEDED_)
		#error "Not enough shared memory declared for workgroup ballot!"
	#endif
#else
	#if IRR_GLSL_GREATER(_IRR_GLSL_WORKGROUP_BALLOT_SHARED_SIZE_NEEDED_,0)
		#define _IRR_GLSL_SCRATCH_SHARED_DEFINED_ irr_glsl_workgroupBallotScratchShared
		#define _IRR_GLSL_SCRATCH_SHARED_SIZE_DEFINED_ _IRR_GLSL_WORKGROUP_BALLOT_SHARED_SIZE_NEEDED_
		shared uint _IRR_GLSL_SCRATCH_SHARED_DEFINED_[_IRR_GLSL_WORKGROUP_BALLOT_SHARED_SIZE_NEEDED_];
	#endif
#endif



#include <irr/builtin/glsl/subgroup/arithmetic_portability.glsl>



// puts the result into shared memory at offsets [0,_IRR_GLSL_WORKGROUP_SIZE_/32)
void irr_glsl_workgroupBallot_noBarriers(in bool value)
{
	// TODO: Optimization using subgroupBallot in an ifdef IRR_GL_something (need to do feature mapping first)
	if (gl_LocalInvocationIndex<irr_glsl_workgroupBallot_impl_BitfieldDWORDs)
		_IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex] = 0u;
	barrier();
	if (value)
		atomicOr(_IRR_GLSL_SCRATCH_SHARED_DEFINED_[irr_glsl_workgroupBallot_impl_getDWORD(gl_LocalInvocationIndex)],1u<<(gl_LocalInvocationIndex&31u));
}
void irr_glsl_workgroupBallot(in bool value)
{
	barrier();
	irr_glsl_workgroupBallot_noBarriers(value);
	barrier();
}

// the ballot is expected to be in _IRR_GLSL_SCRATCH_SHARED_DEFINED_ at offsets [0,_IRR_GLSL_WORKGROUP_SIZE_/32)
bool irr_glsl_workgroupBallotBitExtract_noEndBarriers(in uint index)
{
	return (_IRR_GLSL_SCRATCH_SHARED_DEFINED_[irr_glsl_workgroupBallot_impl_getDWORD(index)]&(1u<<(index&31u)))!=0u;
}
bool irr_glsl_workgroupBallotBitExtract(in uint index)
{
	const bool retval = irr_glsl_workgroupBallotBitExtract_noEndBarriers(index);
	barrier();
	return retval;
}

bool irr_glsl_workgroupInverseBallot_noEndBarriers()
{
	return irr_glsl_workgroupBallotBitExtract_noEndBarriers(gl_LocalInvocationIndex);
}
bool irr_glsl_workgroupInverseBallot()
{
	return irr_glsl_workgroupBallotBitExtract(gl_LocalInvocationIndex);
}


uint irr_glsl_workgroupBallotBitCount_noEndBarriers()
{
	_IRR_GLSL_SCRATCH_SHARED_DEFINED_[irr_glsl_workgroupBallot_impl_BitfieldDWORDs] = 0u;
	barrier();
	if (gl_LocalInvocationIndex<irr_glsl_workgroupBallot_impl_BitfieldDWORDs)
	{
		const uint localBallot = _IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex];
		const uint localBallotBitCount = bitCount(localBallot);
		atomicAdd(_IRR_GLSL_SCRATCH_SHARED_DEFINED_[irr_glsl_workgroupBallot_impl_BitfieldDWORDs],localBallotBitCount);
	}
	barrier();

	return _IRR_GLSL_SCRATCH_SHARED_DEFINED_[irr_glsl_workgroupBallot_impl_BitfieldDWORDs];
}
uint irr_glsl_workgroupBallotBitCount()
{
	const uint retval = irr_glsl_workgroupBallotBitCount_noEndBarriers();
	barrier();
	return retval;
}


uint irr_glsl_workgroupBroadcast_noBarriers(in uint val, in uint id)
{
	if (gl_LocalInvocationIndex==id)
		_IRR_GLSL_SCRATCH_SHARED_DEFINED_[irr_glsl_workgroupBallot_impl_BitfieldDWORDs] = val;
	barrier();
	return _IRR_GLSL_SCRATCH_SHARED_DEFINED_[irr_glsl_workgroupBallot_impl_BitfieldDWORDs];
}
uint irr_glsl_workgroupBroadcast(in uint val, in uint id)
{
	barrier();
	const uint retval = irr_glsl_workgroupBroadcast_noBarriers(val,id);
	barrier();
	return retval;
}

uint irr_glsl_workgroupBroadcastFirst_noBarriers(in uint val)
{
	if (irr_glsl_workgroupElect())
		_IRR_GLSL_SCRATCH_SHARED_DEFINED_[irr_glsl_workgroupBallot_impl_BitfieldDWORDs] = val;
	barrier();
	return _IRR_GLSL_SCRATCH_SHARED_DEFINED_[irr_glsl_workgroupBallot_impl_BitfieldDWORDs];
}
uint irr_glsl_workgroupBroadcastFirst(in uint val)
{
	barrier();
	const uint retval = irr_glsl_workgroupBroadcastFirst_noBarriers(val);
	barrier();
	return retval;
}

/** TODO @Hazardu, @Przemog or @Anastazluk
bool irr_glsl_workgroupBroadcast(in bool val, in uint id);
float irr_glsl_workgroupBroadcast(in float val, in uint id);
int irr_glsl_workgroupBroadcast(in int val, in uint id);

bool irr_glsl_workgroupBroadcastFirst(in bool val) {return irr_glsl_workgroupBroadcast(val,0u);}
float irr_glsl_workgroupBroadcastFirst(in float val) {return irr_glsl_workgroupBroadcast(val,0u);}
int irr_glsl_workgroupBroadcastFirst(in int val) {return irr_glsl_workgroupBroadcast(val,0u);}

// these could use optimization from `bitcount` on shared memory, then a part-sized arithmetic scan
uint irr_glsl_workgroupBallotFindLSB();
uint irr_glsl_workgroupBallotFindMSB();
**/


// TODO: [[unroll]] the while 5-times ?
#define IRR_GLSL_WORKGROUP_COMMON_IMPL_HEAD(CONV,INCLUSIVE_SUBGROUP_OP,VALUE,IDENTITY,INVCONV,ITEM_COUNT,SCAN) SUBGROUP_SCRATCH_INITIALIZE(VALUE,ITEM_COUNT,IDENTITY,INVCONV) \
	const uint lastInvocation = ITEM_COUNT-1u; \
	uint lastInvocationInLevel = lastInvocation; \
	uint firstLevelScan = INVCONV(INCLUSIVE_SUBGROUP_OP(false,VALUE)); \
	uint scan = firstLevelScan; \
	const bool possibleProp = pseudoSubgroupInvocation==loMask; \
	const uint subgroupSizeLog2 = findLSB(irr_glsl_SubgroupSize); \
	const uint pseudoSubgroupID = (gl_LocalInvocationIndex>>subgroupSizeLog2); \
	const uint nextStoreIndex = irr_glsl_subgroup_getSubgroupEmulationMemoryStoreOffset(loMask,pseudoSubgroupID); \
	uint scanStoreIndex = irr_glsl_subgroup_getSubgroupEmulationMemoryStoreOffset(loMask,lastInvocation)+gl_LocalInvocationIndex+1u; \
	bool participate = gl_LocalInvocationIndex<=lastInvocationInLevel; \
	while (lastInvocationInLevel>=irr_glsl_SubgroupSize*irr_glsl_SubgroupSize) \
	{ \
		CONDITIONAL_BARRIER \
		if (participate) \
		{ \
			if (any(bvec2(gl_LocalInvocationIndex==lastInvocationInLevel,possibleProp))) \
				_IRR_GLSL_SCRATCH_SHARED_DEFINED_[nextStoreIndex] = scan; \
		} \
		barrier(); \
		participate = gl_LocalInvocationIndex<=(lastInvocationInLevel>>=subgroupSizeLog2); \
		if (participate) \
		{ \
			const uint prevLevelScan = _IRR_GLSL_SCRATCH_SHARED_DEFINED_[subgroupScanStoreOffset]; \
			scan = INVCONV(INCLUSIVE_SUBGROUP_OP(false,CONV(prevLevelScan))); \
			if (SCAN) _IRR_GLSL_SCRATCH_SHARED_DEFINED_[scanStoreIndex] = scan; \
		} \
		if (SCAN) scanStoreIndex += lastInvocationInLevel+1u; \
	} \
	if (lastInvocationInLevel>=irr_glsl_SubgroupSize) \
	{ \
		CONDITIONAL_BARRIER \
		if (participate) \
		{ \
			if (any(bvec2(gl_LocalInvocationIndex==lastInvocationInLevel,possibleProp))) \
				_IRR_GLSL_SCRATCH_SHARED_DEFINED_[nextStoreIndex] = scan; \
		} \
		barrier(); \
		participate = gl_LocalInvocationIndex<=(lastInvocationInLevel>>=subgroupSizeLog2); \
		if (participate) \
		{ \
			const uint prevLevelScan = _IRR_GLSL_SCRATCH_SHARED_DEFINED_[subgroupScanStoreOffset]; \
			scan = INVCONV(INCLUSIVE_SUBGROUP_OP(false,CONV(prevLevelScan))); \
			if (SCAN) _IRR_GLSL_SCRATCH_SHARED_DEFINED_[scanStoreIndex] = scan; \
		} \
	}

#define IRR_GLSL_WORKGROUP_SCAN_IMPL_TAIL(EXCLUSIVE,CONV,INCLUSIVE_SUBGROUP_OP,IDENTITY,INVCONV,OP) CONDITIONAL_BARRIER \
	if (lastInvocation>=irr_glsl_SubgroupSize) \
	{ \
		uint scanLoadIndex = scanStoreIndex+irr_glsl_SubgroupSize; \
		const uint shiftedInvocationIndex = gl_LocalInvocationIndex+irr_glsl_SubgroupSize; \
		const uint currentToHighLevel = pseudoSubgroupID-shiftedInvocationIndex; \
		for (uint logShift=(findMSB(lastInvocation)/subgroupSizeLog2-1u)*subgroupSizeLog2; logShift>0u; logShift-=subgroupSizeLog2) \
		{ \
			lastInvocationInLevel = lastInvocation>>logShift; \
			barrier(); \
			const uint currentLevelIndex = scanLoadIndex-(lastInvocationInLevel+1u); \
			if (shiftedInvocationIndex<=lastInvocationInLevel) \
				_IRR_GLSL_SCRATCH_SHARED_DEFINED_[currentLevelIndex] = INVCONV(OP (CONV(_IRR_GLSL_SCRATCH_SHARED_DEFINED_[scanLoadIndex+currentToHighLevel]),CONV(_IRR_GLSL_SCRATCH_SHARED_DEFINED_[currentLevelIndex]))); \
			scanLoadIndex = currentLevelIndex; \
		} \
		barrier(); \
		if (gl_LocalInvocationIndex<=lastInvocation && pseudoSubgroupID!=0u) \ 
		{ \
			const uint higherLevelExclusive = _IRR_GLSL_SCRATCH_SHARED_DEFINED_[scanLoadIndex+currentToHighLevel-1u]; \
			firstLevelScan = INVCONV(OP(CONV(higherLevelExclusive), CONV(firstLevelScan))); \
		} \
	} \
	if (EXCLUSIVE) \
	{ \
			if (gl_LocalInvocationIndex<lastInvocation) \
				_IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex+1u] = firstLevelScan; \
			barrier(); \
			if (gl_LocalInvocationIndex<lastInvocation) \
				return gl_LocalInvocationIndex!=0u ? CONV(_IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex]):IDENTITY; \
			else \
				return IDENTITY; \
	} \
	else \
		return CONV(firstLevelScan);


uint irr_glsl_workgroupBallotScanBitCount_impl(in bool exclusive);

uint irr_glsl_workgroupBallotInclusiveBitCount()
{
	return irr_glsl_workgroupBallotScanBitCount_impl(false);
}
uint irr_glsl_workgroupBallotExclusiveBitCount()
{
	return irr_glsl_workgroupBallotScanBitCount_impl(true);
}

uint irr_glsl_workgroupBallotScanBitCount_impl_impl(in uint localBitfield)
{
	barrier();
	IRR_GLSL_WORKGROUP_COMMON_IMPL_HEAD(irr_glsl_identityFunction,irr_glsl_subgroupInclusiveAdd_impl,localBitfield,0u,irr_glsl_identityFunction,irr_glsl_workgroupBallot_impl_BitfieldDWORDs,true)
	IRR_GLSL_WORKGROUP_SCAN_IMPL_TAIL(true,irr_glsl_identityFunction,irr_glsl_subgroupInclusiveAdd_impl,0u,irr_glsl_identityFunction,irr_glsl_add)
}
uint irr_glsl_workgroupBallotScanBitCount_impl(in bool exclusive)
{
	const uint _dword = irr_glsl_workgroupBallot_impl_getDWORD(gl_LocalInvocationIndex);
	const uint localBitfield = _IRR_GLSL_SCRATCH_SHARED_DEFINED_[_dword];

	uint globalCount;
	{
		uint localBitfieldBackup;
		if (gl_LocalInvocationIndex<irr_glsl_workgroupBallot_impl_BitfieldDWORDs)
			localBitfieldBackup = _IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex];
		// scan hierarchically, invocations with `gl_LocalInvocationIndex>=irr_glsl_workgroupBallot_impl_BitfieldDWORDs` will have garbage here
		irr_glsl_workgroupBallotScanBitCount_impl_impl(localBitfieldBackup);
		// fix it (abuse the fact memory is left over)
		globalCount = _dword!=0u ? _IRR_GLSL_SCRATCH_SHARED_DEFINED_[_dword]:0u;
		barrier();

		// restore
		if (gl_LocalInvocationIndex<irr_glsl_workgroupBallot_impl_BitfieldDWORDs)
			_IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex] = localBitfieldBackup;
		barrier();
	}

	const uint mask = 0xffffffffu>>((exclusive ? 32u:31u)-(gl_LocalInvocationIndex&31u));
	return globalCount+bitCount(localBitfield&mask);
}

#undef CONDITIONAL_BARRIER

#endif
