#ifndef _IRR_BUILTIN_GLSL_WORKGROUP_BALLOT_INCLUDED_
#define _IRR_BUILTIN_GLSL_WORKGROUP_BALLOT_INCLUDED_


#include <irr/builtin/glsl/subgroup/arithmetic_portability.glsl>
#include <irr/builtin/glsl/workgroup/basic.glsl>


#define irr_glsl_workgroupBallot_impl_getDWORD(IX) (IX>>5)
#define irr_glsl_workgroupBallot_impl_BitfieldDWORDs irr_glsl_workgroupBallot_impl_getDWORD(_IRR_GLSL_WORKGROUP_SIZE_+31)


#define _IRR_GLSL_WORKGROUP_BALLOT_SHARED_SIZE_NEEDED_IMPL_  (irr_glsl_workgroupBallot_impl_BitfieldDWORDs+_IRR_GLSL_WORKGROUP_SIZE_)


/*
#ifdef GL_KHR_subgroup_arithmetic

#define CONDITIONAL_BARRIER

#else
*/

#define CONDITIONAL_BARRIER barrier();

/*
If `GL_KHR_subgroup_arithmetic` is not available then these functions require emulated subgroup operations, which in turn means that if you're using the
`irr_glsl_workgroupOp`s then the workgroup size must not be smaller than half a subgroup but having workgroups smaller than a subgroup is extremely bad practice.
*/

#if IRR_GLSL_GREATER(_IRR_GLSL_SUBGROUP_ARITHMETIC_EMULATION_SHARED_SIZE_NEEDED_,_IRR_GLSL_WORKGROUP_BALLOT_SHARED_SIZE_NEEDED_IMPL_)
	#define _IRR_GLSL_WORKGROUP_BALLOT_SHARED_SIZE_NEEDED_ _IRR_GLSL_SUBGROUP_ARITHMETIC_EMULATION_SHARED_SIZE_NEEDED_
#else
	#define _IRR_GLSL_WORKGROUP_BALLOT_SHARED_SIZE_NEEDED_ _IRR_GLSL_WORKGROUP_BALLOT_SHARED_SIZE_NEEDED_IMPL_
#endif

//#endif



#ifdef _IRR_GLSL_SCRATCH_SHARED_DEFINED_
	#if IRR_GLSL_LESS(_IRR_GLSL_SCRATCH_SHARED_SIZE_DEFINED_,_IRR_GLSL_WORKGROUP_BALLOT_SHARED_SIZE_NEEDED_)
		#error "Not enough shared memory declared for workgroup ballot!"
	#endif
#else
	#if IRR_GLSL_GREATER(_IRR_GLSL_WORKGROUP_BALLOT_SHARED_SIZE_NEEDED_,0)
		#define _IRR_GLSL_SCRATCH_SHARED_DEFINED_ irr_glsl_workgroupBallotScratchShared
		shared uint _IRR_GLSL_SCRATCH_SHARED_DEFINED_[_IRR_GLSL_WORKGROUP_BALLOT_SHARED_SIZE_NEEDED_];
	#endif
#endif



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
#define IRR_GLSL_WORKGROUP_COMMON_IMPL_HEAD(CONV,INCLUSIVE_SUBGROUP_OP,VALUE,IDENTITY,INVCONV,ITEM_COUNT) SUBGROUP_SCRATCH_INITIALIZE(VALUE,ITEM_COUNT,IDENTITY,INVCONV) \
	const uint lastInvocation = ITEM_COUNT-1u; \
	uint lastInvocationInLevel = lastInvocation; \
	const uint firstLevelScan = INVCONV(INCLUSIVE_SUBGROUP_OP(false,VALUE)); \
	uint scan = firstLevelScan; \
	const bool possibleProp = pseudoSubgroupInvocation==loMask; \
	const uint subgroupSizeLog2 = findLSB(irr_glsl_SubgroupSize); \
	const uint pseudoSubgroupID = (gl_LocalInvocationIndex>>subgroupSizeLog2); \
	const uint nextPassStoreIndex = irr_glsl_subgroup_impl_getSubgroupEmulationMemoryStoreOffset( \
		irr_glsl_subgroup_impl_getSubgroupEmulationMemoryStart( \
			irr_glsl_subgroup_impl_pseudoSubgroupElectedInvocation(loMask,pseudoSubgroupID) \
		), \
		irr_glsl_subgroup_impl_pseudoSubgroupInvocation(loMask,pseudoSubgroupID) \
	); \
	bool participate = gl_LocalInvocationIndex<=lastInvocationInLevel;

#define IRR_GLSL_WORKGROUP_REDUCE_IMPL_TAIL(CONV,INCLUSIVE_SUBGROUP_OP,INVCONV) while (lastInvocationInLevel>=irr_glsl_SubgroupSize) \
	{ \
		CONDITIONAL_BARRIER \
		if (participate) \
		{ \
			if (any(bvec2(gl_LocalInvocationIndex==lastInvocationInLevel,possibleProp))) \
				_IRR_GLSL_SCRATCH_SHARED_DEFINED_[nextPassStoreIndex] = scan; \
		} \
		barrier(); \
		participate = gl_LocalInvocationIndex<=(lastInvocationInLevel>>=subgroupSizeLog2); \
		if (participate) \
		{ \
			const uint prevLevelScan = _IRR_GLSL_SCRATCH_SHARED_DEFINED_[subgroupScanStoreOffset]; \
			scan = INVCONV(INCLUSIVE_SUBGROUP_OP(false,CONV(prevLevelScan))); \
		} \
	} \
	CONDITIONAL_BARRIER

#define IRR_GLSL_WORKGROUP_SCAN_IMPL_TAIL(CONV,INCLUSIVE_SUBGROUP_OP,INVCONV,OP) ;
/*
#define IRR_GLSL_WORKGROUP_SCAN_IMPL_TAIL(CONV,INCLUSIVE_SUBGROUP_OP,INVCONV,OP) while (lastInvocationInLevel>=irr_glsl_SubgroupSize) \
	{ \
		CONDITIONAL_BARRIER \
		const bool prop = propagateReduction&&gl_LocalInvocationIndex<lastInvocationInLevel || gl_LocalInvocationIndex==lastInvocationInLevel; \
		if (prop) \
			_IRR_GLSL_SCRATCH_SHARED_DEFINED_[lowerIndex] = scan; \
		barrier(); \
		memoryBarrierShared(); \
		lastInvocationInLevel >>= subgroupSizeLog2; \
		if (gl_LocalInvocationIndex<=lastInvocationInLevel) \
		{ \
			const uint prevLevelScan = _IRR_GLSL_SCRATCH_SHARED_DEFINED_[lowerIndex+lo2HiIndex]; \
			scan = INVCONV(INCLUSIVE_SUBGROUP_OP(false,CONV(prevLevelScan))); \
		} \
		lowerIndex += lastInvocationInLevel+1u; \
	} \
	CONDITIONAL_BARRIER \
	if (lastInvocation>=irr_glsl_SubgroupSize) \
	{ \
		lowerIndex -= lastInvocationInLevel; \
		if (gl_LocalInvocationIndex<lastInvocationInLevel) \
			_IRR_GLSL_SCRATCH_SHARED_DEFINED_[lowerIndex+lo2HiIndex] = scan; \
		lowerIndex--; \
		barrier(); \
		memoryBarrierShared(); \
		const uint shiftedInvocationIndex = gl_LocalInvocationIndex+irr_glsl_SubgroupSize; \
		const uint lo2HiIndexDownsweep = lo2HiIndex+irr_glsl_SubgroupSize; \
		for (uint logShift=(findMSB(lastInvocation)/subgroupSizeLog2-1u)*subgroupSizeLog2; logShift>0u; logShift-=subgroupSizeLog2) \
		{ \
			lastInvocationInLevel = lastInvocation>>logShift; \
			const uint higherIndex = lowerIndex-(lastInvocationInLevel+1u); \
			if (shiftedInvocationIndex<=lastInvocationInLevel) \
			{ \
				const uint outIx = higherIndex+lo2HiIndexDownsweep; \
				_IRR_GLSL_SCRATCH_SHARED_DEFINED_[outIx] = INVCONV(OP (CONV(_IRR_GLSL_SCRATCH_SHARED_DEFINED_[outIx]),CONV(_IRR_GLSL_SCRATCH_SHARED_DEFINED_[lowerIndex]))); \
			} \
			barrier(); \
			memoryBarrierShared(); \
			lowerIndex = higherIndex; \
		} \
		if (shiftedInvocationIndex<=lastInvocation) \
			firstLevelScan = INVCONV(OP (CONV(firstLevelScan),CONV(_IRR_GLSL_SCRATCH_SHARED_DEFINED_[lowerIndex-1u]))); \
	}
*/
uint irr_glsl_workgroupBallotScanBitCount_impl(in bool exclusive);

uint irr_glsl_workgroupBallotInclusiveBitCount()
{
	return irr_glsl_workgroupBallotScanBitCount_impl(false);
}
uint irr_glsl_workgroupBallotExclusiveBitCount()
{
	return irr_glsl_workgroupBallotScanBitCount_impl(true);
}

uint irr_glsl_workgroupBallotScanBitCount_impl(in bool exclusive)
{
	uint localBitfieldBackup;
	if (gl_LocalInvocationIndex<irr_glsl_workgroupBallot_impl_BitfieldDWORDs)
		localBitfieldBackup = _IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex];

	// scan hierarchically
	uint globalCount;
	{
		barrier();
		IRR_GLSL_WORKGROUP_COMMON_IMPL_HEAD(irr_glsl_identityFunction,irr_glsl_subgroupInclusiveAdd_impl,localBitfieldBackup,0u,irr_glsl_identityFunction,irr_glsl_workgroupBallot_impl_BitfieldDWORDs)
		IRR_GLSL_WORKGROUP_SCAN_IMPL_TAIL(irr_glsl_identityFunction,irr_glsl_subgroupInclusiveAdd_impl,irr_glsl_identityFunction,irr_glsl_add)
		_IRR_GLSL_SCRATCH_SHARED_DEFINED_[lastInvocationInLevel+1u+gl_LocalInvocationIndex] = firstLevelScan;
		barrier();
		globalCount = gl_LocalInvocationIndex!=0u ? _IRR_GLSL_SCRATCH_SHARED_DEFINED_[lastInvocationInLevel+gl_LocalInvocationIndex]:0u;
	}
	
	// restore
	if (gl_LocalInvocationIndex<irr_glsl_workgroupBallot_impl_BitfieldDWORDs)
		_IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex] = localBitfieldBackup;
	barrier();

	const uint mask = 0xffffffffu>>((exclusive ? 32u:31u)-(gl_LocalInvocationIndex&31u));
	return globalCount+bitCount(localBitfieldBackup&mask);
}

#undef CONDITIONAL_BARRIER

#endif
