#ifndef _IRR_BUILTIN_GLSL_WORKGROUP_BALLOT_INCLUDED_
#define _IRR_BUILTIN_GLSL_WORKGROUP_BALLOT_INCLUDED_


#include <irr/builtin/glsl/subgroup/arithmetic_portability.glsl>
#include <irr/builtin/glsl/workgroup/basic.glsl>


#define irr_glsl_workgroupBallot_impl_getDWORD(IX) (IX>>5)
#define irr_glsl_workgroupBallot_impl_BitfieldDWORDs irr_glsl_workgroupBallot_impl_getDWORD(_IRR_GLSL_WORKGROUP_SIZE_+31)


/*
#ifdef GL_KHR_subgroup_arithmetic


#define _IRR_GLSL_WORKGROUP_BALLOT_SHARED_SIZE_NEEDED_IMPL_  irr_glsl_workgroupBallot_impl_BitfieldDWORDs

#define CONDITIONAL_BARRIER

#else
*/

#ifdef IRR_GLSL_SUBGROUP_SIZE_IS_CONSTEXPR
	#define irr_glsl_workgroupBallot_impl_BitDWORDs_rounded IRR_GLSL_AND(irr_glsl_workgroupBallot_impl_BitfieldDWORDs+irr_glsl_SubgroupSize-1,-irr_glsl_SubgroupSize)
#else
	#define irr_glsl_workgroupBallot_impl_BitDWORDs_rounded IRR_GLSL_AND(irr_glsl_workgroupBallot_impl_BitfieldDWORDs+irr_glsl_MaxSubgroupSize-1,-irr_glsl_MaxSubgroupSize)
#endif
#define _IRR_GLSL_WORKGROUP_BALLOT_SHARED_SIZE_NEEDED_IMPL_  IRR_GLSL_EVAL(irr_glsl_workgroupBallot_impl_BitDWORDs_rounded+(irr_glsl_workgroupBallot_impl_BitDWORDs_rounded>>1))

#define CONDITIONAL_BARRIER barrier(); \
	memoryBarrierShared();

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
	memoryBarrierShared();
	if (value)
		atomicOr(_IRR_GLSL_SCRATCH_SHARED_DEFINED_[irr_glsl_workgroupBallot_impl_getDWORD(gl_LocalInvocationIndex)],1u<<(gl_LocalInvocationIndex&31u));
}
void irr_glsl_workgroupBallot(in bool value)
{
	barrier();
	memoryBarrierShared();
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
	memoryBarrierShared();
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
	memoryBarrierShared();
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
	memoryBarrierShared();
	return retval;
}


uint irr_glsl_workgroupBroadcast_noBarriers(in uint val, in uint id)
{
	if (gl_LocalInvocationIndex==id)
		_IRR_GLSL_SCRATCH_SHARED_DEFINED_[irr_glsl_workgroupBallot_impl_BitfieldDWORDs] = val;
	barrier();
	memoryBarrierShared();
	return _IRR_GLSL_SCRATCH_SHARED_DEFINED_[irr_glsl_workgroupBallot_impl_BitfieldDWORDs];
}
uint irr_glsl_workgroupBroadcast(in uint val, in uint id)
{
	barrier();
	memoryBarrierShared();
	const uint retval = irr_glsl_workgroupBroadcast_noBarriers(val,id);
	barrier();
	memoryBarrierShared();
	return retval;
}

uint irr_glsl_workgroupBroadcastFirst_noBarriers(in uint val)
{
	if (irr_glsl_workgroupElect())
		_IRR_GLSL_SCRATCH_SHARED_DEFINED_[irr_glsl_workgroupBallot_impl_BitfieldDWORDs] = val;
	barrier();
	memoryBarrierShared();
	return _IRR_GLSL_SCRATCH_SHARED_DEFINED_[irr_glsl_workgroupBallot_impl_BitfieldDWORDs];
}
uint irr_glsl_workgroupBroadcastFirst(in uint val)
{
	barrier();
	memoryBarrierShared();
	const uint retval = irr_glsl_workgroupBroadcastFirst_noBarriers(val);
	barrier();
	memoryBarrierShared();
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


// TODO: unroll the while 5-times ?
#define IRR_GLSL_WORKGROUP_COMMON_IMPL_HEAD(CONV,INCLUSIVE_SUBGROUP_OP,VALUE,IDENTITY,INVCONV,ITEM_COUNT) const uint lastInvocation = ITEM_COUNT-1u; \
	const uint subgroupSizeLog2 = findLSB(irr_glsl_SubgroupSize); \
	const uint loMask = irr_glsl_SubgroupSize-1u; \
	const bool propagateReduction = (gl_LocalInvocationIndex&loMask)==loMask; \
	uint firstLevelScan = INVCONV(INCLUSIVE_SUBGROUP_OP(false,VALUE)); \
	uint lastInvocationInLevel = lastInvocation; \
	uint lowerIndex = gl_LocalInvocationIndex>>subgroupSizeLog2; \
	const uint higherIndexDiff = gl_LocalInvocationIndex-lowerIndex; \
	uint scan = firstLevelScan; \
	CONDITIONAL_BARRIER \
	while (lastInvocationInLevel>=irr_glsl_SubgroupSize) \
	{ \
		const bool prop = propagateReduction&&gl_LocalInvocationIndex<lastInvocationInLevel || gl_LocalInvocationIndex==lastInvocationInLevel; \
		if (prop) \
			_IRR_GLSL_SCRATCH_SHARED_DEFINED_[lowerIndex] = scan; \
		barrier(); \
		memoryBarrierShared(); \
		lastInvocationInLevel >>= subgroupSizeLog2; \
		if (gl_LocalInvocationIndex<=lastInvocationInLevel) \
			scan = INVCONV(INCLUSIVE_SUBGROUP_OP(false,CONV(_IRR_GLSL_SCRATCH_SHARED_DEFINED_[lowerIndex+higherIndexDiff]))); \
		lowerIndex += lastInvocationInLevel+1u; \
	} 
#define IRR_GLSL_WORKGROUP_SCAN_IMPL_TAIL(CONV,OP,INVCONV) CONDITIONAL_BARRIER \
	if (lastInvocation>=irr_glsl_SubgroupSize) \
	{ \
		lowerIndex -= lastInvocationInLevel+1u; \
		if (gl_LocalInvocationIndex<lastInvocationInLevel) \
			_IRR_GLSL_SCRATCH_SHARED_DEFINED_[lowerIndex+higherIndexDiff] = scan; \
		barrier(); \
		memoryBarrierShared(); \
		for (uint logShift=(findMSB(lastInvocation)/subgroupSizeLog2-1u)*subgroupSizeLog2; logShift>0u; logShift-=subgroupSizeLog2) \
		{ \
			lastInvocationInLevel = lastInvocation>>logShift; \
			const uint higherIndex = lowerIndex-(lastInvocationInLevel+1u); \
			if (gl_LocalInvocationIndex<lastInvocationInLevel) \
			{ \
				const uint outIx = higherIndex+higherIndexDiff+1u; \
				_IRR_GLSL_SCRATCH_SHARED_DEFINED_[outIx] = INVCONV(OP (CONV(_IRR_GLSL_SCRATCH_SHARED_DEFINED_[outIx]),CONV(_IRR_GLSL_SCRATCH_SHARED_DEFINED_[lowerIndex]))); \
			} \
			barrier(); \
			memoryBarrierShared(); \
			lowerIndex = higherIndex; \
		} \
		if (gl_LocalInvocationIndex!=0u) \
			firstLevelScan = INVCONV(OP (CONV(firstLevelScan),CONV(_IRR_GLSL_SCRATCH_SHARED_DEFINED_[lowerIndex-1u]))); \
	}


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
	//#ifndef GL_KHR_subgroup_arithmetic
	barrier();
	memoryBarrierShared();
	SUBGROUP_SCRATCH_CLEAR(irr_glsl_workgroupBallot_impl_BitfieldDWORDs,0u)
	//#endif

	// scan hierarchically
	uint globalCount;
	{
		IRR_GLSL_WORKGROUP_COMMON_IMPL_HEAD(irr_glsl_identityFunction,irr_glsl_subgroupInclusiveAdd_impl,localBitfieldBackup,0u,irr_glsl_identityFunction,irr_glsl_workgroupBallot_impl_BitfieldDWORDs)
		IRR_GLSL_WORKGROUP_SCAN_IMPL_TAIL(irr_glsl_identityFunction,irr_glsl_add,irr_glsl_identityFunction)
		barrier();
		memoryBarrierShared();
		globalCount = firstLevelScan; // TODO: this is broken, needs a shuffle to be a an exclusive scan
	}
	
	// restore
	if (gl_LocalInvocationIndex<irr_glsl_workgroupBallot_impl_BitfieldDWORDs)
		_IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex] = localBitfieldBackup;
	barrier();
	memoryBarrierShared();

	const uint mask = 0xffffffffu>>((exclusive ? 32u:31u)-(gl_LocalInvocationIndex&31u));
	return globalCount+bitCount(localBitfieldBackup&mask);
}

#undef CONDITIONAL_BARRIER

#endif
