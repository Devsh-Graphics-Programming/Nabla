// TODO: move all of this out to a builtin but after a major refactor
// basically implement all of https://github.com/KhronosGroup/GLSL/blob/master/extensions/khr/GL_KHR_shader_subgroup.txt for subgroups 

//! workgroup/basic.glsl
#include <irr/builtin/glsl/macros.glsl>
#include <irr/builtin/glsl/math/typeless_arithmetic.glsl>


#ifndef _IRR_GLSL_WORKGROUP_SIZE_
#error "User needs to let us know the size of the workgroup via _IRR_GLSL_WORKGROUP_SIZE_!"
#endif

bool irr_glsl_workgroupElect()
{
	return gl_LocalInvocationIndex==0u;
}

// TODO: maybe improve the macro so the if statement performs better
#define IRR_GLSL_WORKGROUP_REDUCE_IN_SHARED(SIZE,OP) { uint partition = 0x1u<<findMSB(SIZE-1u); \
		if (gl_LocalInvocationIndex<partition && (IRR_GLSL_IS_POT(SIZE) || gl_LocalInvocationIndex+partition<SIZE)) \
		{ \
			uint _IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex] OP _IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex+partition]; \
			_IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex] OP _IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex+partition]; \
		} \
		for (partition>>=1u; partition>0u; partition>>=1u) \
		{ \
			barrier(); \
			memoryBarrierShared(); \
			if (gl_LocalInvocationIndex<partition) \
				_IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex] OP _IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex+partition]; \
		} \
	}

//! workgroup/vote.glsl
#include <irr/builtin/glsl/macros.glsl>
#include <irr/builtin/glsl/math/typeless_arithmetic.glsl>

// TODO: depending on subgroup extensions available this will vary in size (usually divided by the subgroup size lower bound)
#define _IRR_GLSL_WORKGROUP_VOTE_SHARED_SIZE_NEEDED_  IRR_GLSL_ROUND_UP_POT(_IRR_GLSL_WORKGROUP_SIZE_)

#ifdef _IRR_GLSL_SCRATCH_SHARED_DEFINED_
/* can't get this to work either
	#if _IRR_GLSL_SCRATCH_SHARED_SIZE_DEFINED_<_IRR_GLSL_WORKGROUP_VOTE_SHARED_SIZE_NEEDED_
		#error "Not enough shared memory declared"
	#endif
*/
#else
	#if _IRR_GLSL_WORKGROUP_VOTE_SHARED_SIZE_NEEDED_>0
		#define _IRR_GLSL_SCRATCH_SHARED_DEFINED_ irr_glsl_workgroupVoteScratchShared
		shared uint _IRR_GLSL_SCRATCH_SHARED_DEFINED_[_IRR_GLSL_WORKGROUP_VOTE_SHARED_SIZE_NEEDED_];
	#endif
#endif


bool irr_glsl_workgroupAll(in bool value)
{
	// TODO: Optimization using subgroupAll in an ifdef IRR_GL_something (need to do feature mapping first), probably only first 2 lines need to change
	_IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex] = value ? 1u:0u;
	barrier();
	memoryBarrierShared();
	IRR_GLSL_WORKGROUP_REDUCE_IN_SHARED(kOptimalWorkgroupSize,&=)

	return _IRR_GLSL_SCRATCH_SHARED_DEFINED_[0u]!=0u;
}

bool irr_glsl_workgroupAny(in bool value)
{
	// TODO: Optimization using subgroupAny in an ifdef IRR_GL_something (need to do feature mapping first), probably only first 2 lines need to change
	_IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex] = value ? 1u:0u;
	barrier();
	memoryBarrierShared();
	IRR_GLSL_WORKGROUP_REDUCE_IN_SHARED(kOptimalWorkgroupSize,|=)

	return _IRR_GLSL_SCRATCH_SHARED_DEFINED_[0u]!=0u;
}

/** TODO @Cyprian or @Anastazluk
bool irr_glsl_workgroupAllEqual(in bool val);
float irr_glsl_workgroupAllEqual(in float val);
uint irr_glsl_workgroupAllEqual(in uint val);
int irr_glsl_workgroupAllEqual(in int val);
**/



//! workgroup/ballot.glsl
#include <irr/builtin/glsl/macros.glsl>
#include <irr/builtin/glsl/math/typeless_arithmetic.glsl>

// TODO: depending on subgroup extensions available this will vary in size (usually divided by the subgroup size lower bound)
#define _IRR_GLSL_WORKGROUP_BALLOT_SHARED_SIZE_NEEDED_  (IRR_GLSL_ROUND_UP_POT(_IRR_GLSL_WORKGROUP_SIZE_)*2u)

#ifdef _IRR_GLSL_SCRATCH_SHARED_DEFINED_
/* can't get this to work either
	#if _IRR_GLSL_SCRATCH_SHARED_SIZE_DEFINED_<_IRR_GLSL_WORKGROUP_BALLOT_SHARED_SIZE_NEEDED_
		#error "Not enough shared memory declared"
	#endif
*/
#else
	#if _IRR_GLSL_WORKGROUP_BALLOT_SHARED_SIZE_NEEDED_>0
		#define _IRR_GLSL_SCRATCH_SHARED_DEFINED_ irr_glsl_workgroupBallotScratchShared
		shared uint _IRR_GLSL_SCRATCH_SHARED_DEFINED_[_IRR_GLSL_WORKGROUP_BALLOT_SHARED_SIZE_NEEDED_];
	#endif
#endif

#define irr_glsl_workgroupBallot_impl_getDWORD(IX) (IX>>5u)
#define irr_glsl_workgroupBallot_impl_BitfieldDWORDs irr_glsl_workgroupBallot_impl_getDWORD(_IRR_GLSL_WORKGROUP_SIZE_+31u))

// puts the result into shared memory at offsets [0,_IRR_GLSL_WORKGROUP_SIZE_/32)
void irr_glsl_workgroupBallot(in bool value)
{
	// TODO: Optimization using subgroupBallot in an ifdef IRR_GL_something (need to do feature mapping first)
	if (gl_LocalInvocationIndex<irr_glsl_workgroupBallot_impl_BitfieldDWORDs)
		_IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex] = 0u;
	barrier();
	memoryBarrierShared();
	atomicOr(_IRR_GLSL_SCRATCH_SHARED_DEFINED_[irr_glsl_workgroupBallot_impl_getDWORD(gl_LocalInvocationIndex)],1u<<(gl_LocalInvocationIndex&31u));
	barrier();
}

// the ballot is expected to be in _IRR_GLSL_SCRATCH_SHARED_DEFINED_ at offsets [0,_IRR_GLSL_WORKGROUP_SIZE_/32)
bool irr_glsl_workgroupBallotBitExtract(in uint index)
{
	return (_IRR_GLSL_SCRATCH_SHARED_DEFINED_[irr_glsl_workgroupBallot_impl_getDWORD(index)]&(1u<<(index&31u)))!=0u;
}
bool irr_glsl_workgroupInverseBallot()
{
	return irr_glsl_workgroupBallotBitExtract(gl_LocalInvocationIndex);
}

uint irr_glsl_workgroupBallotBitCount()
{
	if (gl_LocalInvocationIndex<irr_glsl_workgroupBallot_impl_BitfieldDWORDs)
		_IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex] = bitcount(_IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex]);
	barrier();
	memoryBarrierShared();
	// TODO: optimize with `subgroupAdd`
	IRR_GLSL_WORKGROUP_REDUCE_IN_SHARED(irr_glsl_workgroupBallot_impl_BitfieldDWORDs,+=)

	return _IRR_GLSL_SCRATCH_SHARED_DEFINED_[0u];
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


uint irr_glsl_workgroupBroadcast(in uint val, in uint id)
{
	if (gl_LocalInvocationIndex==id)
		_IRR_GLSL_SCRATCH_SHARED_DEFINED_[0u] = val;
	barrier();
	memoryBarrierShared();
	return _IRR_GLSL_SCRATCH_SHARED_DEFINED_[0u];
}

uint irr_glsl_workgroupBroadcastFirst(in uint val) { return irr_glsl_workgroupBroadcast(val,0u); }

/** TODO @Hazardu, @Przemog or @Anastazluk
// these could use optimization from `bitcount` on shared memory, then a part-sized arithmetic scan
uint irr_glsl_workgroupBallotFindLSB();
uint irr_glsl_workgroupBallotFindMSB();

bool irr_glsl_workgroupBroadcast(in bool val, in uint id);
float irr_glsl_workgroupBroadcast(in float val, in uint id);
int irr_glsl_workgroupBroadcast(in int val, in uint id);

bool irr_glsl_workgroupBroadcastFirst(in bool val) {return irr_glsl_workgroupBroadcast(val,0u);}
float irr_glsl_workgroupBroadcastFirst(in float val) {return irr_glsl_workgroupBroadcast(val,0u);}
int irr_glsl_workgroupBroadcastFirst(in int val) {return irr_glsl_workgroupBroadcast(val,0u);}
**/
uint irr_glsl_workgroupBallotScanBitCount_impl(in bool exclusive)
{
	uint localBitfieldBackup;
	if (gl_LocalInvocationIndex<irr_glsl_workgroupBallot_impl_BitfieldDWORDs)
		localBitfieldBackup = _IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex];
	const uint localCount = bitcount(localBitfieldBackup);

	// @Crisspl @Anastazluk @Cyprian take note of this epic hierarchical scan!
	irr_glsl_subgroupScanExclusiveAdd(localCount);
	// TODO: optimize with `subgroupAdd`
	// TODO: IRR_GLSL_WORKGROUP_SCAN_IN_SHARED(localCount,irr_glsl_workgroupBallot_impl_BitfieldDWORDs,+=)

	const uint mask = 0xffffffffu>>((exclusive ? 32u:31u)-(gl_LocalInvocationIndex&31u));
	return _IRR_GLSL_SCRATCH_SHARED_DEFINED_[irr_glsl_workgroupBallot_impl_getDWORD(gl_LocalInvocationIndex)]+bitcount(localBitfieldBackup&mask);
}