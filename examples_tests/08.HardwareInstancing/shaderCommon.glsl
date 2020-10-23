// TODO: move to macro GLSL header
#define IRR_GLSL_IS_POT(v) (v&(v-1u))
#define IRR_GLSL_ROUND_UP_POT(v) (1u + \
(((((((((v) - 1u) | (((v) - 1u) >> 0x10u) | \
      (((v) - 1u) | (((v) - 1u) >> 0x10u) >> 0x08u)) | \
     ((((v) - 1u) | (((v) - 1u) >> 0x10u) | \
      (((v) - 1u) | (((v) - 1u) >> 0x10u) >> 0x08u)) >> 0x04u))) | \
   ((((((v) - 1u) | (((v) - 1u) >> 0x10u) | \
      (((v) - 1u) | (((v) - 1u) >> 0x10u) >> 0x08u)) | \
     ((((v) - 1u) | (((v) - 1u) >> 0x10u) | \
      (((v) - 1u) | (((v) - 1u) >> 0x10u) >> 0x08u)) >> 0x04u))) >> 0x02u))) | \
 ((((((((v) - 1u) | (((v) - 1u) >> 0x10u) | \
      (((v) - 1u) | (((v) - 1u) >> 0x10u) >> 0x08u)) | \
     ((((v) - 1u) | (((v) - 1u) >> 0x10u) | \
      (((v) - 1u) | (((v) - 1u) >> 0x10u) >> 0x08u)) >> 0x04u))) | \
   ((((((v) - 1u) | (((v) - 1u) >> 0x10u) | \
      (((v) - 1u) | (((v) - 1u) >> 0x10u) >> 0x08u)) | \
     ((((v) - 1u) | (((v) - 1u) >> 0x10u) | \
      (((v) - 1u) | (((v) - 1u) >> 0x10u) >> 0x08u)) >> 0x04u))) >> 0x02u))) >> 0x01u))))

// TODO: move this out to a builtin but after a major refactor
// basically implement all of https://github.com/KhronosGroup/GLSL/blob/master/extensions/khr/GL_KHR_shader_subgroup.txt for subgroups 



//! subgroup/emulation.glsl
#ifndef _IRR_GLSL_WORKGROUP_SIZE_
#error "User needs to let us know the size of the workgroup via _IRR_GLSL_WORKGROUP_SIZE_!"
#endif

#define irr_glsl_MinSubgroupSize 4u
#define irr_glsl_MaxSubgroupSize 128u
// TODO: define this properly from gl_SubgroupSize and available extensions
#define IRR_GLSL_SUBGROUP_SIZE_IS_CONSTANT
#define irr_glsl_SubgroupSize 4u

#if irr_glsl_SubgroupSize<irr_glsl_MinSubgroupSize
	#error "Something went very wrong when figuring out irr_glsl_SubgroupSize!"
#endif
#define irr_glsl_HalfSubgroupSize (irr_glsl_SubgroupSize>>1u)

/*
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

#define irr_glsl_subgroupScanExclusiveAdd subgroupExclusiveAdd

#else
*/

// TODO: depending on subgroup extensions available this will vary in size (usually divided by the subgroup size lower bound)
#define _IRR_GLSL_WORKGROUP_VOTE_SHARED_SIZE_NEEDED_  IRR_GLSL_ROUND_UP_POT(_IRR_GLSL_WORKGROUP_SIZE_)

#ifdef _IRR_GLSL_SCRATCH_SHARED_DEFINED_
/* can't get this to work either
	#if _IRR_GLSL_SCRATCH_SHARED_SIZE_DEFINED_<_IRR_GLSL_SUBGROUP_EMULATION_SHARED_SIZE_NEEDED_
		#error "Not enough shared memory declared"
	#endif
*/
#else
	#if _IRR_GLSL_SUBGROUP_EMULATION_SHARED_SIZE_NEEDED_>0
		#define _IRR_GLSL_SCRATCH_SHARED_DEFINED_ irr_glsl_subgroupEmulationScratchShared
		shared uint _IRR_GLSL_SCRATCH_SHARED_DEFINED_[_IRR_GLSL_SUBGROUP_EMULATION_SHARED_SIZE_NEEDED_];
	#endif
#endif
/*
How to avoid bank conflicts
write:	00,01,02,03,    08,09,10,11,	16,17,18,19,    24,25,26,27,    04,05,06,07,    12,13,14,15,    20,21,22,23,    28,29,30,31
read:	30,31,00,01,    06,07,08,09,    14,15,16,17,    22,23,24,25,    02,03,04,05,    10,11,12,13,    18,19,20,21,    26,27,28,29
*/

uint irr_glsl_subgroupScanExclusiveAdd(in uint val)
{
	const uint pseudoSubgroup = gl_LocalInvocationIndex&(irr_glsl_MaxSubgroupSize-irr_glsl_SubgroupSize);
	const uint scratchOffset = (pseudoSubgroup<<1u)+(gl_LocalInvocationIndex&(irr_glsl_SubgroupSize-1u));
	const uint primaryOffset = scratchOffset+irr_glsl_HalfSubgroupSize;
	_IRR_GLSL_SCRATCH_SHARED_DEFINED_[primaryOffset] = val;
	_IRR_GLSL_SCRATCH_SHARED_DEFINED_[scratchOffset] = 0u;
	SUBGROUP_BARRIERS
	_IRR_GLSL_SCRATCH_SHARED_DEFINED_[primaryOffset] += _IRR_GLSL_SCRATCH_SHARED_DEFINED_[primaryOffset-1u];
	SUBGROUP_BARRIERS
	_IRR_GLSL_SCRATCH_SHARED_DEFINED_[primaryOffset] += _IRR_GLSL_SCRATCH_SHARED_DEFINED_[primaryOffset-2u];
	for (uint step=irr_glsl_MinSubgroupSize; step<irr_glsl_SubgroupSize; step<<=1u)
	{
		SUBGROUP_BARRIERS
		_IRR_GLSL_SCRATCH_SHARED_DEFINED_[primaryOffset] += _IRR_GLSL_SCRATCH_SHARED_DEFINED_[primaryOffset-step];
	}
	return _IRR_GLSL_SCRATCH_SHARED_DEFINED_[primaryOffset];
}
//#endif

#undef SUBGROUP_BARRIERS



//! workgroup/basic.glsl
#ifndef _IRR_GLSL_WORKGROUP_SIZE_
#error "User needs to let us know the size of the workgroup via _IRR_GLSL_WORKGROUP_SIZE_!"
#endif

bool irr_glsl_workgroupElect()
{
	return gl_LocalInvocationIndex==0u;
}

// TODO: maybe improve the macro so the if statement performs better
#define IRR_GLSL_WORKGROUP_REDUCE_IN_SHARED(SIZE,OP) { \
		barrier(); \
		memoryBarrierShared(); \
		uint partition = 0x1u<<findMSB(SIZE-1u); \
		if (gl_LocalInvocationIndex<partition && (IRR_GLSL_IS_POT(SIZE) || gl_LocalInvocationIndex+partition<SIZE)) \
			_IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex] OP _IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex+partition]; \
		for (partition>>=1u; partition>0u; partition>>=1u) \
		{ \
			barrier(); \
			memoryBarrierShared(); \
			if (gl_LocalInvocationIndex<partition) \
				_IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex] OP _IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex+partition]; \
		} \
	}

// Hillis-Steele Scan
/*
#define IRR_GLSL_WORKGROUP_SCAN_IN_SHARED(VAR,SIZE,OP,IDENTITY) { \
		const bool invocationInRange = gl_LocalInvocationIndex<SIZE; \
		if (gl_LocalInvocationIndex<(SIZE>>1u)+SIZE) \
			_IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex] = invocationInRange ? VAR:IDENTITY; \
		for (uint step=1u; step<SIZE; step<<=1u) \
		{ \
			barrier(); \
			memoryBarrierShared(); \
			uint read;
			if (gl_LocalInvocationIndex<SIZE && gl_LocalInvocationIndex>=step) \
				read = _IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex-step];
			barrier(); \
			memoryBarrierShared(); \
			if (gl_LocalInvocationIndex<SIZE && gl_LocalInvocationIndex>=step) \
				_IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex] += read; \
		} \
		barrier(); \
		memoryBarrierShared();\
	}
*/

//! workgroup/vote.glsl
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

	IRR_GLSL_WORKGROUP_REDUCE_IN_SHARED(kOptimalWorkgroupSize,&=)

	return _IRR_GLSL_SCRATCH_SHARED_DEFINED_[0u]!=0u;
}

bool irr_glsl_workgroupAny(in bool value)
{
	// TODO: Optimization using subgroupAny in an ifdef IRR_GL_something (need to do feature mapping first), probably only first 2 lines need to change
	_IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex] = value ? 1u:0u;

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

/** TODO @Cyprian or @Anastazluk
// these could use optimization from `bitcount` on shared memory, then a part-sized arithmetic scan
uint irr_glsl_workgroupBallotFindLSB();
uint irr_glsl_workgroupBallotFindMSB();

bool irr_glsl_workgroupBroadcast(in bool val, in uint id);
float irr_glsl_workgroupBroadcast(in float val, in uint id);
uint irr_glsl_workgroupBroadcast(in uint val, in uint id);
int irr_glsl_workgroupBroadcast(in int val, in uint id);

bool irr_glsl_workgroupBroadcastFirst(in bool val) {return irr_glsl_workgroupBroadcast(val,0u);}
float irr_glsl_workgroupBroadcastFirst(in float val) {return irr_glsl_workgroupBroadcast(val,0u);}
uint irr_glsl_workgroupBroadcastFirst(in uint val) {return irr_glsl_workgroupBroadcast(val,0u);}
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



//! workgroup/shuffle.glsl
// TODO: depending on subgroup extensions available this will vary in size (usually divided by the subgroup size lower bound)
#define _IRR_GLSL_WORKGROUP_SHUFFLE_SHARED_SIZE_NEEDED_  IRR_GLSL_ROUND_UP_POT(_IRR_GLSL_WORKGROUP_SIZE_)

#ifdef _IRR_GLSL_SCRATCH_SHARED_DEFINED_
/* can't get this to work either
	#if _IRR_GLSL_SCRATCH_SHARED_SIZE_DEFINED_<_IRR_GLSL_WORKGROUP_SHUFFLE_SHARED_SIZE_NEEDED_
		#error "Not enough shared memory declared"
	#endif
*/
#else
#if _IRR_GLSL_WORKGROUP_SHUFFLE_SHARED_SIZE_NEEDED_>0
#define _IRR_GLSL_SCRATCH_SHARED_DEFINED_ irr_glsl_workgroupShuffleScratchShared
shared uint _IRR_GLSL_SCRATCH_SHARED_DEFINED_[_IRR_GLSL_WORKGROUP_SHUFFLE_SHARED_SIZE_NEEDED_];
#endif
#endif


/** TODO @Cyprian or @Anastazluk you can express all of them in terms of the uint variants to safe yourself the trouble

bool irr_glsl_workgroupShuffle(in bool val, in uint id);
float irr_glsl_workgroupShuffle(in float val, in uint id);
uint irr_glsl_workgroupShuffle(in uint val, in uint id);
int irr_glsl_workgroupShuffle(in int val, in uint id);

bool irr_glsl_workgroupShuffleXor(in bool val, in uint mask);
float irr_glsl_workgroupShuffleXor(in float val, in uint mask);
uint irr_glsl_workgroupShuffleXor(in uint val, in uint mask);
int irr_glsl_workgroupShuffleXor(in int val, in uint mask);
*/



//! workgroup/shuffle_relative.glsl
// TODO: depending on subgroup extensions available this will vary in size (usually divided by the subgroup size lower bound)
#define _IRR_GLSL_WORKGROUP_SHUFFLE_RELATIVE_SHARED_SIZE_NEEDED_  IRR_GLSL_ROUND_UP_POT(_IRR_GLSL_WORKGROUP_SIZE_)

#ifdef _IRR_GLSL_SCRATCH_SHARED_DEFINED_
/* can't get this to work either
	#if _IRR_GLSL_SCRATCH_SHARED_SIZE_DEFINED_<_IRR_GLSL_WORKGROUP_SHUFFLE_RELATIVE_SHARED_SIZE_NEEDED_
		#error "Not enough shared memory declared"
	#endif
*/
#else
#if _IRR_GLSL_WORKGROUP_SHUFFLE_RELATIVE_SHARED_SIZE_NEEDED_>0
#define _IRR_GLSL_SCRATCH_SHARED_DEFINED_ irr_glsl_workgroupShuffleRelativeScratchShared
shared uint _IRR_GLSL_SCRATCH_SHARED_DEFINED_[_IRR_GLSL_WORKGROUP_SHUFFLE_RELATIVE_SHARED_SIZE_NEEDED_];
#endif
#endif


/** TODO @Cyprian or @Anastazluk you can express all of them in terms of the uint variants to safe yourself the trouble

bool irr_glsl_workgroupShuffleUp(in bool val, in uint delta);
float irr_glsl_workgroupShuffleUp(in float val, in uint delta);
uint irr_glsl_workgroupShuffleUp(in uint val, in uint delta);
int irr_glsl_workgroupShuffleUp(in int val, in uint delta);

bool irr_glsl_workgroupShuffleDown(in bool val, in uint delta);
float irr_glsl_workgroupShuffleDown(in float val, in uint delta);
uint irr_glsl_workgroupShuffleDown(in uint val, in uint delta);
int irr_glsl_workgroupShuffleDown(in int val, in uint delta);
*/



//! workgroup/arithmetic.glsl
// TODO: depending on subgroup extensions available this will vary in size (usually divided by the subgroup size lower bound)
#define _IRR_GLSL_WORKGROUP_ARITHMETIC_SHARED_SIZE_NEEDED_  (IRR_GLSL_ROUND_UP_POT(_IRR_GLSL_WORKGROUP_SIZE_)*2u)

#ifdef _IRR_GLSL_SCRATCH_SHARED_DEFINED_
/* can't get this to work either
	#if _IRR_GLSL_SCRATCH_SHARED_SIZE_DEFINED_<_IRR_GLSL_WORKGROUP_ARITHMETIC_SHARED_SIZE_NEEDED_
		#error "Not enough shared memory declared"
	#endif
*/
#else
	#if _IRR_GLSL_WORKGROUP_ARITHMETIC_SHARED_SIZE_NEEDED_>0
		#define _IRR_GLSL_SCRATCH_SHARED_DEFINED_ irr_glsl_workgroupArithmeticScratchShared
		shared uint _IRR_GLSL_SCRATCH_SHARED_DEFINED_[_IRR_GLSL_WORKGROUP_ARITHMETIC_SHARED_SIZE_NEEDED_];
	#endif
#endif


uint irr_glsl_workgroupInclusiveAdd(in uint val)
{
#error
	return _IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationID];
}

int irr_glsl_workgroupInclusiveAdd(in int val)
{
	return int(irr_glsl_workgroupInclusiveAdd(uint(val)));
}

/** TODO @Cyprian or @Anastazluk
bool irr_glsl_workgroupAnd(in bool val);
float irr_glsl_workgroupAnd(in float val);
uint irr_glsl_workgroupAnd(in uint val);
int irr_glsl_workgroupAnd(in int val);

bool irr_glsl_workgroupXor(in bool val);
float irr_glsl_workgroupXor(in float val);
uint irr_glsl_workgroupXor(in uint val);
int irr_glsl_workgroupXor(in int val);

bool irr_glsl_workgroupOr(in bool val);
float irr_glsl_workgroupOr(in float val);
uint irr_glsl_workgroupOr(in uint val);
int irr_glsl_workgroupOr(in int val);

float irr_glsl_workgroupAdd(in float val);
uint irr_glsl_workgroupAdd(in uint val);
int irr_glsl_workgroupAdd(in int val);

float irr_glsl_workgroupMul(in float val);
uint irr_glsl_workgroupMul(in uint val);
int irr_glsl_workgroupMul(in int val);

float irr_glsl_workgroupMin(in float val);
uint irr_glsl_workgroupMin(in uint val);
int irr_glsl_workgroupMin(in int val);

float irr_glsl_workgroupMax(in float val);
uint irr_glsl_workgroupMax(in uint val);
int irr_glsl_workgroupMax(in int val);


bool irr_glsl_workgroupInclusiveAnd(in bool val);
float irr_glsl_workgroupInclusiveAnd(in float val);
uint irr_glsl_workgroupInclusiveAnd(in uint val);
int irr_glsl_workgroupInclusiveAnd(in int val);
bool irr_glsl_workgroupExclusiveAnd(in bool val);
float irr_glsl_workgroupExclusiveAnd(in float val);
uint irr_glsl_workgroupExclusiveAnd(in uint val);
int irr_glsl_workgroupExclusiveAnd(in int val);

bool irr_glsl_workgroupInclusiveXor(in bool val);
float irr_glsl_workgroupInclusiveXor(in float val);
uint irr_glsl_workgroupInclusiveXor(in uint val);
int irr_glsl_workgroupInclusiveXor(in int val);
bool irr_glsl_workgroupExclusiveXor(in bool val);
float irr_glsl_workgroupExclusiveXor(in float val);
uint irr_glsl_workgroupExclusiveXor(in uint val);
int irr_glsl_workgroupExclusiveXor(in int val);

bool irr_glsl_workgroupInclusiveOr(in bool val);
float irr_glsl_workgroupInclusiveOr(in float val);
uint irr_glsl_workgroupInclusiveOr(in uint val);
int irr_glsl_workgroupInclusiveOr(in int val);
bool irr_glsl_workgroupExclusiveOr(in bool val);
float irr_glsl_workgroupExclusiveOr(in float val);
uint irr_glsl_workgroupExclusiveOr(in uint val);
int irr_glsl_workgroupExclusiveOr(in int val);

float irr_glsl_workgroupInclusiveAdd(in float val);
float irr_glsl_workgroupExclusiveAdd(in float val);
uint irr_glsl_workgroupExclusiveAdd(in uint val);
int irr_glsl_workgroupExclusiveAdd(in int val);

float irr_glsl_workgroupInclusiveMul(in float val);
uint irr_glsl_workgroupInclusiveMul(in uint val);
int irr_glsl_workgroupInclusiveMul(in int val);
float irr_glsl_workgroupExclusiveMul(in float val);
uint irr_glsl_workgroupExclusiveMul(in uint val);
int irr_glsl_workgroupExclusiveMul(in int val);

float irr_glsl_workgroupInclusiveMin(in float val);
uint irr_glsl_workgroupInclusiveMin(in uint val);
int irr_glsl_workgroupInclusiveMin(in int val);
float irr_glsl_workgroupExclusiveMin(in float val);
uint irr_glsl_workgroupExclusiveMin(in uint val);
int irr_glsl_workgroupExclusiveMin(in int val);

float irr_glsl_workgroupInclusiveMax(in float val);
uint irr_glsl_workgroupInclusiveMax(in uint val);
int irr_glsl_workgroupInclusiveMax(in int val);
float irr_glsl_workgroupExclusiveMax(in float val);
uint irr_glsl_workgroupExclusiveMax(in uint val);
int irr_glsl_workgroupExclusiveMax(in int val);
**/



//! workgroup/clustered.glsl
// TODO: depending on subgroup extensions available this will vary in size (usually divided by the subgroup size lower bound)
#define _IRR_GLSL_WORKGROUP_CLUSTERED_SHARED_SIZE_NEEDED_  IRR_GLSL_ROUND_UP_POT(_IRR_GLSL_WORKGROUP_SIZE_)

#ifdef _IRR_GLSL_SCRATCH_SHARED_DEFINED_
/* can't get this to work either
	#if _IRR_GLSL_SCRATCH_SHARED_SIZE_DEFINED_<_IRR_GLSL_WORKGROUP_CLUSTERED_SHARED_SIZE_NEEDED_
		#error "Not enough shared memory declared"
	#endif
*/
#else
	#if _IRR_GLSL_WORKGROUP_CLUSTERED_SHARED_SIZE_NEEDED_>0
		#define _IRR_GLSL_SCRATCH_SHARED_DEFINED_ irr_glsl_workgroupClusteredScratchShared
		shared uint _IRR_GLSL_SCRATCH_SHARED_DEFINED_[_IRR_GLSL_WORKGROUP_CLUSTERED_SHARED_SIZE_NEEDED_];
	#endif
#endif

/** TODO @Cyprian or @Anastazluk
float irr_glsl_workgroupClusteredAnd(in float val);
uint irr_glsl_workgroupClusteredAnd(in uint val);
int irr_glsl_workgroupClusteredAnd(in int val);

bool irr_glsl_workgroupClusteredXor(in bool val);
float irr_glsl_workgroupClusteredXor(in float val);
uint irr_glsl_workgroupClusteredXor(in uint val);
int irr_glsl_workgroupClusteredXor(in int val);

bool irr_glsl_workgroupClusteredOr(in bool val);
float irr_glsl_workgroupClusteredOr(in float val);
uint irr_glsl_workgroupClusteredOr(in uint val);
int irr_glsl_workgroupClusteredOr(in int val);

bool irr_glsl_workgroupClusteredAdd(in bool val);
float irr_glsl_workgroupClusteredAdd(in float val);
uint irr_glsl_workgroupClusteredAdd(in uint val);
int irr_glsl_workgroupClusteredAdd(in int val);

// mul and min/max dont need boolean variants, since they're achievable with And and Or
float irr_glsl_workgroupClusteredMul(in float val);
uint irr_glsl_workgroupClusteredMul(in uint val);
int irr_glsl_workgroupClusteredMul(in int val);

float irr_glsl_workgroupClusteredMin(in float val);
uint irr_glsl_workgroupClusteredMin(in uint val);
int irr_glsl_workgroupClusteredMin(in int val);

float irr_glsl_workgroupClusteredMax(in float val);
uint irr_glsl_workgroupClusteredMax(in uint val);
int irr_glsl_workgroupClusteredMax(in int val);
*/