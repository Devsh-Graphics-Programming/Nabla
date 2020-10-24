#ifndef _IRR_BUILTIN_GLSL_WORKGROUP_ARITHMETIC_INCLUDED_
#define _IRR_BUILTIN_GLSL_WORKGROUP_ARITHMETIC_INCLUDED_


#include <irr/builtin/glsl/subgroup/arithmetic_portability.glsl>
#define _IRR_GLSL_WORKGROUP_ARITHMETIC_SHARED_SIZE_NEEDED_  ((_IRR_GLSL_WORKGROUP_SIZE_+irr_glsl_MinSubgroupSize-1)/irr_glsl_MinSubgroupSize)

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

// if using native subgroup operations we don't need to worry about stepping on our own shared memory
/*
#ifdef GL_KHR_subgroup_arithmetic

#define CONDITIONAL_BARRIER

#else
*/

#define CONDITIONAL_BARRIER barrier();

//#endif

/*
If `GL_KHR_subgroup_arithmetic` is not available then these functions require emulated subgroup operations, which in turn means that if you're using the
`irr_glsl_workgroupOp`s then the workgroup size must not be smaller than half a subgroup but having workgroups smaller than a subgroup is extremely bad practice.


propagate if IX%S==S-1

[S^3+1,S^4]
S-1, 2S-1, ... N-1 or S^4-1 contain first level reductions
they should go into 0,1,...,S^2 or 0,1,...,S^3-1
reduction = S^2+1 or S^3

S-1,2S-1,...,S^2+1 or S-1,2S-1,...,S^3-1 contain second level reductions
they should go into 0,1,...,S or 0,1,...,S^2-1
reduction = S+1 or S^2

S-1,S or S-1,2S-1,...,S^2-1 contain third level reductions
they should go into 0,1 or 0,1,...,S-1
reduction = 2 or S

0 or S-1 contain the fourth level reduction

return reduction-1
*/

// TODO: unroll the while 5-times
#define IRR_GLSL_WORKGROUP_REDUCE(CONV,OP,VALUE,IDENTITY) DUMB
uint irr_glsl_workgroupAdd(in uint val)
{
	const uint loMask = irr_glsl_SubgroupSize-1u;
	{
		const uint hiMask = ~loMask;
		const uint maxItemsToClear = ((_IRR_GLSL_WORKGROUP_SIZE_+loMask)&hiMask)>>1u;
		if (gl_LocalInvocationIndex<maxItemsToClear)
		{
			const uint halfMask = loMask>>1u;
			const uint clearIndex = (gl_LocalInvocationIndex&(~halfMask))*3u+(gl_LocalInvocationIndex&halfMask);
			_IRR_GLSL_SCRATCH_SHARED_DEFINED_[clearIndex] = IDENTITY;
		}
		barrier();
		memoryBarrierShared();
	}
	const bool propagateReduction = (gl_LocalInvocationIndex&loMask)==loMask;
	const uint outTempIx = gl_LocalInvocationIndex/irr_glsl_SubgroupSize;
	uint sub = irr_glsl_subgroupInclusiveAdd_impl(false,val);
	uint reduction = _IRR_GLSL_WORKGROUP_SIZE_;
	while (reduction>irr_glsl_SubgroupSize)
	{
		CONDITIONAL_BARRIER
		if (propagateReduction || gl_LocalInvocationIndex==) // invocation maps to last
			_IRR_GLSL_SCRATCH_SHARED_DEFINED_[outTempIx] = sub;
		barrier();
		memoryBarrierShared();
		reduction = (reduction+loMask)/irr_glsl_SubgroupSize;
		if (gl_LocalInvocationIndex<reduction)
			sub = irr_glsl_subgroupInclusiveAdd_impl(false,_IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex]);
	}
	CONDITIONAL_BARRIER
	return irr_glsl_workgroupBroadcast(sub,reduction-1u);
}



uint irr_glsl_workgroupInclusiveAdd(in uint val)
{
}
int irr_glsl_workgroupInclusiveAdd(in int val)
{
	return int(irr_glsl_workgroupInclusiveAdd(uint(val)));
}
uint irr_glsl_workgroupExclusiveAdd(in uint val)
{
}
int irr_glsl_workgroupExclusiveAdd(in int val)
{
	return int(irr_glsl_workgroupExclusiveAdd(uint(val)));
}



/** TODO @Hazardu or @Przemog or recruitment task
float irr_glsl_workgroupAnd(in float val);
uint irr_glsl_workgroupAnd(in uint val);
int irr_glsl_workgroupAnd(in int val);

float irr_glsl_workgroupXor(in float val);
uint irr_glsl_workgroupXor(in uint val);
int irr_glsl_workgroupXor(in int val);

float irr_glsl_workgroupOr(in float val);
uint irr_glsl_workgroupOr(in uint val);
int irr_glsl_workgroupOr(in int val);

float irr_glsl_workgroupAdd(in float val);
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

bool irr_glsl_workgroupInclusiveAdd(in bool val);
float irr_glsl_workgroupInclusiveAdd(in float val);
bool irr_glsl_workgroupExclusiveAdd(in bool val);
float irr_glsl_workgroupExclusiveAdd(in float val);

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

#endif
