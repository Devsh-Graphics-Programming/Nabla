#ifndef _IRR_BUILTIN_GLSL_WORKGROUP_ARITHMETIC_INCLUDED_
#define _IRR_BUILTIN_GLSL_WORKGROUP_ARITHMETIC_INCLUDED_


#include <irr/builtin/glsl/workgroup/clustered.glsl>


/*
#ifdef GL_KHR_subgroup_arithmetic


// TODO: specialize for constexpr case (also remove the ugly `+4` its just the "number of passes" for the scan hierarchy
#define _IRR_GLSL_WORKGROUP_ARITHMETIC_SHARED_SIZE_NEEDED_  (_IRR_GLSL_WORKGROUP_SIZE_/(irr_glsl_MinSubgroupSize-1u)+4u+irr_glsl_MaxSubgroupSize)

#define CONDITIONAL_BARRIER


#else
*/

// this is always greater than the case with native subgroup stuff, TODO: is it correct for small workgroups?
#define _IRR_GLSL_WORKGROUP_ARITHMETIC_SHARED_SIZE_NEEDED_  (_IRR_GLSL_SUBGROUP_ARITHMETIC_EMULATION_SHARED_SIZE_NEEDED_)

#define CONDITIONAL_BARRIER barrier();


//#endif


#ifdef _IRR_GLSL_SCRATCH_SHARED_DEFINED_
	#if IRR_GLSL_EVAL(_IRR_GLSL_SCRATCH_SHARED_SIZE_DEFINED_)<IRR_GLSL_EVAL(_IRR_GLSL_WORKGROUP_ARITHMETIC_SHARED_SIZE_NEEDED_)
		#error "Not enough shared memory declared for workgroup arithmetic!"
	#endif
#else
	#define _IRR_GLSL_SCRATCH_SHARED_DEFINED_ irr_glsl_workgroupArithmeticScratchShared
	shared uint _IRR_GLSL_SCRATCH_SHARED_DEFINED_[_IRR_GLSL_WORKGROUP_ARITHMETIC_SHARED_SIZE_NEEDED_];
#endif


// reduction
#define IRR_GLSL_WORKGROUP_REDUCE(CONV,SUBGROUP_OP,VALUE,IDENTITY,INVCONV) { \
		SUBGROUP_SCRATCH_CLEAR(INVCONV(IDENTITY)) \
	} \
	IRR_GLSL_WORKGROUP_COMMON_IMPL_HEAD(CONV,SUBGROUP_OP,SUBGROUP_OP,VALUE,IDENTITY,INVCONV,_IRR_GLSL_WORKGROUP_SIZE_) \
	} \
	CONDITIONAL_BARRIER \
	return CONV(irr_glsl_workgroupBroadcast(scan,lastInvocationInLevel))


uint irr_glsl_workgroupAdd(in uint val)
{
	IRR_GLSL_WORKGROUP_REDUCE(irr_glsl_identityFunction,irr_glsl_subgroupInclusiveAdd_impl,val,0u,irr_glsl_identityFunction);
}
int irr_glsl_workgroupAdd(in int val)
{
	return int(irr_glsl_workgroupAdd(uint(val)));
}
float irr_glsl_workgroupAdd(in float val)
{
	IRR_GLSL_WORKGROUP_REDUCE(uintBitsToFloat,irr_glsl_subgroupInclusiveAdd_impl,val,0.0,floatBitsToUint);
}


// scan
#define IRR_GLSL_WORKGROUP_SCAN(CONV,OP,FIRST_SUBGROUP_OP,SECOND_SUBGROUP_OP,VALUE,IDENTITY,INVCONV) { \
		SUBGROUP_SCRATCH_CLEAR(INVCONV(IDENTITY)) \
	} \
	IRR_GLSL_WORKGROUP_COMMON_IMPL_HEAD(CONV,FIRST_SUBGROUP_OP,SECOND_SUBGROUP_OP,VALUE,IDENTITY,INVCONV,_IRR_GLSL_WORKGROUP_SIZE_) \
	IRR_GLSL_WORKGROUP_SCAN_IMPL_TAIL(CONV,OP,INVCONV) \
	return CONV(firstLevelScan);


uint irr_glsl_workgroupInclusiveAdd(in uint val)
{
	IRR_GLSL_WORKGROUP_SCAN(irr_glsl_identityFunction,irr_glsl_add,irr_glsl_subgroupInclusiveAdd_impl,irr_glsl_subgroupExclusiveAdd_impl,val,0u,irr_glsl_identityFunction);
}
int irr_glsl_workgroupInclusiveAdd(in int val)
{
	return int(irr_glsl_workgroupInclusiveAdd(uint(val)));
}
uint irr_glsl_workgroupExclusiveAdd(in uint val)
{
	IRR_GLSL_WORKGROUP_SCAN(irr_glsl_identityFunction,irr_glsl_add,irr_glsl_subgroupExclusiveAdd_impl,irr_glsl_subgroupExclusiveAdd_impl,val,0u,irr_glsl_identityFunction);
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

#undef CONDITIONAL_BARRIER

#endif
