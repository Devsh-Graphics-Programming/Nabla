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


#define DECLARE_OVERLOAD_WITH_BARRIERS(TYPE,FUNC_NAME) TYPE irr_glsl_##FUNC_NAME (in TYPE val) \
{ \
	barrier(); \
	const TYPE retval = irr_glsl_##FUNC_NAME##_noBarriers (val); \
	barrier(); \
	return retval; \
}


 
#ifdef _IRR_GLSL_SCRATCH_SHARED_DEFINED_
	#if IRR_GLSL_EVAL(_IRR_GLSL_SCRATCH_SHARED_SIZE_DEFINED_)<IRR_GLSL_EVAL(_IRR_GLSL_WORKGROUP_ARITHMETIC_SHARED_SIZE_NEEDED_)
		#error "Not enough shared memory declared for workgroup arithmetic!"
	#endif
#else
	#define _IRR_GLSL_SCRATCH_SHARED_DEFINED_ irr_glsl_workgroupArithmeticScratchShared
	shared uint _IRR_GLSL_SCRATCH_SHARED_DEFINED_[_IRR_GLSL_WORKGROUP_ARITHMETIC_SHARED_SIZE_NEEDED_];
#endif



// reduction
#define IRR_GLSL_WORKGROUP_REDUCE(CONV,INCLUSIVE_SUBGROUP_OP,VALUE,IDENTITY,INVCONV) IRR_GLSL_WORKGROUP_COMMON_IMPL_HEAD(CONV,INCLUSIVE_SUBGROUP_OP,VALUE,IDENTITY,INVCONV,_IRR_GLSL_WORKGROUP_SIZE_,;); \
	barrier(); \
	return CONV(irr_glsl_workgroupBroadcast_noBarriers(scan,lastInvocationInLevel))


uint irr_glsl_workgroupAnd_noBarriers(in uint val)
{
	IRR_GLSL_WORKGROUP_REDUCE(irr_glsl_identityFunction,irr_glsl_subgroupInclusiveAnd_impl,val,~0u,irr_glsl_identityFunction);
}
int irr_glsl_workgroupAnd_noBarriers(in int val)
{
	return int(irr_glsl_workgroupAnd_noBarriers(uint(val)));
}
float irr_glsl_workgroupAnd_noBarriers(in float val)
{
	IRR_GLSL_WORKGROUP_REDUCE(uintBitsToFloat,irr_glsl_subgroupInclusiveAnd_impl,val,0.0,floatBitsToUint);
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupAnd)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupAnd)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupAnd)


uint irr_glsl_workgroupOr_noBarriers(in uint val)
{
	IRR_GLSL_WORKGROUP_REDUCE(irr_glsl_identityFunction,irr_glsl_subgroupInclusiveOr_impl,val,0u,irr_glsl_identityFunction);
}
int irr_glsl_workgroupOr_noBarriers(in int val)
{
	return int(irr_glsl_workgroupOr_noBarriers(uint(val)));
}
float irr_glsl_workgroupOr_noBarriers(in float val)
{
	IRR_GLSL_WORKGROUP_REDUCE(uintBitsToFloat,irr_glsl_subgroupInclusiveOr_impl,val,0.0,floatBitsToUint);
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupOr)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupOr)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupOr)


uint irr_glsl_workgroupXor_noBarriers(in uint val)
{
	IRR_GLSL_WORKGROUP_REDUCE(irr_glsl_identityFunction,irr_glsl_subgroupInclusiveXor_impl,val,0u,irr_glsl_identityFunction);
}
int irr_glsl_workgroupXor_noBarriers(in int val)
{
	return int(irr_glsl_workgroupXor_noBarriers(uint(val)));
}
float irr_glsl_workgroupXor_noBarriers(in float val)
{
	IRR_GLSL_WORKGROUP_REDUCE(uintBitsToFloat,irr_glsl_subgroupInclusiveXor_impl,val,0.0,floatBitsToUint);
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupXor)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupXor)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupXor)


uint irr_glsl_workgroupAdd_noBarriers(in uint val)
{
	IRR_GLSL_WORKGROUP_REDUCE(irr_glsl_identityFunction,irr_glsl_subgroupInclusiveAdd_impl,val,0u,irr_glsl_identityFunction);
}
int irr_glsl_workgroupAdd_noBarriers(in int val)
{
	return int(irr_glsl_workgroupAdd_noBarriers(uint(val)));
}
float irr_glsl_workgroupAdd_noBarriers(in float val)
{
	IRR_GLSL_WORKGROUP_REDUCE(uintBitsToFloat,irr_glsl_subgroupInclusiveAdd_impl,val,0.0,floatBitsToUint);
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupAdd)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupAdd)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupAdd)


uint irr_glsl_workgroupMul_noBarriers(in uint val)
{
	IRR_GLSL_WORKGROUP_REDUCE(irr_glsl_identityFunction,irr_glsl_subgroupInclusiveMul_impl,val,1u,irr_glsl_identityFunction);
}
int irr_glsl_workgroupMul_noBarriers(in int val)
{
	return int(irr_glsl_workgroupMul_noBarriers(uint(val)));
}
float irr_glsl_workgroupMul_noBarriers(in float val)
{
	IRR_GLSL_WORKGROUP_REDUCE(uintBitsToFloat,irr_glsl_subgroupInclusiveMul_impl,val,1.0,floatBitsToUint);
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupMul)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupMul)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupMul)


uint irr_glsl_workgroupMin_noBarriers(in uint val)
{
	IRR_GLSL_WORKGROUP_REDUCE(irr_glsl_identityFunction,irr_glsl_subgroupInclusiveMin_impl,val,UINT_MAX,irr_glsl_identityFunction);
}
int irr_glsl_workgroupMin_noBarriers(in int val)
{
	return int(irr_glsl_workgroupMin_noBarriers(uint(val)));
}
float irr_glsl_workgroupMin_noBarriers(in float val)
{
	IRR_GLSL_WORKGROUP_REDUCE(uintBitsToFloat,irr_glsl_subgroupInclusiveMin_impl,val,FLT_INF,floatBitsToUint);
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupMin)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupMin)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupMin)


uint irr_glsl_workgroupMax_noBarriers(in uint val)
{
	IRR_GLSL_WORKGROUP_REDUCE(irr_glsl_identityFunction,irr_glsl_subgroupInclusiveMax_impl,val,UINT_MIN,irr_glsl_identityFunction);
}
int irr_glsl_workgroupMax_noBarriers(in int val)
{
	return int(irr_glsl_workgroupMax_noBarriers(uint(val)));
}
float irr_glsl_workgroupMax_noBarriers(in float val)
{
	IRR_GLSL_WORKGROUP_REDUCE(uintBitsToFloat,irr_glsl_subgroupInclusiveMax_impl,val,-FLT_INF,floatBitsToUint);
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupMax)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupMax)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupMax)



// scan
#define IRR_GLSL_WORKGROUP_SCAN(EXCLUSIVE,CONV,OP,INCLUSIVE_SUBGROUP_OP,VALUE,IDENTITY,INVCONV) IRR_GLSL_WORKGROUP_COMMON_IMPL_HEAD(CONV,INCLUSIVE_SUBGROUP_OP,VALUE,IDENTITY,INVCONV,_IRR_GLSL_WORKGROUP_SIZE_,IRR_GLSL_WORKGROUP_SCAN_IMPL_LOOP_POSTLUDE) \
	IRR_GLSL_WORKGROUP_SCAN_IMPL_TAIL(EXCLUSIVE,CONV,INCLUSIVE_SUBGROUP_OP,INVCONV,OP)



uint irr_glsl_workgroupInclusiveAdd_noBarriers(in uint val)
{
	IRR_GLSL_WORKGROUP_SCAN(false,irr_glsl_identityFunction,irr_glsl_add,irr_glsl_subgroupInclusiveAdd_impl,val,0u,irr_glsl_identityFunction);
}
int irr_glsl_workgroupInclusiveAdd_noBarriers(in int val)
{
	return int(irr_glsl_workgroupInclusiveAdd_noBarriers(uint(val)));
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupInclusiveAdd)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupInclusiveAdd)


uint irr_glsl_workgroupExclusiveAdd_noBarriers(in uint val)
{
	IRR_GLSL_WORKGROUP_SCAN(true,irr_glsl_identityFunction,irr_glsl_add,irr_glsl_subgroupInclusiveAdd_impl,val,0u,irr_glsl_identityFunction);
}
int irr_glsl_workgroupExclusiveAdd_noBarriers(in int val)
{
	return int(irr_glsl_workgroupExclusiveAdd_noBarriers(uint(val)));
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupExclusiveAdd)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupExclusiveAdd)



/** TODO @Hazardu or @Przemog or recruitment task
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

#undef DECLARE_OVERLOAD_WITH_BARRIERS

#undef CONDITIONAL_BARRIER

#endif
