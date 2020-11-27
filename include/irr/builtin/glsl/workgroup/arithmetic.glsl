#ifndef _IRR_BUILTIN_GLSL_WORKGROUP_ARITHMETIC_INCLUDED_
#define _IRR_BUILTIN_GLSL_WORKGROUP_ARITHMETIC_INCLUDED_


#include <irr/builtin/glsl/workgroup/shared_arithmetic.glsl>



#ifdef _IRR_GLSL_SCRATCH_SHARED_DEFINED_
	#if IRR_GLSL_EVAL(_IRR_GLSL_SCRATCH_SHARED_SIZE_DEFINED_)<IRR_GLSL_EVAL(_IRR_GLSL_WORKGROUP_ARITHMETIC_SHARED_SIZE_NEEDED_)
		#error "Not enough shared memory declared"
	#endif
#else
	#define _IRR_GLSL_SCRATCH_SHARED_DEFINED_ irr_glsl_workgroupArithmeticScratchShared
	#define _IRR_GLSL_SCRATCH_SHARED_SIZE_DEFINED_ _IRR_GLSL_WORKGROUP_ARITHMETIC_SHARED_SIZE_NEEDED_
	shared uint _IRR_GLSL_SCRATCH_SHARED_DEFINED_[_IRR_GLSL_WORKGROUP_ARITHMETIC_SHARED_SIZE_NEEDED_];
#endif



#include <irr/builtin/glsl/workgroup/clustered.glsl>



/*
#ifdef GL_KHR_subgroup_arithmetic

#define CONDITIONAL_BARRIER

#else
*/
#define CONDITIONAL_BARRIER barrier();

//#endif



#define DECLARE_OVERLOAD_WITH_BARRIERS(TYPE,FUNC_NAME) TYPE irr_glsl_##FUNC_NAME (in TYPE val) \
{ \
	barrier(); \
	const TYPE retval = irr_glsl_##FUNC_NAME##_noBarriers (val); \
	barrier(); \
	return retval; \
}


// reduction
#define IRR_GLSL_WORKGROUP_REDUCE(CONV,INCLUSIVE_SUBGROUP_OP,VALUE,IDENTITY,INVCONV) IRR_GLSL_WORKGROUP_COMMON_IMPL_HEAD(CONV,INCLUSIVE_SUBGROUP_OP,VALUE,IDENTITY,INVCONV,_IRR_GLSL_WORKGROUP_SIZE_,false); \
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
	IRR_GLSL_WORKGROUP_REDUCE(int,irr_glsl_subgroupInclusiveMin_impl,val,INT_MAX,uint);
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
	IRR_GLSL_WORKGROUP_REDUCE(int,irr_glsl_subgroupInclusiveMax_impl,val,INT_MIN,uint);
}
float irr_glsl_workgroupMax_noBarriers(in float val)
{
	IRR_GLSL_WORKGROUP_REDUCE(uintBitsToFloat,irr_glsl_subgroupInclusiveMax_impl,val,-FLT_INF,floatBitsToUint);
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupMax)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupMax)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupMax)



// scan
#define IRR_GLSL_WORKGROUP_SCAN(EXCLUSIVE,CONV,OP,INCLUSIVE_SUBGROUP_OP,VALUE,IDENTITY,INVCONV) IRR_GLSL_WORKGROUP_COMMON_IMPL_HEAD(CONV,INCLUSIVE_SUBGROUP_OP,VALUE,IDENTITY,INVCONV,_IRR_GLSL_WORKGROUP_SIZE_,true) \
	IRR_GLSL_WORKGROUP_SCAN_IMPL_TAIL(EXCLUSIVE,CONV,INCLUSIVE_SUBGROUP_OP,IDENTITY,INVCONV,OP)



uint irr_glsl_workgroupInclusiveAnd_noBarriers(in uint val)
{
	IRR_GLSL_WORKGROUP_SCAN(false,irr_glsl_identityFunction,irr_glsl_and,irr_glsl_subgroupInclusiveAnd_impl,val,~0u,irr_glsl_identityFunction);
}
int irr_glsl_workgroupInclusiveAnd_noBarriers(in int val)
{
	return int(irr_glsl_workgroupInclusiveAnd_noBarriers(uint(val)));
}
float irr_glsl_workgroupInclusiveAnd_noBarriers(in float val)
{
	return uintBitsToFloat(irr_glsl_workgroupInclusiveAnd_noBarriers(floatBitsToUint(val)));
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupInclusiveAnd)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupInclusiveAnd)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupInclusiveAnd)


uint irr_glsl_workgroupExclusiveAnd_noBarriers(in uint val)
{
	IRR_GLSL_WORKGROUP_SCAN(true,irr_glsl_identityFunction,irr_glsl_and,irr_glsl_subgroupInclusiveAnd_impl,val,~0u,irr_glsl_identityFunction);
}
int irr_glsl_workgroupExclusiveAnd_noBarriers(in int val)
{
	return int(irr_glsl_workgroupExclusiveAnd_noBarriers(uint(val)));
}
float irr_glsl_workgroupExclusiveAnd_noBarriers(in float val)
{
	return uintBitsToFloat(irr_glsl_workgroupExclusiveAnd_noBarriers(floatBitsToUint(val)));
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupExclusiveAnd)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupExclusiveAnd)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupExclusiveAnd)



uint irr_glsl_workgroupInclusiveOr_noBarriers(in uint val)
{
	IRR_GLSL_WORKGROUP_SCAN(false,irr_glsl_identityFunction,irr_glsl_or,irr_glsl_subgroupInclusiveOr_impl,val,0u,irr_glsl_identityFunction);
}
int irr_glsl_workgroupInclusiveOr_noBarriers(in int val)
{
	return int(irr_glsl_workgroupInclusiveOr_noBarriers(uint(val)));
}
float irr_glsl_workgroupInclusiveOr_noBarriers(in float val)
{
	return uintBitsToFloat(irr_glsl_workgroupInclusiveOr_noBarriers(floatBitsToUint(val)));
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupInclusiveOr)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupInclusiveOr)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupInclusiveOr)


uint irr_glsl_workgroupExclusiveOr_noBarriers(in uint val)
{
	IRR_GLSL_WORKGROUP_SCAN(true,irr_glsl_identityFunction,irr_glsl_or,irr_glsl_subgroupInclusiveOr_impl,val,0u,irr_glsl_identityFunction);
}
int irr_glsl_workgroupExclusiveOr_noBarriers(in int val)
{
	return int(irr_glsl_workgroupExclusiveOr_noBarriers(uint(val)));
}
float irr_glsl_workgroupExclusiveOr_noBarriers(in float val)
{
	return uintBitsToFloat(irr_glsl_workgroupExclusiveOr_noBarriers(floatBitsToUint(val)));
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupExclusiveOr)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupExclusiveOr)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupExclusiveOr)



uint irr_glsl_workgroupInclusiveXor_noBarriers(in uint val)
{
	IRR_GLSL_WORKGROUP_SCAN(false,irr_glsl_identityFunction,irr_glsl_xor,irr_glsl_subgroupInclusiveXor_impl,val,0u,irr_glsl_identityFunction);
}
int irr_glsl_workgroupInclusiveXor_noBarriers(in int val)
{
	return int(irr_glsl_workgroupInclusiveXor_noBarriers(uint(val)));
}
float irr_glsl_workgroupInclusiveXor_noBarriers(in float val)
{
	return uintBitsToFloat(irr_glsl_workgroupInclusiveXor_noBarriers(floatBitsToUint(val)));
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupInclusiveXor)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupInclusiveXor)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupInclusiveXor)


uint irr_glsl_workgroupExclusiveXor_noBarriers(in uint val)
{
	IRR_GLSL_WORKGROUP_SCAN(true,irr_glsl_identityFunction,irr_glsl_xor,irr_glsl_subgroupInclusiveXor_impl,val,0u,irr_glsl_identityFunction);
}
int irr_glsl_workgroupExclusiveXor_noBarriers(in int val)
{
	return int(irr_glsl_workgroupExclusiveXor_noBarriers(uint(val)));
}
float irr_glsl_workgroupExclusiveXor_noBarriers(in float val)
{
	return uintBitsToFloat(irr_glsl_workgroupExclusiveXor_noBarriers(floatBitsToUint(val)));
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupExclusiveXor)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupExclusiveXor)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupExclusiveXor)



uint irr_glsl_workgroupInclusiveAdd_noBarriers(in uint val)
{
	IRR_GLSL_WORKGROUP_SCAN(false,irr_glsl_identityFunction,irr_glsl_add,irr_glsl_subgroupInclusiveAdd_impl,val,0u,irr_glsl_identityFunction);
}
int irr_glsl_workgroupInclusiveAdd_noBarriers(in int val)
{
	return int(irr_glsl_workgroupInclusiveAdd_noBarriers(uint(val)));
}
float irr_glsl_workgroupInclusiveAdd_noBarriers(in float val)
{
	IRR_GLSL_WORKGROUP_SCAN(false,uintBitsToFloat,irr_glsl_add,irr_glsl_subgroupInclusiveAdd_impl,val,0.0,floatBitsToUint);
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupInclusiveAdd)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupInclusiveAdd)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupInclusiveAdd)


uint irr_glsl_workgroupExclusiveAdd_noBarriers(in uint val)
{
	IRR_GLSL_WORKGROUP_SCAN(true,irr_glsl_identityFunction,irr_glsl_add,irr_glsl_subgroupInclusiveAdd_impl,val,0u,irr_glsl_identityFunction);
}
int irr_glsl_workgroupExclusiveAdd_noBarriers(in int val)
{
	return int(irr_glsl_workgroupExclusiveAdd_noBarriers(uint(val)));
}
float irr_glsl_workgroupExclusiveAdd_noBarriers(in float val)
{
	IRR_GLSL_WORKGROUP_SCAN(true,uintBitsToFloat,irr_glsl_add,irr_glsl_subgroupInclusiveAdd_impl,val,0.0,floatBitsToUint);
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupExclusiveAdd)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupExclusiveAdd)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupExclusiveAdd)



uint irr_glsl_workgroupInclusiveMul_noBarriers(in uint val)
{
	IRR_GLSL_WORKGROUP_SCAN(false,irr_glsl_identityFunction,irr_glsl_mul,irr_glsl_subgroupInclusiveMul_impl,val,1u,irr_glsl_identityFunction);
}
int irr_glsl_workgroupInclusiveMul_noBarriers(in int val)
{
	return int(irr_glsl_workgroupInclusiveMul_noBarriers(uint(val)));
}
float irr_glsl_workgroupInclusiveMul_noBarriers(in float val)
{
	IRR_GLSL_WORKGROUP_SCAN(false,uintBitsToFloat,irr_glsl_mul,irr_glsl_subgroupInclusiveMul_impl,val,1.0,floatBitsToUint);
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupInclusiveMul)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupInclusiveMul)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupInclusiveMul)


uint irr_glsl_workgroupExclusiveMul_noBarriers(in uint val)
{
	IRR_GLSL_WORKGROUP_SCAN(true,irr_glsl_identityFunction,irr_glsl_mul,irr_glsl_subgroupInclusiveMul_impl,val,1u,irr_glsl_identityFunction);
}
int irr_glsl_workgroupExclusiveMul_noBarriers(in int val)
{
	return int(irr_glsl_workgroupExclusiveMul_noBarriers(uint(val)));
}
float irr_glsl_workgroupExclusiveMul_noBarriers(in float val)
{
	IRR_GLSL_WORKGROUP_SCAN(true,uintBitsToFloat,irr_glsl_mul,irr_glsl_subgroupInclusiveMul_impl,val,1.0,floatBitsToUint);
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupExclusiveMul)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupExclusiveMul)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupExclusiveMul)



uint irr_glsl_workgroupInclusiveMin_noBarriers(in uint val)
{
	IRR_GLSL_WORKGROUP_SCAN(false,irr_glsl_identityFunction,min,irr_glsl_subgroupInclusiveMin_impl,val,UINT_MAX,irr_glsl_identityFunction);
}
int irr_glsl_workgroupInclusiveMin_noBarriers(in int val)
{
	IRR_GLSL_WORKGROUP_SCAN(false,int,min,irr_glsl_subgroupInclusiveMin_impl,val,INT_MAX,uint);
}
float irr_glsl_workgroupInclusiveMin_noBarriers(in float val)
{
	IRR_GLSL_WORKGROUP_SCAN(false,uintBitsToFloat,min,irr_glsl_subgroupInclusiveMin_impl,val,FLT_INF,floatBitsToUint);
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupInclusiveMin)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupInclusiveMin)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupInclusiveMin)


uint irr_glsl_workgroupExclusiveMin_noBarriers(in uint val)
{
	IRR_GLSL_WORKGROUP_SCAN(true,irr_glsl_identityFunction,min,irr_glsl_subgroupInclusiveMin_impl,val,UINT_MAX,irr_glsl_identityFunction);
}
int irr_glsl_workgroupExclusiveMin_noBarriers(in int val)
{
	IRR_GLSL_WORKGROUP_SCAN(true,int,min,irr_glsl_subgroupExclusiveMin_impl,val,INT_MAX,uint);
}
float irr_glsl_workgroupExclusiveMin_noBarriers(in float val)
{
	IRR_GLSL_WORKGROUP_SCAN(true,uintBitsToFloat,min,irr_glsl_subgroupInclusiveMin_impl,val,FLT_INF,floatBitsToUint);
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupExclusiveMin)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupExclusiveMin)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupExclusiveMin)



uint irr_glsl_workgroupInclusiveMax_noBarriers(in uint val)
{
	IRR_GLSL_WORKGROUP_SCAN(false,irr_glsl_identityFunction,max,irr_glsl_subgroupInclusiveMax_impl,val,UINT_MIN,irr_glsl_identityFunction);
}
int irr_glsl_workgroupInclusiveMax_noBarriers(in int val)
{
	IRR_GLSL_WORKGROUP_SCAN(false,int,max,irr_glsl_subgroupInclusiveMax_impl,val,INT_MIN,uint);
}
float irr_glsl_workgroupInclusiveMax_noBarriers(in float val)
{
	IRR_GLSL_WORKGROUP_SCAN(false,uintBitsToFloat,max,irr_glsl_subgroupInclusiveMax_impl,val,-FLT_INF,floatBitsToUint);
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupInclusiveMax)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupInclusiveMax)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupInclusiveMax)


uint irr_glsl_workgroupExclusiveMax_noBarriers(in uint val)
{
	IRR_GLSL_WORKGROUP_SCAN(true,irr_glsl_identityFunction,max,irr_glsl_subgroupInclusiveMax_impl,val,UINT_MIN,irr_glsl_identityFunction);
}
int irr_glsl_workgroupExclusiveMax_noBarriers(in int val)
{
	IRR_GLSL_WORKGROUP_SCAN(true,int,max,irr_glsl_subgroupInclusiveMax_impl,val,INT_MIN,uint);
}
float irr_glsl_workgroupExclusiveMax_noBarriers(in float val)
{
	IRR_GLSL_WORKGROUP_SCAN(true,uintBitsToFloat,max,irr_glsl_subgroupInclusiveMax_impl,val,-FLT_INF,floatBitsToUint);
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupExclusiveMax)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupExclusiveMax)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupExclusiveMax)



#undef DECLARE_OVERLOAD_WITH_BARRIERS

#undef CONDITIONAL_BARRIER

#endif
