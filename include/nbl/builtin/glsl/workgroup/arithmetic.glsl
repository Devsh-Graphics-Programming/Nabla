#ifndef _NBL_BUILTIN_GLSL_WORKGROUP_ARITHMETIC_INCLUDED_
#define _NBL_BUILTIN_GLSL_WORKGROUP_ARITHMETIC_INCLUDED_


#include <nbl/builtin/glsl/workgroup/shared_arithmetic.glsl>



#ifdef _NBL_GLSL_SCRATCH_SHARED_DEFINED_
	#if NBL_GLSL_EVAL(_NBL_GLSL_SCRATCH_SHARED_SIZE_DEFINED_)<NBL_GLSL_EVAL(_NBL_GLSL_WORKGROUP_ARITHMETIC_SHARED_SIZE_NEEDED_)
		#error "Not enough shared memory declared"
	#endif
#else
	#define _NBL_GLSL_SCRATCH_SHARED_DEFINED_ nbl_glsl_workgroupArithmeticScratchShared
	#define _NBL_GLSL_SCRATCH_SHARED_SIZE_DEFINED_ _NBL_GLSL_WORKGROUP_ARITHMETIC_SHARED_SIZE_NEEDED_
	shared uint _NBL_GLSL_SCRATCH_SHARED_DEFINED_[_NBL_GLSL_WORKGROUP_ARITHMETIC_SHARED_SIZE_NEEDED_];
#endif



#include <nbl/builtin/glsl/workgroup/clustered.glsl>



/*
#ifdef GL_KHR_subgroup_arithmetic

#define CONDITIONAL_BARRIER

#else
*/
#define CONDITIONAL_BARRIER barrier();

//#endif



#define DECLARE_OVERLOAD_WITH_BARRIERS(TYPE,FUNC_NAME) TYPE nbl_glsl_##FUNC_NAME (in TYPE val) \
{ \
	barrier(); \
	const TYPE retval = nbl_glsl_##FUNC_NAME##_noBarriers (val); \
	barrier(); \
	return retval; \
}


// reduction
#define NBL_GLSL_WORKGROUP_REDUCE(CONV,INCLUSIVE_SUBGROUP_OP,VALUE,IDENTITY,INVCONV) NBL_GLSL_WORKGROUP_COMMON_IMPL_HEAD(CONV,INCLUSIVE_SUBGROUP_OP,VALUE,IDENTITY,INVCONV,_NBL_GLSL_WORKGROUP_SIZE_,false); \
	barrier(); \
	return CONV(nbl_glsl_workgroupBroadcast_noBarriers(scan,lastInvocationInLevel))


uint nbl_glsl_workgroupAnd_noBarriers(in uint val)
{
	NBL_GLSL_WORKGROUP_REDUCE(nbl_glsl_identityFunction,nbl_glsl_subgroupInclusiveAnd_impl,val,0xffFFffFFu,nbl_glsl_identityFunction);
}
int nbl_glsl_workgroupAnd_noBarriers(in int val)
{
	return int(nbl_glsl_workgroupAnd_noBarriers(uint(val)));
}
float nbl_glsl_workgroupAnd_noBarriers(in float val)
{
	return uintBitsToFloat(nbl_glsl_workgroupAnd_noBarriers(floatBitsToUint(val)));
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupAnd)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupAnd)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupAnd)


uint nbl_glsl_workgroupOr_noBarriers(in uint val)
{
	NBL_GLSL_WORKGROUP_REDUCE(nbl_glsl_identityFunction,nbl_glsl_subgroupInclusiveOr_impl,val,0u,nbl_glsl_identityFunction);
}
int nbl_glsl_workgroupOr_noBarriers(in int val)
{
	return int(nbl_glsl_workgroupOr_noBarriers(uint(val)));
}
float nbl_glsl_workgroupOr_noBarriers(in float val)
{
	NBL_GLSL_WORKGROUP_REDUCE(uintBitsToFloat,nbl_glsl_subgroupInclusiveOr_impl,val,0.0,floatBitsToUint);
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupOr)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupOr)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupOr)


uint nbl_glsl_workgroupXor_noBarriers(in uint val)
{
	NBL_GLSL_WORKGROUP_REDUCE(nbl_glsl_identityFunction,nbl_glsl_subgroupInclusiveXor_impl,val,0u,nbl_glsl_identityFunction);
}
int nbl_glsl_workgroupXor_noBarriers(in int val)
{
	return int(nbl_glsl_workgroupXor_noBarriers(uint(val)));
}
float nbl_glsl_workgroupXor_noBarriers(in float val)
{
	NBL_GLSL_WORKGROUP_REDUCE(uintBitsToFloat,nbl_glsl_subgroupInclusiveXor_impl,val,0.0,floatBitsToUint);
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupXor)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupXor)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupXor)


uint nbl_glsl_workgroupAdd_noBarriers(in uint val)
{
	NBL_GLSL_WORKGROUP_REDUCE(nbl_glsl_identityFunction,nbl_glsl_subgroupInclusiveAdd_impl,val,0u,nbl_glsl_identityFunction);
}
int nbl_glsl_workgroupAdd_noBarriers(in int val)
{
	return int(nbl_glsl_workgroupAdd_noBarriers(uint(val)));
}
float nbl_glsl_workgroupAdd_noBarriers(in float val)
{
	NBL_GLSL_WORKGROUP_REDUCE(uintBitsToFloat,nbl_glsl_subgroupInclusiveAdd_impl,val,0.0,floatBitsToUint);
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupAdd)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupAdd)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupAdd)


uint nbl_glsl_workgroupMul_noBarriers(in uint val)
{
	NBL_GLSL_WORKGROUP_REDUCE(nbl_glsl_identityFunction,nbl_glsl_subgroupInclusiveMul_impl,val,1u,nbl_glsl_identityFunction);
}
int nbl_glsl_workgroupMul_noBarriers(in int val)
{
	return int(nbl_glsl_workgroupMul_noBarriers(uint(val)));
}
float nbl_glsl_workgroupMul_noBarriers(in float val)
{
	NBL_GLSL_WORKGROUP_REDUCE(uintBitsToFloat,nbl_glsl_subgroupInclusiveMul_impl,val,1.0,floatBitsToUint);
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupMul)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupMul)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupMul)


uint nbl_glsl_workgroupMin_noBarriers(in uint val)
{
	NBL_GLSL_WORKGROUP_REDUCE(nbl_glsl_identityFunction,nbl_glsl_subgroupInclusiveMin_impl,val,nbl_glsl_UINT_MAX,nbl_glsl_identityFunction);
}
int nbl_glsl_workgroupMin_noBarriers(in int val)
{
	NBL_GLSL_WORKGROUP_REDUCE(int,nbl_glsl_subgroupInclusiveMin_impl,val,nbl_glsl_INT_MAX,uint);
}
float nbl_glsl_workgroupMin_noBarriers(in float val)
{
	NBL_GLSL_WORKGROUP_REDUCE(uintBitsToFloat,nbl_glsl_subgroupInclusiveMin_impl,val,nbl_glsl_FLT_INF,floatBitsToUint);
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupMin)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupMin)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupMin)


uint nbl_glsl_workgroupMax_noBarriers(in uint val)
{
	NBL_GLSL_WORKGROUP_REDUCE(nbl_glsl_identityFunction,nbl_glsl_subgroupInclusiveMax_impl,val,nbl_glsl_UINT_MIN,nbl_glsl_identityFunction);
}
int nbl_glsl_workgroupMax_noBarriers(in int val)
{
	NBL_GLSL_WORKGROUP_REDUCE(int,nbl_glsl_subgroupInclusiveMax_impl,val,nbl_glsl_INT_MIN,uint);
}
float nbl_glsl_workgroupMax_noBarriers(in float val)
{
	NBL_GLSL_WORKGROUP_REDUCE(uintBitsToFloat,nbl_glsl_subgroupInclusiveMax_impl,val,-nbl_glsl_FLT_INF,floatBitsToUint);
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupMax)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupMax)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupMax)



// scan
#define NBL_GLSL_WORKGROUP_SCAN(EXCLUSIVE,CONV,OP,INCLUSIVE_SUBGROUP_OP,VALUE,IDENTITY,INVCONV) NBL_GLSL_WORKGROUP_COMMON_IMPL_HEAD(CONV,INCLUSIVE_SUBGROUP_OP,VALUE,IDENTITY,INVCONV,_NBL_GLSL_WORKGROUP_SIZE_,true) \
	NBL_GLSL_WORKGROUP_SCAN_IMPL_TAIL(EXCLUSIVE,CONV,INCLUSIVE_SUBGROUP_OP,IDENTITY,INVCONV,OP)



uint nbl_glsl_workgroupInclusiveAnd_noBarriers(in uint val)
{
	NBL_GLSL_WORKGROUP_SCAN(false,nbl_glsl_identityFunction,nbl_glsl_and,nbl_glsl_subgroupInclusiveAnd_impl,val,0xffFFffFFu,nbl_glsl_identityFunction);
}
int nbl_glsl_workgroupInclusiveAnd_noBarriers(in int val)
{
	return int(nbl_glsl_workgroupInclusiveAnd_noBarriers(uint(val)));
}
float nbl_glsl_workgroupInclusiveAnd_noBarriers(in float val)
{
	return uintBitsToFloat(nbl_glsl_workgroupInclusiveAnd_noBarriers(floatBitsToUint(val)));
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupInclusiveAnd)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupInclusiveAnd)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupInclusiveAnd)


uint nbl_glsl_workgroupExclusiveAnd_noBarriers(in uint val)
{
	NBL_GLSL_WORKGROUP_SCAN(true,nbl_glsl_identityFunction,nbl_glsl_and,nbl_glsl_subgroupInclusiveAnd_impl,val,0xffFFffFFu,nbl_glsl_identityFunction);
}
int nbl_glsl_workgroupExclusiveAnd_noBarriers(in int val)
{
	return int(nbl_glsl_workgroupExclusiveAnd_noBarriers(uint(val)));
}
float nbl_glsl_workgroupExclusiveAnd_noBarriers(in float val)
{
	return uintBitsToFloat(nbl_glsl_workgroupExclusiveAnd_noBarriers(floatBitsToUint(val)));
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupExclusiveAnd)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupExclusiveAnd)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupExclusiveAnd)



uint nbl_glsl_workgroupInclusiveOr_noBarriers(in uint val)
{
	NBL_GLSL_WORKGROUP_SCAN(false,nbl_glsl_identityFunction,nbl_glsl_or,nbl_glsl_subgroupInclusiveOr_impl,val,0u,nbl_glsl_identityFunction);
}
int nbl_glsl_workgroupInclusiveOr_noBarriers(in int val)
{
	return int(nbl_glsl_workgroupInclusiveOr_noBarriers(uint(val)));
}
float nbl_glsl_workgroupInclusiveOr_noBarriers(in float val)
{
	return uintBitsToFloat(nbl_glsl_workgroupInclusiveOr_noBarriers(floatBitsToUint(val)));
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupInclusiveOr)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupInclusiveOr)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupInclusiveOr)


uint nbl_glsl_workgroupExclusiveOr_noBarriers(in uint val)
{
	NBL_GLSL_WORKGROUP_SCAN(true,nbl_glsl_identityFunction,nbl_glsl_or,nbl_glsl_subgroupInclusiveOr_impl,val,0u,nbl_glsl_identityFunction);
}
int nbl_glsl_workgroupExclusiveOr_noBarriers(in int val)
{
	return int(nbl_glsl_workgroupExclusiveOr_noBarriers(uint(val)));
}
float nbl_glsl_workgroupExclusiveOr_noBarriers(in float val)
{
	return uintBitsToFloat(nbl_glsl_workgroupExclusiveOr_noBarriers(floatBitsToUint(val)));
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupExclusiveOr)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupExclusiveOr)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupExclusiveOr)



uint nbl_glsl_workgroupInclusiveXor_noBarriers(in uint val)
{
	NBL_GLSL_WORKGROUP_SCAN(false,nbl_glsl_identityFunction,nbl_glsl_xor,nbl_glsl_subgroupInclusiveXor_impl,val,0u,nbl_glsl_identityFunction);
}
int nbl_glsl_workgroupInclusiveXor_noBarriers(in int val)
{
	return int(nbl_glsl_workgroupInclusiveXor_noBarriers(uint(val)));
}
float nbl_glsl_workgroupInclusiveXor_noBarriers(in float val)
{
	return uintBitsToFloat(nbl_glsl_workgroupInclusiveXor_noBarriers(floatBitsToUint(val)));
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupInclusiveXor)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupInclusiveXor)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupInclusiveXor)


uint nbl_glsl_workgroupExclusiveXor_noBarriers(in uint val)
{
	NBL_GLSL_WORKGROUP_SCAN(true,nbl_glsl_identityFunction,nbl_glsl_xor,nbl_glsl_subgroupInclusiveXor_impl,val,0u,nbl_glsl_identityFunction);
}
int nbl_glsl_workgroupExclusiveXor_noBarriers(in int val)
{
	return int(nbl_glsl_workgroupExclusiveXor_noBarriers(uint(val)));
}
float nbl_glsl_workgroupExclusiveXor_noBarriers(in float val)
{
	return uintBitsToFloat(nbl_glsl_workgroupExclusiveXor_noBarriers(floatBitsToUint(val)));
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupExclusiveXor)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupExclusiveXor)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupExclusiveXor)



uint nbl_glsl_workgroupInclusiveAdd_noBarriers(in uint val)
{
	NBL_GLSL_WORKGROUP_SCAN(false,nbl_glsl_identityFunction,nbl_glsl_add,nbl_glsl_subgroupInclusiveAdd_impl,val,0u,nbl_glsl_identityFunction);
}
int nbl_glsl_workgroupInclusiveAdd_noBarriers(in int val)
{
	return int(nbl_glsl_workgroupInclusiveAdd_noBarriers(uint(val)));
}
float nbl_glsl_workgroupInclusiveAdd_noBarriers(in float val)
{
	NBL_GLSL_WORKGROUP_SCAN(false,uintBitsToFloat,nbl_glsl_add,nbl_glsl_subgroupInclusiveAdd_impl,val,0.0,floatBitsToUint);
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupInclusiveAdd)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupInclusiveAdd)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupInclusiveAdd)


uint nbl_glsl_workgroupExclusiveAdd_noBarriers(in uint val)
{
	NBL_GLSL_WORKGROUP_SCAN(true,nbl_glsl_identityFunction,nbl_glsl_add,nbl_glsl_subgroupInclusiveAdd_impl,val,0u,nbl_glsl_identityFunction);
}
int nbl_glsl_workgroupExclusiveAdd_noBarriers(in int val)
{
	return int(nbl_glsl_workgroupExclusiveAdd_noBarriers(uint(val)));
}
float nbl_glsl_workgroupExclusiveAdd_noBarriers(in float val)
{
	NBL_GLSL_WORKGROUP_SCAN(true,uintBitsToFloat,nbl_glsl_add,nbl_glsl_subgroupInclusiveAdd_impl,val,0.0,floatBitsToUint);
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupExclusiveAdd)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupExclusiveAdd)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupExclusiveAdd)



uint nbl_glsl_workgroupInclusiveMul_noBarriers(in uint val)
{
	NBL_GLSL_WORKGROUP_SCAN(false,nbl_glsl_identityFunction,nbl_glsl_mul,nbl_glsl_subgroupInclusiveMul_impl,val,1u,nbl_glsl_identityFunction);
}
int nbl_glsl_workgroupInclusiveMul_noBarriers(in int val)
{
	return int(nbl_glsl_workgroupInclusiveMul_noBarriers(uint(val)));
}
float nbl_glsl_workgroupInclusiveMul_noBarriers(in float val)
{
	NBL_GLSL_WORKGROUP_SCAN(false,uintBitsToFloat,nbl_glsl_mul,nbl_glsl_subgroupInclusiveMul_impl,val,1.0,floatBitsToUint);
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupInclusiveMul)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupInclusiveMul)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupInclusiveMul)


uint nbl_glsl_workgroupExclusiveMul_noBarriers(in uint val)
{
	NBL_GLSL_WORKGROUP_SCAN(true,nbl_glsl_identityFunction,nbl_glsl_mul,nbl_glsl_subgroupInclusiveMul_impl,val,1u,nbl_glsl_identityFunction);
}
int nbl_glsl_workgroupExclusiveMul_noBarriers(in int val)
{
	return int(nbl_glsl_workgroupExclusiveMul_noBarriers(uint(val)));
}
float nbl_glsl_workgroupExclusiveMul_noBarriers(in float val)
{
	NBL_GLSL_WORKGROUP_SCAN(true,uintBitsToFloat,nbl_glsl_mul,nbl_glsl_subgroupInclusiveMul_impl,val,1.0,floatBitsToUint);
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupExclusiveMul)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupExclusiveMul)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupExclusiveMul)



uint nbl_glsl_workgroupInclusiveMin_noBarriers(in uint val)
{
	NBL_GLSL_WORKGROUP_SCAN(false,nbl_glsl_identityFunction,min,nbl_glsl_subgroupInclusiveMin_impl,val,nbl_glsl_UINT_MAX,nbl_glsl_identityFunction);
}
int nbl_glsl_workgroupInclusiveMin_noBarriers(in int val)
{
	NBL_GLSL_WORKGROUP_SCAN(false,int,min,nbl_glsl_subgroupInclusiveMin_impl,val,nbl_glsl_INT_MAX,uint);
}
float nbl_glsl_workgroupInclusiveMin_noBarriers(in float val)
{
	NBL_GLSL_WORKGROUP_SCAN(false,uintBitsToFloat,min,nbl_glsl_subgroupInclusiveMin_impl,val,nbl_glsl_FLT_INF,floatBitsToUint);
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupInclusiveMin)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupInclusiveMin)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupInclusiveMin)


uint nbl_glsl_workgroupExclusiveMin_noBarriers(in uint val)
{
	NBL_GLSL_WORKGROUP_SCAN(true,nbl_glsl_identityFunction,min,nbl_glsl_subgroupInclusiveMin_impl,val,nbl_glsl_UINT_MAX,nbl_glsl_identityFunction);
}
int nbl_glsl_workgroupExclusiveMin_noBarriers(in int val)
{
	NBL_GLSL_WORKGROUP_SCAN(true,int,min,nbl_glsl_subgroupInclusiveMin_impl,val,nbl_glsl_INT_MAX,uint);
}
float nbl_glsl_workgroupExclusiveMin_noBarriers(in float val)
{
	NBL_GLSL_WORKGROUP_SCAN(true,uintBitsToFloat,min,nbl_glsl_subgroupInclusiveMin_impl,val,nbl_glsl_FLT_INF,floatBitsToUint);
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupExclusiveMin)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupExclusiveMin)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupExclusiveMin)



uint nbl_glsl_workgroupInclusiveMax_noBarriers(in uint val)
{
	NBL_GLSL_WORKGROUP_SCAN(false,nbl_glsl_identityFunction,max,nbl_glsl_subgroupInclusiveMax_impl,val,nbl_glsl_UINT_MIN,nbl_glsl_identityFunction);
}
int nbl_glsl_workgroupInclusiveMax_noBarriers(in int val)
{
	NBL_GLSL_WORKGROUP_SCAN(false,int,max,nbl_glsl_subgroupInclusiveMax_impl,val,nbl_glsl_INT_MIN,uint);
}
float nbl_glsl_workgroupInclusiveMax_noBarriers(in float val)
{
	NBL_GLSL_WORKGROUP_SCAN(false,uintBitsToFloat,max,nbl_glsl_subgroupInclusiveMax_impl,val,-nbl_glsl_FLT_INF,floatBitsToUint);
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupInclusiveMax)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupInclusiveMax)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupInclusiveMax)


uint nbl_glsl_workgroupExclusiveMax_noBarriers(in uint val)
{
	NBL_GLSL_WORKGROUP_SCAN(true,nbl_glsl_identityFunction,max,nbl_glsl_subgroupInclusiveMax_impl,val,nbl_glsl_UINT_MIN,nbl_glsl_identityFunction);
}
int nbl_glsl_workgroupExclusiveMax_noBarriers(in int val)
{
	NBL_GLSL_WORKGROUP_SCAN(true,int,max,nbl_glsl_subgroupInclusiveMax_impl,val,nbl_glsl_INT_MIN,uint);
}
float nbl_glsl_workgroupExclusiveMax_noBarriers(in float val)
{
	NBL_GLSL_WORKGROUP_SCAN(true,uintBitsToFloat,max,nbl_glsl_subgroupInclusiveMax_impl,val,-nbl_glsl_FLT_INF,floatBitsToUint);
}

DECLARE_OVERLOAD_WITH_BARRIERS(uint,workgroupExclusiveMax)
DECLARE_OVERLOAD_WITH_BARRIERS(int,workgroupExclusiveMax)
DECLARE_OVERLOAD_WITH_BARRIERS(float,workgroupExclusiveMax)



#undef DECLARE_OVERLOAD_WITH_BARRIERS

#undef CONDITIONAL_BARRIER

#endif
