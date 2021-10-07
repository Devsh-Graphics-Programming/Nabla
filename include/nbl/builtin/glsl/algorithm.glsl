#ifndef _NBL_BUILTIN_GLSL_ALGORITHM_INCLUDED_
#define _NBL_BUILTIN_GLSL_ALGORITHM_INCLUDED_

#include <nbl/builtin/glsl/macros.glsl>


#define NBL_GLSL_DECLARE_LOWER_BOUND_COMP(ARRAY_NAME,TYPE,COMP) NBL_GLSL_CONCATENATE4(uint lower_bound_,ARRAY_NAME,_,NBL_GLSL_LESS)(uint begin, in uint end, in TYPE value);

#define NBL_GLSL_DECLARE_UPPER_BOUND_COMP(ARRAY_NAME,TYPE,COMP) NBL_GLSL_CONCATENATE4(uint lower_bound_,ARRAY_NAME,_,NBL_GLSL_LESS)(uint begin, in uint end, in TYPE value);


#define NBL_GLSL_DECLARE_LOWER_BOUND(ARRAY_NAME,TYPE) NBL_GLSL_DECLARE_LOWER_BOUND_COMP(ARRAY_NAME,TYPE,NBL_GLSL_LESS) \
NBL_GLSL_CONCATENATE2(uint lower_bound_,ARRAY_NAME)(uint begin, in uint end, in TYPE value) {return NBL_GLSL_CONCATENATE4(lower_bound_,ARRAY_NAME,_,NBL_GLSL_LESS)(begin,end,value);}

#define NBL_GLSL_DECLARE_UPPER_BOUND(ARRAY_NAME,TYPE) NBL_GLSL_DECLARE_UPPER_BOUND_COMP(ARRAY_NAME,TYPE,NBL_GLSL_LESS) \
NBL_GLSL_CONCATENATE2(uint upper_bound_,ARRAY_NAME)(uint begin, in uint end, in TYPE value) {return NBL_GLSL_CONCATENATE4(upper_bound_,ARRAY_NAME,_,NBL_GLSL_LESS)(begin,end,value);}


#define NBL_GLSL_DEFINE_BOUND_COMP_IMPL(FUNC_NAME,ARRAY_NAME,TYPE,COMP) NBL_GLSL_CONCATENATE4(uint FUNC_NAME,ARRAY_NAME,_,COMP)(uint begin, in uint end, in TYPE value) \
{ \
	uint len = end-begin; \
	if (NBL_GLSL_IS_NOT_POT(len)) \
	{ \
		const uint newLen = 0x1u<<findMSB(len); \
		const uint diff = len-newLen; \
		begin = COMP(NBL_GLSL_EVAL(ARRAY_NAME)[newLen],value) ? diff:0u; \
		len = newLen; \
	} \
	while (len!=0u) \
	{


// could unroll 3 or more times
#define NBL_GLSL_DEFINE_LOWER_BOUND_COMP(ARRAY_NAME,TYPE,COMP) NBL_GLSL_DEFINE_BOUND_COMP_IMPL(lower_bound_,ARRAY_NAME,TYPE,COMP) \
		begin += COMP(NBL_GLSL_EVAL(ARRAY_NAME)[begin+(len>>=1u)],value) ? len:0u; \
		begin += COMP(NBL_GLSL_EVAL(ARRAY_NAME)[begin+(len>>=1u)],value) ? len:0u; \
	} \
	return begin+(COMP(NBL_GLSL_EVAL(ARRAY_NAME)[begin],value) ? 1u:0u); \
}

#define NBL_GLSL_DEFINE_LOWER_BOUND(ARRAY_NAME,TYPE) NBL_GLSL_DEFINE_LOWER_BOUND_COMP(ARRAY_NAME,TYPE,NBL_GLSL_LESS)

#define NBL_GLSL_DEFINE_UPPER_BOUND_COMP(ARRAY_NAME,TYPE,COMP) NBL_GLSL_DEFINE_BOUND_COMP_IMPL(upper_bound_,ARRAY_NAME,TYPE,COMP) \
		begin += COMP(value,NBL_GLSL_EVAL(ARRAY_NAME)[begin+(len>>=1u)]) ? 0u:len; \
		begin += COMP(value,NBL_GLSL_EVAL(ARRAY_NAME)[begin+(len>>=1u)]) ? 0u:len; \
	} \
	return begin+(COMP(value,NBL_GLSL_EVAL(ARRAY_NAME)[begin]) ? 0u:1u); \
}

#define NBL_GLSL_DEFINE_UPPER_BOUND(ARRAY_NAME,TYPE) NBL_GLSL_DEFINE_UPPER_BOUND_COMP(ARRAY_NAME,TYPE,NBL_GLSL_LESS)


/**

TODOs:
Higher Priority:
- https://moderngpu.github.io/sortedsearch.html
We need to make a `nbl_glsl_workgroupSort()` function as a utility to facilitate it.
- https://moderngpu.github.io/loadbalance.html

Low Priority:
- https://moderngpu.github.io/bulkinsert.html
- https://moderngpu.github.io/merge.html 

**/

#endif
