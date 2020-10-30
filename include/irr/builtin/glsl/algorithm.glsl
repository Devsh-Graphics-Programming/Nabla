#ifndef _IRR_BUILTIN_GLSL_ALGORITHM_INCLUDED_
#define _IRR_BUILTIN_GLSL_ALGORITHM_INCLUDED_

#include "irr/builtin/glsl/macros.h"


#define IRR_GLSL_DECLARE_LOWER_BOUND_COMP(ARRAY_NAME,TYPE,COMP) IRR_GLSL_CONCATENATE4(uint lower_bound_,ARRAY_NAME,_,IRR_GLSL_LESS)(uint begin, in uint end, in TYPE value);

#define IRR_GLSL_DECLARE_UPPER_BOUND_COMP(ARRAY_NAME,TYPE,COMP) IRR_GLSL_CONCATENATE4(uint lower_bound_,ARRAY_NAME,_,IRR_GLSL_LESS)(uint begin, in uint end, in TYPE value);


#define IRR_GLSL_DECLARE_LOWER_BOUND(ARRAY_NAME,TYPE) IRR_GLSL_DECLARE_LOWER_BOUND_COMP(ARRAY_NAME,TYPE,IRR_GLSL_LESS) \
IRR_GLSL_CONCATENATE2(uint lower_bound_,ARRAY_NAME)(uint begin, in uint end, in TYPE value) {return IRR_GLSL_CONCATENATE4(lower_bound_,ARRAY_NAME,_,IRR_GLSL_LESS)(begin,end,value);}

#define IRR_GLSL_DECLARE_UPPER_BOUND(ARRAY_NAME,TYPE) IRR_GLSL_DECLARE_UPPER_BOUND_COMP(ARRAY_NAME,TYPE,IRR_GLSL_LESS) \
IRR_GLSL_CONCATENATE2(uint upper_bound_,ARRAY_NAME)(uint begin, in uint end, in TYPE value) {return IRR_GLSL_CONCATENATE4(upper_bound_,ARRAY_NAME,_,IRR_GLSL_LESS)(begin,end,value);}


#define IRR_GLSL_DEFINE_BOUND_COMP_IMPL(FUNC_NAME,ARRAY_NAME,TYPE,COMP) IRR_GLSL_CONCATENATE4(uint FUNC_NAME,ARRAY_NAME,_,COMP)(uint begin, in uint end, in TYPE value) \
{ \
	const uint len = end-begin; \
	if (IRR_GLSL_IS_NOT_POT(len)) \
	{ \
		const uint newLen = 0x1u<<findMSB(len); \
		const uint diff = len-newLen; \
		begin = COMP(IRR_GLSL_EVAL(ARRAY_NAME)[newLen],value) ? diff:0u; \
		len = newLen; \
	} \
	while (len) \
	{


// could unroll 3 or more times
#define IRR_GLSL_DEFINE_LOWER_BOUND_COMP(ARRAY_NAME,TYPE,COMP) IRR_GLSL_DEFINE_BOUND_COMP_IMPL(lower_bound_,ARRAY_NAME,TYPE,COMP) \
		begin += COMP(IRR_GLSL_EVAL(ARRAY_NAME)[begin+(len>>=1u)],value) ? len:0u; \
		begin += COMP(IRR_GLSL_EVAL(ARRAY_NAME)[begin+(len>>=1u)],value) ? len:0u; \
	} \
	return begin+(COMP(IRR_GLSL_EVAL(ARRAY_NAME)[begin],value) ? 1u:0u); \
}

#define IRR_GLSL_DEFINE_LOWER_BOUND(ARRAY_NAME,TYPE) IRR_GLSL_DEFINE_LOWER_BOUND_COMP(ARRAY_NAME,TYPE,IRR_GLSL_LESS)

#define IRR_GLSL_DEFINE_UPPER_BOUND_COMP(ARRAY_NAME,TYPE,COMP) IRR_GLSL_DEFINE_BOUND_COMP_IMPL(upper_bound_,ARRAY_NAME,TYPE,COMP) \
		begin += COMP(value,IRR_GLSL_EVAL(ARRAY_NAME)[begin+(len>>=1u)]) ? 0u:len; \
		begin += COMP(value,IRR_GLSL_EVAL(ARRAY_NAME)[begin+(len>>=1u)]) ? 0u:len; \
	} \
	return begin+(COMP(value,IRR_GLSL_EVAL(ARRAY_NAME)[begin]) ? 0u:1u); \
}

#define IRR_GLSL_DEFINE_UPPER_BOUND(ARRAY_NAME,TYPE) IRR_GLSL_DEFINE_UPPER_BOUND_COMP(ARRAY_NAME,TYPE,IRR_GLSL_LESS)


/**

TODOs:
Higher Priority:
- https://moderngpu.github.io/sortedsearch.html
We need to make a `irr_glsl_workgroupSort()` function as a utility to facilitate it.

Low Priority:
- https://moderngpu.github.io/bulkinsert.html
- https://moderngpu.github.io/merge.html 

**/

#endif
