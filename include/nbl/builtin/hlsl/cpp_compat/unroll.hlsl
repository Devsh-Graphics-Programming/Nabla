#ifndef _NBL_BUILTIN_HLSL_CPP_COMPAT_UNROLL_INCLUDED_
#define _NBL_BUILTIN_HLSL_CPP_COMPAT_UNROLL_INCLUDED_

#ifdef __HLSL_VERSION
#define NBL_UNROLL [unroll]
#define NBL_UNROLL_LIMITED(LIMIT) [unroll(LIMIT)]
#else
#define NBL_UNROLL // can't be bothered / TODO
#define NBL_UNROLL_LIMITED(LIMIT)
#endif

#endif
