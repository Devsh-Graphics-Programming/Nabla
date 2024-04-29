#ifndef _NBL_DISABLE_3RDPARTY_WARNINGS_
#define _NBL_DISABLE_3RDPARTY_WARNINGS_

#ifdef _MSC_VER

#define NBL_PUSH_DISABLE_WARNINGS __pragma(warning(push, 0))
#define NBL_POP_DISABLE_WARNINGS __pragma(warning(pop))

#else

#endif

#endif