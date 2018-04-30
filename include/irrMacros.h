// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_MACROS_H_INCLUDED__
#define __IRR_MACROS_H_INCLUDED__

#include "IrrCompileConfig.h"


//! When a new include file irrMemory.h will be made the following will move there
//! -- these are stubs not used anywhere yet
#define _IRR_NEW_OBJ(_obj)      new _obj
#define _IRR_DELETE_OBJ(_obj)   delete _obj

#define _IRR_NEW_ARRAY(_obj)    new _obj[]
#define _IRR_DELETE_ARRAY(_obj) delete [] _obj

/*
//this will always allocate to default _IRR_MEMORY_ALIGNMENT =32
#ifdef _IRR_WINDOWS_
    #define _IRR_MALLOC(size)       _aligned_malloc(size,32)
    #define _IRR_FREE(addr)         _aligned_free(addr)
#else
    #define _IRR_MALLOC(size)       posix_memalign(size,32)
    #define _IRR_FREE(addr)         _aligned_free(addr)
		posix_memalign((void**)&memoryallocatedaligned,Alignment,cnt);
#endif
*/
//! END of section


//this one needs to be declared for every single child class for it to work
#define _IRR_NO_PUBLIC_DELETE(TYPE) \
            protected: \
                virtual ~TYPE()

#define _IRR_NO_PUBLIC_DELETE_DEFAULT(TYPE) \
            protected: \
                virtual ~TYPE() = default

//most probably useless (question: To virtual or to no virtual?)
#define _IRR_NO_DELETE_FINAL(TYPE) \
            private: \
                virtual ~TYPE()

#define _IRR_NO_DEFAULT_FINAL(TYPE) \
                TYPE() = delete

#define _IRR_NO_COPY_FINAL(TYPE) \
                TYPE(const TYPE& other) = delete; \
                TYPE& operator=(const TYPE& other) = delete

#define _IRR_NO_MOVE_FINAL(TYPE) \
                TYPE(TYPE&& other) = delete; \
                TYPE& operator=(TYPE&& other) = delete



//! define a break macro for debugging.
#if defined(_DEBUG)
#if defined(_IRR_WINDOWS_API_) && defined(_MSC_VER) && !defined (_WIN32_WCE)
  #if defined(WIN64) || defined(_WIN64) // using portable common solution for x64 configuration
  #include <crtdbg.h>
  #define _IRR_DEBUG_BREAK_IF( _CONDITION_ ) if (_CONDITION_) {_CrtDbgBreak();}
  #else
  #define _IRR_DEBUG_BREAK_IF( _CONDITION_ ) if (_CONDITION_) {_asm int 3}
  #endif
#else
#include "assert.h"
#define _IRR_DEBUG_BREAK_IF( _CONDITION_ ) assert( !(_CONDITION_) );
#endif
#else
#define _IRR_DEBUG_BREAK_IF( _CONDITION_ )
#endif

//! Defines a deprecated macro which generates a warning at compile time
/** The usage is simple
For typedef:		typedef _IRR_DEPRECATED_ int test1;
For classes/structs:	class _IRR_DEPRECATED_ test2 { ... };
For methods:		class test3 { _IRR_DEPRECATED_ virtual void foo() {} };
For functions:		template<class T> _IRR_DEPRECATED_ void test4(void) {}
**/
#if defined(IGNORE_DEPRECATED_WARNING)
#define _IRR_DEPRECATED_
#elif _MSC_VER >= 1310 //vs 2003 or higher
#define _IRR_DEPRECATED_ __declspec(deprecated)
#elif (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 1)) // all versions above 3.0 should support this feature
#define _IRR_DEPRECATED_  __attribute__ ((deprecated))
#else
#define _IRR_DEPRECATED_
#endif

#endif // __IRR_MACROS_H_INCLUDED__
