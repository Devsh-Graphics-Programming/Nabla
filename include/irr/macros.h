// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_MACROS_H_INCLUDED__
#define __IRR_MACROS_H_INCLUDED__

#include "IrrCompileConfig.h"
#include "assert.h"

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


//! Unifying common concepts via compiler specific macros
#ifndef IRR_FORCE_INLINE
	#ifdef _MSC_VER
		#define IRR_FORCE_INLINE __forceinline
	#else
		#define IRR_FORCE_INLINE inline
	#endif
#endif

// define a break macro for debugging.
#if defined(_IRR_WINDOWS_API_) && defined(_MSC_VER)
  #include <crtdbg.h>
  #define _IRR_BREAK_IF( _CONDITION_ ) if (_CONDITION_) {_CrtDbgBreak();}
#else
#include "signal.h"
#define _IRR_BREAK_IF( _CONDITION_ ) if ( (_CONDITION_) ) raise(SIGTRAP);
#endif

// kind like a debug only assert
#if defined(_IRR_DEBUG)
#define _IRR_DEBUG_BREAK_IF( _CONDITION_ ) _IRR_BREAK_IF(_CONDITION_)
#else
#define _IRR_DEBUG_BREAK_IF( _CONDITION_ )
#endif

// Defines a deprecated macro which generates a warning at compile time
#define _IRR_DEPRECATED_ [[deprecated]]

// Disables a switch case fallthrough warning for a particular case label
#if __cplusplus >= 201703L
    #define _IRR_FALLTHROUGH [[fallthrough]]
#else
    #define _IRR_FALLTHROUGH
#endif // __cplusplus

// Disables a maybe used uninitialized warning for a particular variable
#if __cplusplus >= 201703L
    #define _IRR_MAYBE_UNUSED [[maybe_unused]]
#else
    #define _IRR_MAYBE_UNUSED
#endif // __cplusplus

//! Workarounds for compiler specific bugs
// MSVC 2019 is a special snowflake
#if defined(_MSC_VER) && _MSC_VER>=1920
    #define IRR_TYPENAME_4_STTC_MBR typename
#else
    #define IRR_TYPENAME_4_STTC_MBR
#endif // _MSC_VER

// MSVC doesn't accept constexpr inline constants (claims it's c+17 feature) and ld (this linker which comes with GCC compiler) generates 'undefined references' without inline while defining in header file
#if defined(_MSC_VER) || __GNUC__ < 7
    #define _IRR_STATIC_INLINE_CONSTEXPR static constexpr
#else
    #define _IRR_STATIC_INLINE_CONSTEXPR static inline constexpr
#endif

// For dumb MSVC which now has to keep a spec bug to avoid breaking existing source code
#if defined(_MSC_VER)
    #define IRR_FORCE_EBO __declspec(empty_bases)
#else
    #define IRR_FORCE_EBO
#endif // old FORCE_EMPTY_BASE_OPT

#if __GNUC__ < 7 || (__GNUC__ == 7 && (__GNUC_MINOR__ < 2)) // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=67054
    #define GCC_CONSTRUCTOR_INHERITANCE_BUG_WORKAROUND(DEFAULT_CTOR) DEFAULT_CTOR
#else
    #define GCC_CONSTRUCTOR_INHERITANCE_BUG_WORKAROUND(DEFAULT_CTOR)
#endif

#endif // __IRR_MACROS_H_INCLUDED__
