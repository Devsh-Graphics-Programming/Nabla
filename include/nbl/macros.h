// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_MACROS_H_INCLUDED__
#define __NBL_MACROS_H_INCLUDED__

#include "nbl/core/decl/compile_config.h"
#include <nbl/builtin/hlsl/macros.h>
#include "assert.h"




//this one needs to be declared for every single child class for it to work
#define _NBL_NO_PUBLIC_DELETE(TYPE) \
            protected: \
                virtual ~TYPE()

#define _NBL_NO_PUBLIC_DELETE_DEFAULT(TYPE) \
            protected: \
                virtual ~TYPE() = default

//most probably useless (question: To virtual or to no virtual?)
#define _NBL_NO_DELETE_FINAL(TYPE) \
            private: \
                virtual ~TYPE()

#define _NBL_NO_DEFAULT_FINAL(TYPE) \
                TYPE() = delete

#define _NBL_NO_COPY_FINAL(TYPE) \
                TYPE(const TYPE& other) = delete; \
                TYPE& operator=(const TYPE& other) = delete

#define _NBL_NO_MOVE_FINAL(TYPE) \
                TYPE(TYPE&& other) = delete; \
                TYPE& operator=(TYPE&& other) = delete

//
#define _NBL_NO_NEW_DELETE \
            static inline void* operator new(size_t size)                = delete; \
            static inline void* operator new[](size_t size)              = delete; \
            static inline void* operator new(size_t size, void* where)   = delete; \
            static inline void* operator new[](size_t size, void* where) = delete; \
            static inline void operator delete(void* ptr)                = delete; \
            static inline void operator delete[](void* ptr)              = delete; \
            static inline void operator delete(void* ptr, size_t size)   = delete; \
            static inline void operator delete[](void* ptr, size_t size) = delete; \
            static inline void operator delete(void* dummy, void* ptr)   = delete


//! Unifying common concepts via compiler specific macros
#ifndef NBL_FORCE_INLINE
	#ifdef _MSC_VER
		#define NBL_FORCE_INLINE __forceinline
	#else
		#define NBL_FORCE_INLINE inline
	#endif
#endif

// define a break macro for debugging.
#if defined(_NBL_WINDOWS_API_) && defined(_MSC_VER)
  #include <crtdbg.h>
  #define _NBL_BREAK_IF( _CONDITION_ ) if (_CONDITION_) {_CrtDbgBreak();}
#else
#include "signal.h"
#define _NBL_BREAK_IF( _CONDITION_ ) if ( (_CONDITION_) ) raise(SIGTRAP);
#endif

// kind like a debug only assert
#if defined(_NBL_DEBUG) || defined(_NBL_RELWITHDEBINFO)
// TODO even though it is defined in RWDI build, _DEBUG #define is not defined in msvc (in rwdi) so debug break is not triggered anyway
// idk what about other compilers
#define _NBL_DEBUG_BREAK_IF( _CONDITION_ ) _NBL_BREAK_IF(_CONDITION_)
#else
#define _NBL_DEBUG_BREAK_IF( _CONDITION_ )
#endif

#define _NBL_TODO() _NBL_DEBUG_BREAK_IF(true)

//! Workarounds for compiler specific bugs
// MSVC 2019 is a special snowflake
#if defined(_MSC_VER) && _MSC_VER>=1920
    #define NBL_TYPENAME_4_STTC_MBR typename
#else
    #define NBL_TYPENAME_4_STTC_MBR
#endif // _MSC_VER

// MSVC doesn't accept constexpr inline constants (claims it's c+17 feature) and ld (this linker which comes with GCC compiler) generates 'undefined references' without inline while defining in header file
#if defined(_MSC_VER) || __GNUC__ < 7
    #define _NBL_STATIC_INLINE static inline
    #define _NBL_STATIC_INLINE_CONSTEXPR static constexpr
#else
    #define _NBL_STATIC_INLINE static inline
    #define _NBL_STATIC_INLINE_CONSTEXPR static inline constexpr
#endif

// `arg` arg must be of pointer type, must be mutable and must be lvalue
#ifdef _MSC_VER
    #define NBL_ASSUME_ALIGNED_NDEBUG(arg, align) __assume((reinterpret_cast<const char*>(arg) - reinterpret_cast<const char*>(0)) % (align) == 0)
#elif (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 7)
    #define NBL_ASSUME_ALIGNED_NDEBUG(arg, align) arg = reinterpret_cast<decltype(arg)>(__builtin_assume_aligned(arg, align))
#else
    #define NBL_ASSUME_ALIGNED_NDEBUG(arg, align)
#endif
#define NBL_ASSUME_ALIGNED(arg,align)	NBL_ASSUME_ALIGNED_NDEBUG(arg,align); \
										assert(core::isPoT(align)); \
										assert((reinterpret_cast<const char*>(arg) - reinterpret_cast<const char*>(0)) % (align) == 0)

// For dumb MSVC which now has to keep a spec bug to avoid breaking existing source code
#if defined(_MSC_VER)
    #define NBL_FORCE_EBO __declspec(empty_bases)
	#define NBL_NO_VTABLE __declspec(novtable)
#else
    #define NBL_FORCE_EBO
    #define NBL_NO_VTABLE
#endif // old FORCE_EMPTY_BASE_OPT

#if __GNUC__ < 7 || (__GNUC__ == 7 && (__GNUC_MINOR__ < 2)) // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=67054
    #define GCC_CONSTRUCTOR_INHERITANCE_BUG_WORKAROUND(DEFAULT_CTOR) DEFAULT_CTOR
#else
    #define GCC_CONSTRUCTOR_INHERITANCE_BUG_WORKAROUND(DEFAULT_CTOR)
#endif


#define NBL_ENUM_ADD_BITWISE_OPERATORS(EnumType)												\
	inline constexpr EnumType operator&(const EnumType& lhs, const EnumType& rhs) noexcept		\
	{																							\
		using T = typename std::underlying_type_t<EnumType>;									\
		return static_cast<EnumType>(static_cast<T>(lhs) & static_cast<T>(rhs));				\
	}																							\
																								\
	inline constexpr EnumType& operator&=(EnumType& lhs, const EnumType& rhs) noexcept			\
	{																							\
	return lhs = lhs & rhs;																		\
	}																							\
																								\
	inline constexpr EnumType operator|(const EnumType& lhs, const EnumType& rhs) noexcept		\
	{																							\
	using T = typename std::underlying_type_t<EnumType>;										\
	return static_cast<EnumType>(static_cast<T>(lhs) | static_cast<T>(rhs));					\
	}																							\
																								\
	inline constexpr EnumType& operator|=(EnumType& lhs, const EnumType& rhs) noexcept			\
	{																							\
	return lhs = lhs | rhs;																		\
	}																							\
																								\
	inline constexpr EnumType operator^(const EnumType& lhs, const EnumType& rhs) noexcept		\
	{																							\
	using T = typename std::underlying_type_t<EnumType>;										\
	return static_cast<EnumType>(static_cast<T>(lhs) ^ static_cast<T>(rhs));					\
	}																							\
																								\
	inline constexpr EnumType& operator^=(EnumType& lhs, const EnumType& rhs) noexcept			\
	{																							\
	return lhs = lhs ^ rhs;																		\
	}																							\
																								\
	inline constexpr EnumType operator~(const EnumType& e) noexcept								\
	{																							\
	using T = typename std::underlying_type_t<EnumType>;										\
	return static_cast<EnumType>(~static_cast<T>(e));											\
	}																							\
	/**/


#endif
