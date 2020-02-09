// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_MACROS_H_INCLUDED__
#define __IRR_MACROS_H_INCLUDED__

#include "IrrCompileConfig.h"
#include "assert.h"

// basics
#define IRR_EVAL(...) __VA_ARGS__

#define IRR_CONCATENATE(X,Y) IRR_EVAL(X) ## IRR_EVAL(Y)


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
    #define _IRR_STATIC_INLINE static inline
    #define _IRR_STATIC_INLINE_CONSTEXPR static constexpr
#else
    #define _IRR_STATIC_INLINE static inline
    #define _IRR_STATIC_INLINE_CONSTEXPR static inline constexpr
#endif

// For dumb MSVC which now has to keep a spec bug to avoid breaking existing source code
#if defined(_MSC_VER)
    #define IRR_FORCE_EBO __declspec(empty_bases)
	#define IRR_NO_VTABLE __declspec(novtable)
#else
    #define IRR_FORCE_EBO
    #define IRR_NO_VTABLE
#endif // old FORCE_EMPTY_BASE_OPT

#if __GNUC__ < 7 || (__GNUC__ == 7 && (__GNUC_MINOR__ < 2)) // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=67054
    #define GCC_CONSTRUCTOR_INHERITANCE_BUG_WORKAROUND(DEFAULT_CTOR) DEFAULT_CTOR
#else
    #define GCC_CONSTRUCTOR_INHERITANCE_BUG_WORKAROUND(DEFAULT_CTOR)
#endif



// variadics


//
#define IRR_ARG_125(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,a32,a33,a34,a35,a36,a37,a38,a39,a40,a41,a42,a43,a44,a45,a46,a47,a48,a49,a50,a51,a52,a53,a54,a55,a56,a57,a58,a59,a60,a61,a62,a63,a64,a65,a66,a67,a68,a69,a70,a71,a72,a73,a74,a75,a76,a77,a78,a79,a80,a81,a82,a83,a84,a85,a86,a87,a88,a89,a90,a91,a92,a93,a94,a95,a96,a97,a98,a99,a100,a101,a102,a103,a104,a105,a106,a107,a108,a109,a110,a111,a112,a113,a114,a115,a116,a117,a118,a119,a120,a121,a122,a123,a124,a125, ... ) a125
#define IRR_VA_ARGS_COUNT( ... ) IRR_EVAL(IRR_ARG_125(__VA_ARGS__,125,124,123,122,121,120,119,118,117,116,115,114,113,112,111,110,109,108,107,106,105,104,103,102,101,100,99,98,97,96,95,94,93,92,91,90,89,88,87,86,85,84,83,82,81,80,79,78,77,76,75,74,73,72,71,70,69,68,67,66,65,64,63,62,61,60,59,58,57,56,55,54,53,52,51,50,49,48,47,46,45,44,43,42,41,40,39,38,37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0))

//
#define IRR_FOREACH_0(WHAT)
#define IRR_FOREACH_1(WHAT, X) IRR_EVAL(WHAT(X))
#define IRR_FOREACH_2(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_1(WHAT, __VA_ARGS__))
#define IRR_FOREACH_3(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_2(WHAT, __VA_ARGS__))
#define IRR_FOREACH_4(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_3(WHAT, __VA_ARGS__))
#define IRR_FOREACH_5(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_4(WHAT, __VA_ARGS__))
#define IRR_FOREACH_6(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_5(WHAT, __VA_ARGS__))
#define IRR_FOREACH_7(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_6(WHAT, __VA_ARGS__))
#define IRR_FOREACH_8(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_7(WHAT, __VA_ARGS__))
#define IRR_FOREACH_9(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_8(WHAT, __VA_ARGS__))
#define IRR_FOREACH_10(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_9(WHAT, __VA_ARGS__))
#define IRR_FOREACH_11(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_10(WHAT, __VA_ARGS__))
#define IRR_FOREACH_12(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_11(WHAT, __VA_ARGS__))
#define IRR_FOREACH_13(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_12(WHAT, __VA_ARGS__))
#define IRR_FOREACH_14(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_13(WHAT, __VA_ARGS__))
#define IRR_FOREACH_15(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_14(WHAT, __VA_ARGS__))
#define IRR_FOREACH_16(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_15(WHAT, __VA_ARGS__))
#define IRR_FOREACH_17(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_16(WHAT, __VA_ARGS__))
#define IRR_FOREACH_18(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_17(WHAT, __VA_ARGS__))
#define IRR_FOREACH_19(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_18(WHAT, __VA_ARGS__))
#define IRR_FOREACH_20(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_19(WHAT, __VA_ARGS__))
#define IRR_FOREACH_21(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_20(WHAT, __VA_ARGS__))
#define IRR_FOREACH_22(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_21(WHAT, __VA_ARGS__))
#define IRR_FOREACH_23(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_22(WHAT, __VA_ARGS__))
#define IRR_FOREACH_24(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_23(WHAT, __VA_ARGS__))
#define IRR_FOREACH_25(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_24(WHAT, __VA_ARGS__))
#define IRR_FOREACH_26(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_25(WHAT, __VA_ARGS__))
#define IRR_FOREACH_27(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_26(WHAT, __VA_ARGS__))
#define IRR_FOREACH_28(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_27(WHAT, __VA_ARGS__))
#define IRR_FOREACH_29(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_28(WHAT, __VA_ARGS__))
#define IRR_FOREACH_30(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_29(WHAT, __VA_ARGS__))
#define IRR_FOREACH_31(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_30(WHAT, __VA_ARGS__))
#define IRR_FOREACH_32(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_31(WHAT, __VA_ARGS__))
#define IRR_FOREACH_33(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_32(WHAT, __VA_ARGS__))
#define IRR_FOREACH_34(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_33(WHAT, __VA_ARGS__))
#define IRR_FOREACH_35(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_34(WHAT, __VA_ARGS__))
#define IRR_FOREACH_36(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_35(WHAT, __VA_ARGS__))
#define IRR_FOREACH_37(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_36(WHAT, __VA_ARGS__))
#define IRR_FOREACH_38(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_37(WHAT, __VA_ARGS__))
#define IRR_FOREACH_39(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_38(WHAT, __VA_ARGS__))
#define IRR_FOREACH_40(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_39(WHAT, __VA_ARGS__))
#define IRR_FOREACH_41(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_40(WHAT, __VA_ARGS__))
#define IRR_FOREACH_42(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_41(WHAT, __VA_ARGS__))
#define IRR_FOREACH_43(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_42(WHAT, __VA_ARGS__))
#define IRR_FOREACH_44(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_43(WHAT, __VA_ARGS__))
#define IRR_FOREACH_45(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_44(WHAT, __VA_ARGS__))
#define IRR_FOREACH_46(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_45(WHAT, __VA_ARGS__))
#define IRR_FOREACH_47(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_46(WHAT, __VA_ARGS__))
#define IRR_FOREACH_48(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_47(WHAT, __VA_ARGS__))
#define IRR_FOREACH_49(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_48(WHAT, __VA_ARGS__))
#define IRR_FOREACH_50(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_49(WHAT, __VA_ARGS__))
#define IRR_FOREACH_51(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_50(WHAT, __VA_ARGS__))
#define IRR_FOREACH_52(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_51(WHAT, __VA_ARGS__))
#define IRR_FOREACH_53(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_52(WHAT, __VA_ARGS__))
#define IRR_FOREACH_54(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_53(WHAT, __VA_ARGS__))
#define IRR_FOREACH_55(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_54(WHAT, __VA_ARGS__))
#define IRR_FOREACH_56(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_55(WHAT, __VA_ARGS__))
#define IRR_FOREACH_57(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_56(WHAT, __VA_ARGS__))
#define IRR_FOREACH_58(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_57(WHAT, __VA_ARGS__))
#define IRR_FOREACH_59(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_58(WHAT, __VA_ARGS__))
#define IRR_FOREACH_60(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_59(WHAT, __VA_ARGS__))
#define IRR_FOREACH_61(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_60(WHAT, __VA_ARGS__))
#define IRR_FOREACH_62(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_61(WHAT, __VA_ARGS__))
#define IRR_FOREACH_63(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_62(WHAT, __VA_ARGS__))
#define IRR_FOREACH_64(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_63(WHAT, __VA_ARGS__))
#define IRR_FOREACH_65(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_64(WHAT, __VA_ARGS__))
#define IRR_FOREACH_66(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_65(WHAT, __VA_ARGS__))
#define IRR_FOREACH_67(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_66(WHAT, __VA_ARGS__))
#define IRR_FOREACH_68(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_67(WHAT, __VA_ARGS__))
#define IRR_FOREACH_69(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_68(WHAT, __VA_ARGS__))
#define IRR_FOREACH_70(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_69(WHAT, __VA_ARGS__))
#define IRR_FOREACH_71(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_70(WHAT, __VA_ARGS__))
#define IRR_FOREACH_72(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_71(WHAT, __VA_ARGS__))
#define IRR_FOREACH_73(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_72(WHAT, __VA_ARGS__))
#define IRR_FOREACH_74(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_73(WHAT, __VA_ARGS__))
#define IRR_FOREACH_75(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_74(WHAT, __VA_ARGS__))
#define IRR_FOREACH_76(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_75(WHAT, __VA_ARGS__))
#define IRR_FOREACH_77(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_76(WHAT, __VA_ARGS__))
#define IRR_FOREACH_78(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_77(WHAT, __VA_ARGS__))
#define IRR_FOREACH_79(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_78(WHAT, __VA_ARGS__))
#define IRR_FOREACH_80(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_79(WHAT, __VA_ARGS__))
#define IRR_FOREACH_81(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_80(WHAT, __VA_ARGS__))
#define IRR_FOREACH_82(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_81(WHAT, __VA_ARGS__))
#define IRR_FOREACH_83(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_82(WHAT, __VA_ARGS__))
#define IRR_FOREACH_84(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_83(WHAT, __VA_ARGS__))
#define IRR_FOREACH_85(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_84(WHAT, __VA_ARGS__))
#define IRR_FOREACH_86(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_85(WHAT, __VA_ARGS__))
#define IRR_FOREACH_87(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_86(WHAT, __VA_ARGS__))
#define IRR_FOREACH_88(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_87(WHAT, __VA_ARGS__))
#define IRR_FOREACH_89(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_88(WHAT, __VA_ARGS__))
#define IRR_FOREACH_90(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_89(WHAT, __VA_ARGS__))
#define IRR_FOREACH_91(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_90(WHAT, __VA_ARGS__))
#define IRR_FOREACH_92(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_91(WHAT, __VA_ARGS__))
#define IRR_FOREACH_93(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_92(WHAT, __VA_ARGS__))
#define IRR_FOREACH_94(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_93(WHAT, __VA_ARGS__))
#define IRR_FOREACH_95(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_94(WHAT, __VA_ARGS__))
#define IRR_FOREACH_96(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_95(WHAT, __VA_ARGS__))
#define IRR_FOREACH_97(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_96(WHAT, __VA_ARGS__))
#define IRR_FOREACH_98(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_97(WHAT, __VA_ARGS__))
#define IRR_FOREACH_99(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_98(WHAT, __VA_ARGS__))
#define IRR_FOREACH_100(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_99(WHAT, __VA_ARGS__))
#define IRR_FOREACH_101(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_100(WHAT, __VA_ARGS__))
#define IRR_FOREACH_102(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_101(WHAT, __VA_ARGS__))
#define IRR_FOREACH_103(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_102(WHAT, __VA_ARGS__))
#define IRR_FOREACH_104(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_103(WHAT, __VA_ARGS__))
#define IRR_FOREACH_105(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_104(WHAT, __VA_ARGS__))
#define IRR_FOREACH_106(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_105(WHAT, __VA_ARGS__))
#define IRR_FOREACH_107(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_106(WHAT, __VA_ARGS__))
#define IRR_FOREACH_108(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_107(WHAT, __VA_ARGS__))
#define IRR_FOREACH_109(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_108(WHAT, __VA_ARGS__))
#define IRR_FOREACH_110(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_109(WHAT, __VA_ARGS__))
#define IRR_FOREACH_111(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_110(WHAT, __VA_ARGS__))
#define IRR_FOREACH_112(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_111(WHAT, __VA_ARGS__))
#define IRR_FOREACH_113(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_112(WHAT, __VA_ARGS__))
#define IRR_FOREACH_114(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_113(WHAT, __VA_ARGS__))
#define IRR_FOREACH_115(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_114(WHAT, __VA_ARGS__))
#define IRR_FOREACH_116(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_115(WHAT, __VA_ARGS__))
#define IRR_FOREACH_117(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_116(WHAT, __VA_ARGS__))
#define IRR_FOREACH_118(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_117(WHAT, __VA_ARGS__))
#define IRR_FOREACH_119(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_118(WHAT, __VA_ARGS__))
#define IRR_FOREACH_120(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_119(WHAT, __VA_ARGS__))
#define IRR_FOREACH_121(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_120(WHAT, __VA_ARGS__))
#define IRR_FOREACH_122(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_121(WHAT, __VA_ARGS__))
#define IRR_FOREACH_123(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_122(WHAT, __VA_ARGS__))
#define IRR_FOREACH_124(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_123(WHAT, __VA_ARGS__))
#define IRR_FOREACH_125(WHAT, X, ...) IRR_EVAL(WHAT(X)IRR_FOREACH_124(WHAT, __VA_ARGS__))

#define IRR_FOREACH(WHAT, ... ) IRR_EVAL(IRR_CONCATENATE(IRR_FOREACH_,IRR_VA_ARGS_COUNT(__VA_ARGS__))(WHAT, __VA_ARGS__))


#endif // __IRR_MACROS_H_INCLUDED__
