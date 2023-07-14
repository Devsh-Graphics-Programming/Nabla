// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SUBGROUP_ARITHMETIC_PORTABILITY_IMPL_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP_ARITHMETIC_PORTABILITY_IMPL_INCLUDED_

#include "nbl/builtin/hlsl/binops.hlsl"
#include "nbl/builtin/hlsl/subgroup/scratch.hlsl"
#include "nbl/builtin/hlsl/subgroup/shuffle_portability.hlsl"

// REVIEW:  Location and need of these. They need to be over a function but
//          there's no need to have them over every subgroup func.
//          The compiler doesn't seem to whine about these missing
//          for subgroup*Min/Max functions even though the spec seems
//          to mandate them https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#Group_Operation
[[vk::ext_extension("GL_KHR_shader_subgroup_basic")]]
[[vk::ext_extension("GL_KHR_shader_subgroup_arithmetic")]]
[[vk::ext_capability(/* GroupNonUniformArithmetic */ 63)]]
[[vk::ext_capability(/* GroupNonUniformBallot */ 64)]]
void fake_for_capability_and_extension(){}

//const uint LastWorkgroupInvocation = _NBL_HLSL_WORKGROUP_SIZE_-1; // REVIEW: Where should this be defined?
#define LastWorkgroupInvocation (_NBL_HLSL_WORKGROUP_SIZE_-1U)

namespace nbl
{
namespace hlsl
{
namespace subgroup
{

#ifdef NBL_GL_KHR_shader_subgroup_arithmetic
namespace native
{

template<typename T, class Binop>
struct reduction;
template<typename T, class Binop>
struct exclusive_scan;
template<typename T, class Binop>
struct inclusive_scan;

// *** AND ***
template<typename T>
struct reduction<T, binops::bitwise_and<T> >
{
    T operator()(const T x)
    {
        return WaveActiveBitAnd(x);
    }
};

// For all WaveMultiPrefix* ops, an example can be found here https://github.com/microsoft/DirectXShaderCompiler/blob/4e5440e1ee1f30d1164f90445611328293de08fa/tools/clang/test/HLSLFileCheck/hlsl/intrinsics/wave/prefix/sm_6_5_wave.hlsl
// However, it seems they do not work for DXC->SPIR-V yet so we implement them via spirv intrinsics

template<typename T>
[[vk::ext_instruction(359)]]
T spirv_subgroupPrefixAnd(uint scope, [[vk::ext_literal]] uint operation, T value);

template<typename T>
struct inclusive_scan<T, binops::bitwise_and<T> >
{
    T operator()(const T x)
    {
        return spirv_subgroupPrefixAnd(3, 1, x);
    }
};

template<typename T>
struct exclusive_scan<T, binops::bitwise_and<T> >
{
    T operator()(const T x)
    {
        return spirv_subgroupPrefixAnd(3, 2, x);
    }
};

// *** OR ***
template<typename T>
struct reduction<T, binops::bitwise_or<T> >
{
    T operator()(const T x)
    {
        return WaveActiveBitOr(x);
    }
};

//template<typename T>
[[vk::ext_instruction(360)]]
//T spirv_subgroupPrefixOr(uint scope, [[vk::ext_literal]] uint operation, T value);
uint spirv_subgroupPrefixOr(uint scope, [[vk::ext_literal]] uint operation, uint value);

template<typename T>
struct inclusive_scan<T, binops::bitwise_or<T> >
{
    T operator()(const T x)
    {
        return spirv_subgroupPrefixOr(3, 1, x);
    }
};

template<typename T>
struct exclusive_scan<T, binops::bitwise_or<T> >
{
    T operator()(const T x)
    {
        return spirv_subgroupPrefixOr(3, 2, x);
    }
};

// *** XOR ***
template<typename T>
struct reduction<T, binops::bitwise_xor<T> >
{
    T operator()(const T x)
    {
        return WaveActiveBitXor(x);
    }
};


template<typename T>
[[vk::ext_instruction(361)]]
T spirv_subgroupPrefixXor(uint scope, [[vk::ext_literal]] uint operation, T value);

template<typename T>
struct inclusive_scan<T, binops::bitwise_xor<T> >
{
    T operator()(const T x)
    {
        return spirv_subgroupPrefixXor(3, 1, x);
    }
};

template<typename T>
struct exclusive_scan<T, binops::bitwise_xor<T> >
{
    T operator()(const T x)
    {
        return spirv_subgroupPrefixXor(3, 2, x);
    }
};

// *** ADD ***
template<typename T>
struct reduction<T, binops::add<T> >
{
    T operator()(const T x)
    {
        return WaveActiveSum(x);
    }
};
template<typename T>
struct exclusive_scan<T, binops::add<T> >
{
    T operator()(const T x)
    {
        return WavePrefixSum(x);
    }
};
template<typename T>
struct inclusive_scan<T, binops::add<T> >
{
    T operator()(const T x)
    {
        return WavePrefixSum(x) + x;
    }
};

// *** MUL ***
template<typename T>
struct reduction<T, binops::mul<T> >
{
    T operator()(const T x)
    {
        return WaveActiveProduct(x);
    }
};
template<typename T>
struct exclusive_scan<T, binops::mul<T> >
{
    T operator()(const T x)
    {
        return WavePrefixProduct(x);
    }
};
template<typename T>
struct inclusive_scan<T, binops::mul<T> >
{
    T operator()(const T x)
    {
        return WavePrefixProduct(x) * x;
    }
};

// *** MIN ***
template<typename T>
struct reduction<T, binops::min<T> >
{
    T operator()(const T x)
    {
        return WaveActiveMin(x);
    }
};

// The MIN and MAX operations in SPIR-V have different Ops for each type
// so we implement them distinctly

[[vk::ext_instruction(353)]]
int spirv_subgroupPrefixMin(uint scope, [[vk::ext_literal]] uint operation, int value);
[[vk::ext_instruction(354)]]
uint spirv_subgroupPrefixMin(uint scope, [[vk::ext_literal]] uint operation, uint value);
[[vk::ext_instruction(355)]]
float spirv_subgroupPrefixMin(uint scope, [[vk::ext_literal]] uint operation, float value);

template<>
struct inclusive_scan<int, binops::min<int> >
{
    int operator()(const int x)
    {
        return spirv_subgroupPrefixMin(3, 1, x);
    }
};

template<>
struct inclusive_scan<uint, binops::min<uint> >
{
    uint operator()(const uint x)
    {
        return spirv_subgroupPrefixMin(3, 1, x);
    }
};

template<>
struct inclusive_scan<uint, binops::min<float> >
{
    float operator()(const float x)
    {
        return spirv_subgroupPrefixMin(3, 1, x);
    }
};

template<>
struct exclusive_scan<int, binops::min<int> >
{
    int operator()(const int x)
    {
        return spirv_subgroupPrefixMin(3, 2, x);
    }
};

template<>
struct exclusive_scan<uint, binops::min<uint> >
{
    uint operator()(const uint x)
    {
        return spirv_subgroupPrefixMin(3, 2, x);
    }
};

template<>
struct exclusive_scan<uint, binops::min<float> >
{
    float operator()(const float x)
    {
        return spirv_subgroupPrefixMin(3, 2, x);
    }
};

// *** MAX ***
template<typename T>
struct reduction<T, binops::max<T> >
{
    T operator()(const T x)
    {
        return WaveActiveMax(x);
    }
};

[[vk::ext_instruction(356)]]
int spirv_subgroupPrefixMax(uint scope, [[vk::ext_literal]] uint operation, int value);
[[vk::ext_instruction(357)]]
uint spirv_subgroupPrefixMax(uint scope, [[vk::ext_literal]] uint operation, uint value);
[[vk::ext_instruction(358)]]
float spirv_subgroupPrefixMax(uint scope, [[vk::ext_literal]] uint operation, float value);

template<>
struct inclusive_scan<int, binops::max<int> >
{
    int operator()(const int x)
    {
        return spirv_subgroupPrefixMax(3, 1, x);
    }
};

template<>
struct inclusive_scan<uint, binops::max<uint> >
{
    uint operator()(const uint x)
    {
        return spirv_subgroupPrefixMax(3, 1, x);
    }
};

template<>
struct inclusive_scan<uint, binops::max<float> >
{
    float operator()(const float x)
    {
        return spirv_subgroupPrefixMax(3, 1, x);
    }
};

template<>
struct exclusive_scan<int, binops::max<int> >
{
    int operator()(const int x)
    {
        return spirv_subgroupPrefixMax(3, 2, x);
    }
};

template<>
struct exclusive_scan<uint, binops::max<uint> >
{
    uint operator()(const uint x)
    {
        return spirv_subgroupPrefixMax(3, 2, x);
    }
};

template<>
struct exclusive_scan<uint, binops::max<float> >
{
    float operator()(const float x)
    {
        return spirv_subgroupPrefixMax(3, 2, x);
    }
};

}
#else
namespace portability
{
template<typename T, class Binop, class ScratchAccessor, bool initializeScratch>
struct inclusive_scan
{
    static inclusive_scan<T, Binop, ScratchAccessor, initializeScratch> create()
    {
		inclusive_scan<T, Binop, ScratchAccessor, initializeScratch> retval;
		retval.offsetsAndMasks = ScratchOffsetsAndMasks::WithSubgroupOpDefaults();
		return retval;
    }

    T operator()(T value)
    {
		Binop op;

		if (initializeScratch)
		{
			Barrier();
			MemoryBarrierShared();
			scratchInitialize<ScratchAccessor, T>(value, op.identity(), _NBL_HLSL_WORKGROUP_SIZE_-1);
		}
		Barrier();
		MemoryBarrierShared();
		
		// Stone-Kogge adder
		// (devsh): it seems that lanes below <HalfSubgroupSize/step are doing useless work,
		// but they're SIMD and adding an `if`/conditional execution is more expensive
	#if 1 // Use shuffling by default (either native or portability implementation)
		uint toAdd = ShuffleUp<T, ScratchAccessor>(value, 1u); // all invocations must execute the shuffle, even if we don't apply the op() to all of them
		if(offsetsAndMasks.subgroupInvocation >= 1u) {
			// the first invocation (index 0) in the subgroup doesn't have anything in its left
			value = op(value, toAdd);
		}
	#else
		value = op(value, scratch.main.get(offsetsAndMasks.scanStoreOffset-1u));
	#endif
		[[unroll]] // REVIEW: I copied this from the example implementation on the github issue but if I'm not mistaken unroll doesn't work since halfSubgroupSize is not constexpr 
		for (uint step=2u; step <= offsetsAndMasks.halfSubgroupSize; step <<= 1u)
		{
		#if 1
			// there is no scratch and padding entries in this case so we have to guard the shuffles to not go out of bounds
			toAdd = ShuffleUp<T, ScratchAccessor>(value, step);
			if(offsetsAndMasks.subgroupInvocation >= step) {
				value = op(value, toAdd);
			}
		#else
			Barrier();
			MemoryBarrierShared();
			scratch.main.set(offsetsAndMasks.scanStoreOffset, value);
			Barrier();
			MemoryBarrierShared();
			value = op(value, scratch.main.get(offsetsAndMasks.scanStoreOffset - step));
			Barrier();
			MemoryBarrierShared();
			scratch.main.set(offsetsAndMasks.scanStoreOffset, value);
			Barrier();
			MemoryBarrierShared();
			
			// REVIEW: The Stone-Kogge adder is done at this point.
			// The GLSL implementation however has a final operation that uses the lastLoadOffset
			// but it actually messes the results, not sure what the point was.
			//value = op(value, scratch.main.get(offsetsAndMasks.lastLoadOffset));
			//Barrier();
			//MemoryBarrierShared();
		#endif
		}
		return value;
    }
// protected:
	ScratchAccessor scratch;
	ScratchOffsetsAndMasks offsetsAndMasks;
};

template<typename T, class Binop, class ScratchAccessor, bool initializeScratch>
struct exclusive_scan
{
    static exclusive_scan<T, Binop, ScratchAccessor, initializeScratch> create()
    {
        exclusive_scan<T, Binop, ScratchAccessor, initializeScratch> retval;
        retval.impl = inclusive_scan<T, Binop, ScratchAccessor, initializeScratch>::create();
        return retval;
    }

    T operator()(T value)
    {
		value = impl(value);

		// store value to smem so we can shuffle it
	#if 1
		Binop op;
		uint left = ShuffleUp<T, ScratchAccessor>(value, 1);
		value = impl.offsetsAndMasks.subgroupInvocation >= 1 ? left : op.identity(); // the first invocation doesn't have anything in its left so we set to the binop's identity value for exlusive scan
	#else
		impl.scratch.main.set(impl.offsetsAndMasks.scanStoreOffset,value);
		Barrier();
		MemoryBarrierShared();
		// get previous item
		value = impl.scratch.main.get(impl.offsetsAndMasks.scanStoreOffset-1u);
		Barrier();
		MemoryBarrierShared();
	#endif
		// return it
		return value;
    }

// protected:
	inclusive_scan<T, Binop, ScratchAccessor, initializeScratch> impl;
};

template<typename T, class Binop, class ScratchAccessor, bool initializeScratch>
struct reduction
{
    static reduction<T, Binop, ScratchAccessor, initializeScratch> create()
    {
        reduction<T, Binop, ScratchAccessor, initializeScratch> retval;
        retval.impl = inclusive_scan<T, Binop, ScratchAccessor, initializeScratch>::create();
        return retval;
    }

    T operator()(T value)
    {
		value = impl(value);
	#if 1
		value = Shuffle<T, ScratchAccessor>(value, impl.offsetsAndMasks.lastSubgroupInvocation); // take the last subgroup invocation's value
	#else
		impl.scratch.main.set(impl.offsetsAndMasks.scanStoreOffset, value);
		Barrier();
		MemoryBarrierShared();
		uint reductionResultOffset = impl.offsetsAndMasks.subgroupPaddingMemoryEnd + impl.offsetsAndMasks.lastSubgroupInvocation; // this should end up being the last subgroup invocation
		// store value to smem so we can broadcast it to everyone
		value = impl.scratch.main.get(reductionResultOffset);
		Barrier();
		MemoryBarrierShared();
	#endif
		// return it
		return value;
    }
// protected:
    inclusive_scan<T, Binop, ScratchAccessor, initializeScratch> impl;
};
}
#endif
}
}
}

#endif