// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SUBGROUP_ARITHMETIC_PORTABILITY_IMPL_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP_ARITHMETIC_PORTABILITY_IMPL_INCLUDED_

#include <nbl/builtin/hlsl/subgroup/scratch.hlsl>

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

#define WHOLE_WAVE ~0

#ifndef _NBL_GL_LOCAL_INVOCATION_IDX_DECLARED_
#define _NBL_GL_LOCAL_INVOCATION_IDX_DECLARED_
const uint gl_LocalInvocationIndex : SV_GroupIndex; // REVIEW: Discuss proper placement of SV_* values. They are not allowed to be defined inside a function scope, only as arguments of main() or global variables in the shader.
#endif

const uint LastWorkgroupInvocation = _NBL_HLSL_WORKGROUP_SIZE_-1; // REVIEW: Where should this be defined?

namespace nbl
{
namespace hlsl
{
namespace subgroup
{

#ifdef NBL_GL_KHR_shader_subgroup_arithmetic
namespace native
{

// *** AND ***
template<typename T>
struct reduction<T, binops::bitwise_and<T> >
{
    T operator()(const T x)
    {
        return WaveActiveBitAnd(x);
    }
};
template<typename T>
struct exclusive_scan<T, binops::bitwise_and<T> >
{
    T operator()(const T x)
    {
        return WaveMultiPrefixAnd(x, WHOLE_WAVE);
    }
};
template<typename T>
struct inclusive_scan<T, binops::bitwise_and<T> >
{
    T operator()(const T x)
    {
        return WaveMultiPrefixAnd(x, WHOLE_WAVE) & x;
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
template<typename T>
struct exclusive_scan<T, binops::bitwise_or<T> >
{
    T operator()(const T x)
    {
        return WaveMultiPrefixOr(x, WHOLE_WAVE);
    }
};
template<typename T>
struct inclusive_scan<T, binops::bitwise_or<T> >
{
    T operator()(const T x)
    {
        return WaveMultiPrefixOr(x, WHOLE_WAVE) | x;
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
struct exclusive_scan<T, binops::bitwise_xor<T> >
{
    T operator()(const T x)
    {
        return WaveMultiPrefixXor(x, WHOLE_WAVE);
    }
};
template<typename T>
struct inclusive_scan<T, binops::bitwise_xor<T> >
{
    T operator()(const T x)
    {
        return WaveMultiPrefixXor(x, WHOLE_WAVE) ^ x;
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

[[vk::ext_instruction(353)]]
int spirv_subgroupInclusiveMin(uint scope, [[vk::ext_literal]] uint operation, int value);
int subgroupInclusiveMin(int t) {
    return spirv_subgroupInclusiveMin(3, 1, t);
}
template<>
struct inclusive_scan<int, binops::min<int> >
{
    int operator()(const int x)
    {
        return subgroupInclusiveMin(x);
    }
};

[[vk::ext_instruction(354)]]
uint spirv_subgroupInclusiveMin(uint scope, [[vk::ext_literal]] uint operation, uint value);
uint subgroupInclusiveMin(uint t) {
    return spirv_subgroupInclusiveMin(3, 1, t);
}
template<>
struct inclusive_scan<uint, binops::min<uint> >
{
    uint operator()(const uint x)
    {
        return subgroupInclusiveMin(x);
    }
};

[[vk::ext_instruction(355)]]
float spirv_subgroupInclusiveMin(uint scope, [[vk::ext_literal]] uint operation, float value);
float subgroupInclusiveMin(float t) {
    return spirv_subgroupInclusiveMin(3, 1, t);
}
template<>
struct inclusive_scan<uint, binops::min<float> >
{
    float operator()(const float x)
    {
        return subgroupInclusiveMin(x);
    }
};

[[vk::ext_instruction(353)]]
int spirv_subgroupExclusiveMin(uint scope, [[vk::ext_literal]] uint operation, int value);
int subgroupExclusiveMin(int t) {
    return spirv_subgroupExclusiveMin(3, 2, t);
}
template<>
struct exclusive_scan<int, binops::min<int> >
{
    int operator()(const int x)
    {
        return subgroupExclusiveMin(x);
    }
};

[[vk::ext_instruction(354)]]
uint spirv_subgroupExclusiveMin(uint scope, [[vk::ext_literal]] uint operation, uint value);
uint subgroupExclusiveMin(uint t) {
    return spirv_subgroupExclusiveMin(3, 2, t);
}
template<>
struct exclusive_scan<uint, binops::min<uint> >
{
    uint operator()(const uint x)
    {
        return subgroupExclusiveMin(x);
    }
};

[[vk::ext_instruction(355)]]
float spirv_subgroupExclusiveMin(uint scope, [[vk::ext_literal]] uint operation, float value);
float subgroupExclusiveMin(float t) {
    return spirv_subgroupExclusiveMin(3, 2, t);
}
template<>
struct exclusive_scan<uint, binops::min<float> >
{
    float operator()(const float x)
    {
        return subgroupExclusiveMin(x);
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
int spirv_subgroupInclusiveMax(uint scope, [[vk::ext_literal]] uint operation, int value);
int subgroupInclusiveMax(int t) {
    return spirv_subgroupInclusiveMax(3, 1, t);
}
template<>
struct inclusive_scan<int, binops::max<int> >
{
    int operator()(const int x)
    {
        return subgroupInclusiveMax(x);
    }
};

[[vk::ext_instruction(357)]]
uint spirv_subgroupInclusiveMax(uint scope, [[vk::ext_literal]] uint operation, uint value);
uint subgroupInclusiveMax(uint t) {
    return spirv_subgroupInclusiveMax(3, 1, t);
}
template<>
struct inclusive_scan<uint, binops::max<uint> >
{
    uint operator()(const uint x)
    {
        return subgroupInclusiveMax(x);
    }
};

[[vk::ext_instruction(358)]]
float spirv_subgroupInclusiveMax(uint scope, [[vk::ext_literal]] uint operation, float value);
float subgroupInclusiveMax(float t) {
    return spirv_subgroupInclusiveMin(3, 1, t);
}
template<>
struct inclusive_scan<uint, binops::max<float> >
{
    float operator()(const float x)
    {
        return subgroupInclusiveMax(x);
    }
};

[[vk::ext_instruction(356)]]
int spirv_subgroupExclusiveMax(uint scope, [[vk::ext_literal]] uint operation, int value);
int subgroupExclusiveMax(int t) {
    return spirv_subgroupExclusiveMax(3, 2, t);
}
template<>
struct exclusive_scan<int, binops::max<int> >
{
    int operator()(const int x)
    {
        return subgroupExclusiveMax(x);
    }
};

[[vk::ext_instruction(357)]]
uint spirv_subgroupExclusiveMax(uint scope, [[vk::ext_literal]] uint operation, uint value);
uint subgroupExclusiveMax(uint t) {
    return spirv_subgroupExclusiveMax(3, 2, t);
}
template<>
struct exclusive_scan<uint, binops::max<uint> >
{
    uint operator()(const uint x)
    {
        return subgroupExclusiveMax(x);
    }
};

[[vk::ext_instruction(358)]]
float spirv_subgroupExclusiveMax(uint scope, [[vk::ext_literal]] uint operation, float value);
float subgroupExclusiveMax(float t) {
    return spirv_subgroupExclusiveMax(3, 2, t);
}
template<>
struct exclusive_scan<uint, binops::max<float> >
{
    float operator()(const float x)
    {
        return subgroupExclusiveMax(x);
    }
};

}
#endif

namespace portability
{

template<class Binop, class ScratchAccessor>
struct inclusive_scan;

struct scan_base
{
    template<class Binop, class ScratchAccessor>
	static inclusive_scan<Binop, ScratchAccessor> create()
	{
		inclusive_scan<Binop, ScratchAccessor> retval;
		retval.offsetsAndMasks = ScratchOffsetsAndMasks::WithDefaults();
		return retval;
    }

// protected:
    ScratchOffsetsAndMasks offsetsAndMasks;
};

template<class Binop, class ScratchAccessor>
struct inclusive_scan : scan_base
{
    static inclusive_scan<Binop,ScratchAccessor> create()
    {    
		return scan_base::create<Binop,ScratchAccessor>(); // REVIEW: Is this correct?
    }

    template<typename T, bool initializeScratch>
    T operator()(T value)
    {
		Binop op;

		if (initializeScratch)
		{
			Barrier();
			MemoryBarrierShared();

			// each invocation initializes its respective slot with its value
			scratchAccessor.set(offsetsAndMasks.scanStoreOffset ,value);

			// additionally, the first half invocations initialize the padding slots
			// with identity values
			if (offsetsAndMasks.subgroupInvocation < offsetsAndMasks.halfSubgroupSize)
				scratchAccessor.set(offsetsAndMasks.lastLoadOffset, op.identity());
		}
		Barrier();
		MemoryBarrierShared();
		// Stone-Kogge adder
		// (devsh): it seems that lanes below <HalfSubgroupSize/step are doing useless work,
		// but they're SIMD and adding an `if`/conditional execution is more expensive
	#ifdef NBL_GL_KHR_shader_subgroup_shuffle
		if(offsetsAndMasks.subgroupInvocation >= 1u)
			// the first invocation (index 0) in the subgroup doesn't have anything in its left
			value = op(value, ShuffleUp(value, 1u));
	#else
		value = op(value, scratchAccessor.get(offsetsAndMasks.scanStoreOffset-1u));
	#endif
		[[unroll]]
		for (uint step=2u; step <= offsetsAndMasks.halfSubgroupSize; step <<= 1u)
		{
		#ifdef NBL_GL_KHR_shader_subgroup_shuffle // REVIEW: maybe use it by default?
			// there is no scratch and padding entries in this case so we have to guard the shuffles to not go out of bounds
			if(offsetsAndMasks.subgroupInvocation >= step)
				value = op(value, ShuffleUp(value, step));
		#else
			scratchAccessor.set(offsetsAndMasks.scanStoreOffset, value);
			Barrier();
			MemoryBarrierShared();
			value = op(value, scratchAccessor.get(offsetsAndMasks.scanStoreOffset - step));
			Barrier();
			MemoryBarrierShared();
		#endif
		}
		return value;
    }

    template<typename T>
    T operator()(const T value)
    {
        return operator()<T,true>(value);
    }
// protected:
	ScratchAccessor scratchAccessor;
};

template<class Binop, class ScratchAccessor>
struct exclusive_scan
{
    static exclusive_scan<Binop,ScratchAccessor> create()
    {
        exclusive_scan<Binop, ScratchAccessor> retval;
        retval.impl = inclusive_scan<Binop, ScratchAccessor>::create();
        return retval;
    }

    template<typename T, bool initializeScratch>
    T operator()(T value)
    {
		value = impl(value);

		// store value to smem so we can shuffle it
	#ifdef NBL_GL_KHR_shader_subgroup_shuffle // REVIEW: Should we check this or just use shuffle by default?
		value = ShuffleUp(value, 1);
	#else
		impl.scratchAccessor.set(impl.offsetsAndMasks.scanStoreOffset,value);
		Barrier();
		MemoryBarrierShared();
		// get previous item
		value = impl.scratchAccessor.get(impl.offsetsAndMasks.scanStoreOffset-1u);
		Barrier();
		MemoryBarrierShared();
	#endif
		// return it
		return value;
    }

    template<typename T>
    T operator()(const T value)
    {
		return operator()<T,true>(value);
    }

// protected:
	inclusive_scan<Binop,ScratchAccessor> impl;
};

template<class Binop, class ScratchAccessor>
struct reduction
{
    static reduction<Binop,ScratchAccessor> create()
    {
        reduction<Binop,ScratchAccessor> retval;
        retval.impl = inclusive_scan<Binop,ScratchAccessor>::create();
        return retval;
    }

    template<typename T, bool initializeScratch>
    T operator()(T value)
    {
		value = impl(value);
        uint reductionResultOffset = impl.offsetsAndMasks.paddingMemoryEnd;
		// in case of multiple subgroups inside the WG
		if ((LastWorkgroupInvocation >> SizeLog2()) != InvocationID())
			reductionResultOffset += LastWorkgroupInvocation & impl.offsetsAndMasks.subgroupMask;
		else // in case of single subgroup in WG
			reductionResultOffset += impl.offsetsAndMasks.subgroupMask;

	#ifdef NBL_GL_KHR_shader_subgroup_shuffle
		Shuffle(value, reductionResultOffset);
	#else
		// store value to smem so we can broadcast it to everyone
		impl.scratchAccessor.set(impl.offsetsAndMasks.scanStoreOffset, value);
		Barrier();
		MemoryBarrierShared();

		value = impl.scratchAccessor.get(reductionResultOffset);
		Barrier();
		MemoryBarrierShared();
	#endif
		// return it
		return value;
    }

    template<typename T>
    T operator()(const T value)
    {
        return operator()<T,true>(value);
    }

// protected:
    inclusive_scan<Binop,ScratchAccessor> impl;
};
}

}
}
}

#undef WHOLE_WAVE

#endif