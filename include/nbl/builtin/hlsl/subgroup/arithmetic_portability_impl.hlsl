#ifndef _NBL_BUILTIN_HLSL_SUBGROUP_ARITHMETIC_PORTABILITY_IMPL_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP_ARITHMETIC_PORTABILITY_IMPL_INCLUDED_

uint localInvocationIndex : SV_GroupIndex; // REVIEW: Discuss proper placement of SV_* values. They are not allowed to be defined inside a function scope, only as arguments of global variables in the shader.

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
template<>
struct reduction<binops::bitwise_and>
{
    template<typename T>
    T operator()(const T x)
    {
        return WaveActiveBitAnd(x);
    }
};
template<>
struct exclusive_scan<binops::bitwise_and>
{
    template<typename T>
    T operator()(const T x)
    {
        return WaveMultiPrefixAnd(x, WHOLE_WAVE);
    }
};
template<>
struct inclusive_scan<binops::bitwise_and>
{
    template<typename T>
    T operator()(const T x)
    {
        return WaveMultiPrefixAnd(x, WHOLE_WAVE) & x;
    }
};

// *** OR ***
template<>
struct reduction<binops::bitwise_or>
{
    template<typename T>
    T operator()(const T x)
    {
        return WaveActiveBitOr(x);
    }
};
template<>
struct exclusive_scan<binops::bitwise_or>
{
    template<typename T>
    T operator()(const T x)
    {
        return WaveMultiPrefixOr(x, WHOLE_WAVE);
    }
};
template<>
struct inclusive_scan<binops::bitwise_or>
{
    template<typename T>
    T operator()(const T x)
    {
        return WaveMultiPrefixOr(x, WHOLE_WAVE) | x;
    }
};

// *** XOR ***
template<>
struct reduction<binops::bitwise_xor>
{
    template<typename T>
    T operator()(const T x)
    {
        return WaveActiveBitXor(x);
    }
};
template<>
struct exclusive_scan<binops::bitwise_xor>
{
    template<typename T>
    T operator()(const T x)
    {
        return WaveMultiPrefixXor(x, WHOLE_WAVE);
    }
};
template<>
struct inclusive_scan<binops::bitwise_xor>
{
    template<typename T>
    T operator()(const T x)
    {
        return WaveMultiPrefixXor(x, WHOLE_WAVE) ^ x;
    }
};

// *** ADD ***
template<>
struct reduction<binops::bitwise_add>
{
    template<typename T>
    T operator()(const T x)
    {
        return WaveActiveSum(x);
    }
};
template<>
struct exclusive_scan<binops::bitwise_add>
{
    template<typename T>
    T operator()(const T x)
    {
        return WaveMultiPrefixSum(x, WHOLE_WAVE);
    }
};
template<>
struct inclusive_scan<binops::bitwise_add>
{
    template<typename T>
    T operator()(const T x)
    {
        return WaveMultiPrefixSum(x, WHOLE_WAVE) + x;
    }
};

// *** MUL ***
template<>
struct reduction<binops::bitwise_mul>
{
    template<typename T>
    T operator()(const T x)
    {
        return WaveActiveProduct(x);
    }
};
template<>
struct exclusive_scan<binops::bitwise_mul>
{
    template<typename T>
    T operator()(const T x)
    {
        return WaveMultiPrefixProduct(x, WHOLE_WAVE);
    }
};
template<>
struct inclusive_scan<binops::bitwise_mul>
{
    template<typename T>
    T operator()(const T x)
    {
        return WaveMultiPrefixProduct(x, WHOLE_WAVE) * x;
    }
};

// *** MIN ***
template<>
struct reduction<binops::bitwise_min>
{
    template<typename T>
    T operator()(const T x)
    {
        return WaveActiveMin(x);
    }
};

// TODO (PentaKon): There is no matching intrinsic to GLSL's subgroupEx/InclusiveMin

// *** MAX ***
template<>
struct reduction<binops::bitwise_max>
{
    template<typename T>
    T operator()(const T x)
    {
        return WaveActiveMax(x);
    }
};

// TODO (PentaKon): There is no matching intrinsic to GLSL's subgroupEx/InclusiveMax

}
#endif

namespace portability
{

template<class NumberScratchAccessor>
struct ScratchAccessorAdaptor {
	NumberScratchAccessor accessor;
	
	void get(const uint ix, out uint value) { value = accessor.get(ix);}
    void get(const uint ix, out uint2 value) { value = uint2(accessor.get(ix), accessor.get(ix + _NBL_HLSL_WORKGROUP_SIZE_));}
    void get(const uint ix, out uint3 value) { value = uint3(accessor.get(ix), accessor.get(ix + _NBL_HLSL_WORKGROUP_SIZE_), accessor.get(ix + 2 * _NBL_HLSL_WORKGROUP_SIZE_));}
    void get(const uint ix, out uint4 value) { value = uint4(accessor.get(ix), accessor.get(ix + _NBL_HLSL_WORKGROUP_SIZE_), accessor.get(ix + 2 * _NBL_HLSL_WORKGROUP_SIZE_), accessor.get(ix + 3 * _NBL_HLSL_WORKGROUP_SIZE_));}

    void get(const uint ix, out int value) { value = asint(accessor.get(ix));}
    void get(const uint ix, out int2 value) { value = asint(uint2(accessor.get(ix), accessor.get(ix + _NBL_HLSL_WORKGROUP_SIZE_)));}
    void get(const uint ix, out int3 value) { value = asint(uint3(accessor.get(ix), accessor.get(ix + _NBL_HLSL_WORKGROUP_SIZE_), accessor.get(ix + 2 * _NBL_HLSL_WORKGROUP_SIZE_)));}
    void get(const uint ix, out int4 value) { value = asint(uint4(accessor.get(ix), accessor.get(ix + _NBL_HLSL_WORKGROUP_SIZE_), accessor.get(ix + 2 * _NBL_HLSL_WORKGROUP_SIZE_), accessor.get(ix + 3 * _NBL_HLSL_WORKGROUP_SIZE_)));}
    
    void get(const uint ix, out float value) { value = asfloat(accessor.get(ix));}
    void get(const uint ix, out float2 value) { value = asfloat(uint2(accessor.get(ix), accessor.get(ix + _NBL_HLSL_WORKGROUP_SIZE_)));}
    void get(const uint ix, out float3 value) { value = asfloat(uint3(accessor.get(ix), accessor.get(ix + _NBL_HLSL_WORKGROUP_SIZE_), accessor.get(ix + 2 * _NBL_HLSL_WORKGROUP_SIZE_)));}
    void get(const uint ix, out float4 value) { value = asfloat(uint4(accessor.get(ix), accessor.get(ix + _NBL_HLSL_WORKGROUP_SIZE_), accessor.get(ix + 2 * _NBL_HLSL_WORKGROUP_SIZE_), accessor.get(ix + 3 * _NBL_HLSL_WORKGROUP_SIZE_)));}

	void set(const uint ix, const uint value) {accessor.set(ix, value);}
    void set(const uint ix, const uint2 value) {
        accessor.set(ix, value.x);
        accessor.set(ix + _NBL_HLSL_WORKGROUP_SIZE_, value.y);
    }
    void set(const uint ix, const uint3 value)  {
        accessor.set(ix, value.x);
        accessor.set(ix + _NBL_HLSL_WORKGROUP_SIZE_, value.y);
        accessor.set(ix + 2 * _NBL_HLSL_WORKGROUP_SIZE_, value.z);
    }
    void set(const uint ix, const uint4 value) {
        accessor.set(ix, value.x);
        accessor.set(ix + _NBL_HLSL_WORKGROUP_SIZE_, value.y);
        accessor.set(ix + 2 * _NBL_HLSL_WORKGROUP_SIZE_, value.z);
        accessor.set(ix + 3 * _NBL_HLSL_WORKGROUP_SIZE_, value.w);
    }

    void set(const uint ix, const int value) {accessor.set(ix, asuint(value));}
    void set(const uint ix, const int2 value) {
        accessor.set(ix, asuint(value.x));
        accessor.set(ix + _NBL_HLSL_WORKGROUP_SIZE_, asuint(value.y));
    }
    void set(const uint ix, const int3 value)  {
        accessor.set(ix, asuint(value.x));
        accessor.set(ix + _NBL_HLSL_WORKGROUP_SIZE_, asuint(value.y));
        accessor.set(ix + 2 * _NBL_HLSL_WORKGROUP_SIZE_, asuint(value.z));
    }
    void set(const uint ix, const int4 value) {
        accessor.set(ix, asuint(value.x));
        accessor.set(ix + _NBL_HLSL_WORKGROUP_SIZE_, asuint(value.y));
        accessor.set(ix + 2 * _NBL_HLSL_WORKGROUP_SIZE_, asuint(value.z));
        accessor.set(ix + 3 * _NBL_HLSL_WORKGROUP_SIZE_, asuint(value.w));
    }

    void set(const uint ix, const float value) {accessor.set(ix, asuint(value));}
    void set(const uint ix, const float2 value) {
        accessor.set(ix, asuint(value.x));
        accessor.set(ix + _NBL_HLSL_WORKGROUP_SIZE_, asuint(value.y));
    }
    void set(const uint ix, const float3 value)  {
        accessor.set(ix, asuint(value.x));
        accessor.set(ix + _NBL_HLSL_WORKGROUP_SIZE_, asuint(value.y));
        accessor.set(ix + 2 * _NBL_HLSL_WORKGROUP_SIZE_, asuint(value.z));
    }
    void set(const uint ix, const float4 value) {
        accessor.set(ix, asuint(value.x));
        accessor.set(ix + _NBL_HLSL_WORKGROUP_SIZE_, asuint(value.y));
        accessor.set(ix + 2 * _NBL_HLSL_WORKGROUP_SIZE_, asuint(value.z));
        accessor.set(ix + 3 * _NBL_HLSL_WORKGROUP_SIZE_, asuint(value.w));
    }
};

struct scan_base
{
   // even if you have a `const uint nbl::hlsl::subgroup::Size` it wont work I think, so `#define` needed
   static const uint SubgroupSize = nbl::hlsl::subgroup::subgroupSize();
   static const uint HalfSubgroupSize = SubgroupSize>>1u; // REVIEW: Is this ok?
   static const uint LoMask = SubgroupSize-1u;
   static const uint LastWorkgroupInvocation = _NBL_HLSL_WORKGROUP_SIZE_-1; // REVIEW: Where should this be defined?
   static const uint pseudoSubgroupInvocation = localInvocationIndex&LoMask; // Also used in substructs, thus static const
   
    static inclusive_scan<Binop,ScratchAccessor> create()
    {
       const uint pseudoSubgroupElectedInvocation = localInvocationIndex&(~LoMask);
    
       inclusive_scan<Binop,ScratchAccessor> retval;
       
       const uint subgroupMemoryBegin = pseudoSubgroupElectedInvocation<<1u;
       retval.lastLoadOffset = subgroupMemoryBegin+pseudoSubgroupInvocation;
       
       const uint paddingMemoryEnd = subgroupMemoryBegin+HalfSubgroupSize;
       retval.scanStoreOffset = paddingMemoryEnd+pseudoSubgroupInvocation;
       
		uint reductionResultOffset = paddingMemoryEnd;
		if ((LastWorkgroupInvocation>>nbl::hlsl::subgroup::subgroupSizeLog2())!=nbl::hlsl::subgroup::subgroupInvocationID())
           retval.reductionResultOffset += LastWorkgroupInvocation&LoMask;
		else
           retval.reductionResultOffset += LoMask;

		retval.paddingMemoryEnd = reductionResultOffset;

       return retval;
    }
   
// protected:   
   uint paddingMemoryEnd;
   uint scanStoreOffset;
   uint lastLoadOffset;
};

template<class Binop, class ScratchAccessor>
struct inclusive_scan : scan_base
{
    static inclusive_scan<Binop,ScratchAccessor> create()
    {    
       return scan_base<Binop,ScratchAccessor>::create(); // REVIEW: Is this correct?
    }

    template<typename T, bool initializeScratch>
    T operator()(T value)
    {
       ScratchAccessor scratchAccessor;
       Binop op;
       
       if (initializeScratch)
       {
           nbl::hlsl::subgroupBarrier();
           nbl::hlsl::subgroupMemoryBarrierShared();
           scratchAccessor.set(scanStoreOffset ,value);
           if (scan_base::pseudoSubgroupInvocation<scan_base::HalfSubgroupSize)
               scratchAccessor.set(lastLoadOffset,op::identity());
       }
       nbl::hlsl::subgroupBarrier();
       nbl::hlsl::subgroupMemoryBarrierShared();
       // Stone-Kogge adder
       // (devsh): it seems that lanes below <HalfSubgroupSize/step are doing useless work,
       // but they're SIMD and adding an `if`/conditional execution is more expensive
       value = op(value,scratchAccessor.get(scanStoreOffset-1u));
       [[unroll]]
       for (uint stp=2u; stp<=scan_base::HalfSubgroupSize; stp<<=1u)
       {
           scratchAccessor.set(scanStoreOffset,value);
           nbl::hlsl::subgroupBarrier();
           nbl::hlsl::subgroupMemoryBarrierShared();
           value = op(value,scratchAccessor.get(scanStoreOffset-stp));
           nbl::hlsl::subgroupBarrier();
           nbl::hlsl::subgroupMemoryBarrierShared();
       }
       return value;
    }
    
    template<typename T>
    T operator()(const T value)
    {
        return operator()<T,true>(value);
    }
};

template<class Binop, class ScratchAccessor>
struct exclusive_scan
{
    static exclusive_scan<Binop,ScratchAccessor> create()
    {
        exclusive_scan<Binop,ScratchAccessor> retval;
        retval.impl = inclusive_scan<Binop,ScratchAccessor>::create();
        return retval;
    }
    
    template<typename T, bool initializeScratch>
    T operator()(T value)
    {
       value = impl.operator()<T,initializeScratch>(value);

       // store value to smem so we can shuffle it
       scratchAccessor.set(impl.scanStoreOffset,value);
       nbl::hlsl::subgroupBarrier();
       nbl::hlsl::subgroupMemoryBarrierShared();
       // get previous item
       value = scratchAccessor.get(impl.scanStoreOffset-1u);
       nbl::hlsl::subgroupBarrier();
       nbl::hlsl::subgroupMemoryBarrierShared();

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
       value = impl.operator()<T,initializeScratch>(value);

       // store value to smem so we can broadcast it to everyone
       scratchAccessor.set(impl.scanStoreOffset,value);
       nbl::hlsl::subgroupBarrier();
       nbl::hlsl::subgroupMemoryBarrierShared();
       uint reductionResultOffset = impl.paddingMemoryEnd;
       if ((scan_base::LastWorkgroupInvocation>>nbl::hlsl::subgroup::subgroupSizeLog2())!=nbl::hlsl::subgroup::ID())
           reductionResultOffset += scan_base::LastWorkgroupInvocation & scan_base::LoMask;
       else
           reductionResultOffset += scan_base::LoMask;
       value = scratchAccessor.get(reductionResultOffset);
       nbl::hlsl::subgroupBarrier();
       nbl::hlsl::subgroupMemoryBarrierShared();

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

#endif