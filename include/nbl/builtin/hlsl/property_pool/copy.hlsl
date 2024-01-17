#include "nbl/builtin/hlsl/jit/device_capabilities.hlsl"
#include "nbl/builtin/hlsl/property_pool/transfer.hlsl"

namespace nbl
{
namespace hlsl
{
namespace property_pools
{
// https://github.com/microsoft/DirectXShaderCompiler/issues/6144
template<typename capability_traits=nbl::hlsl::jit::device_capabilities_traits>
uint32_t3 nbl::hlsl::glsl::gl_WorkGroupSize() {
    return uint32_t3(capability_traits::maxOptimallyResidentWorkgroupInvocations, 1, 1);
}

[[vk::push_constant]] GlobalPushContants globals;

template<bool Fill, bool SrcIndexIota, bool DstIndexIota, uint64_t SrcIndexSizeLog2, uint64_t DstIndexSizeLog2>
struct TransferLoop
{
    void iteration(uint propertyId, uint64_t propertySize, uint64_t srcAddr, uint64_t dstAddr, uint invocationIndex)
    {
        const uint srcOffset = uint64_t(invocationIndex) * (uint64_t(1) << SrcIndexSizeLog2) * propertySize;
        const uint dstOffset = uint64_t(invocationIndex) * (uint64_t(1) << DstIndexSizeLog2) * propertySize;
        
        const uint srcIndexAddress = Fill ? srcAddr + srcOffset : srcAddr;
        const uint dstIndexAddress = Fill ? dstAddr + dstOffset : dstAddr;

        const uint srcAddressMapped = SrcIndexIota ? srcIndexAddress : vk::RawBufferLoad<uint64_t>(srcIndexAddress); 
        const uint dstAddressMapped = DstIndexIota ? dstIndexAddress : vk::RawBufferLoad<uint64_t>(dstIndexAddress); 

        if (SrcIndexSizeLog2 == 0) {} // we can't write individual bytes
        else if (SrcIndexSizeLog2 == 1) vk::RawBufferStore<uint16_t>(dstAddressMapped, vk::RawBufferLoad<uint16_t>(srcAddressMapped));
        else if (SrcIndexSizeLog2 == 2) vk::RawBufferStore<uint32_t>(dstAddressMapped, vk::RawBufferLoad<uint32_t>(srcAddressMapped));
        else if (SrcIndexSizeLog2 == 3) vk::RawBufferStore<uint64_t>(dstAddressMapped, vk::RawBufferLoad<uint64_t>(srcAddressMapped));
    }

    void copyLoop(uint baseInvocationIndex, uint propertyId, TransferRequest transferRequest, uint dispatchSize)
    {
        uint lastInvocation = min(transferRequest.elementCount, gloabls.endOffset);
        for (uint invocationIndex = globals.beginOffset + baseInvocationIndex; invocationIndex < lastInvocation; invocationIndex += dispatchSize)
        {
            iteration(propertyId, transferRequest.propertySize, transferRequest.srcAddr, transferRequest.dstAddr, invocationIndex);
        }
    }
}

// For creating permutations of the functions based on parameters that are constant over the transfer request
// These branches should all be scalar, and because of how templates work, the loops shouldn't have any
// branching within them

template<bool Fill, bool SrcIndexIota, bool DstIndexIota, uint64_t SrcIndexSizeLog2>
struct TransferLoopPermutationSrcIndexSizeLog
{
    void copyLoop(uint baseInvocationIndex, uint propertyId, TransferRequest transferRequest, uint dispatchSize)
    {
       if (transferRequest.dstIndexSizeLog2 == 0) TransferLoop<Fill, SrcIndexIota, DstIndexIota, SrcIndexSizeLog2, 0>.copyLoop(baseInvocationIndex, propertyId, transferRequest, dispatchSize);
       else if (transferRequest.dstIndexSizeLog2 == 1) TransferLoop<Fill, SrcIndexIota, DstIndexIota, SrcIndexSizeLog2, 1>.copyLoop(baseInvocationIndex, propertyId, transferRequest, dispatchSize);
       else if (transferRequest.dstIndexSizeLog2 == 2) TransferLoop<Fill, SrcIndexIota, DstIndexIota, SrcIndexSizeLog2, 2>.copyLoop(baseInvocationIndex, propertyId, transferRequest, dispatchSize);
       else /*if (transferRequest.dstIndexSizeLog2 == 3)*/ TransferLoop<Fill, SrcIndexIota, DstIndexIota, SrcIndexSizeLog2, 3>.copyLoop(baseInvocationIndex, propertyId, transferRequest, dispatchSize);
    }
}

template<bool Fill, bool SrcIndexIota, bool DstIndexIota>
struct TransferLoopPermutationDstIota
{
    void copyLoop(uint baseInvocationIndex, uint propertyId, TransferRequest transferRequest, uint dispatchSize)
    {
       if (transferRequest.srcIndexSizeLog2 == 0) TransferLoopPermutationSrcIndexSizeLog<Fill, SrcIndexIota, DstIndexIota, 0>.copyLoop(baseInvocationIndex, propertyId, transferRequest, dispatchSize);
       else if (transferRequest.srcIndexSizeLog2 == 1) TransferLoopPermutationSrcIndexSizeLog<Fill, SrcIndexIota, DstIndexIota, 1>.copyLoop(baseInvocationIndex, propertyId, transferRequest, dispatchSize);
       else if (transferRequest.srcIndexSizeLog2 == 2) TransferLoopPermutationSrcIndexSizeLog<Fill, SrcIndexIota, DstIndexIota, 2>.copyLoop(baseInvocationIndex, propertyId, transferRequest, dispatchSize);
       else /*if (transferRequest.srcIndexSizeLog2 == 3)*/ TransferLoopPermutationSrcIndexSizeLog<Fill, SrcIndexIota, DstIndexIota, 3>.copyLoop(baseInvocationIndex, propertyId, transferRequest, dispatchSize);
    }
}

template<bool Fill, bool SrcIndexIota>
struct TransferLoopPermutationSrcIota
{
    void copyLoop(uint baseInvocationIndex, uint propertyId, TransferRequest transferRequest, uint dispatchSize)
    {
        bool dstIota = transferRequest.dstAddr == 0;
        if (dstIota) TransferLoopPermutationDstIota<Fill, SrcIndexIota, true>.copyLoop(baseInvocationIndex, propertyId, transferRequest, dispatchSize);
        else TransferLoopPermutationDstIota<Fill, SrcIndexIota, false>.copyLoop(baseInvocationIndex, propertyId, transferRequest, dispatchSize);
    }
}

template<bool Fill>
struct TransferLoopPermutationFill
{
    void copyLoop(uint baseInvocationIndex, uint propertyId, TransferRequest transferRequest, uint dispatchSize)
    {
        bool srcIota = transferRequest.srcAddr == 0;
        if (srcIota) TransferLoopPermutationSrcIota<Fill, true>.copyLoop(baseInvocationIndex, propertyId, transferRequest, dispatchSize);
        else TransferLoopPermutationSrcIota<Fill, false>.copyLoop(baseInvocationIndex, propertyId, transferRequest, dispatchSize);
    }
}

void main(uint32_t3 dispatchId : SV_DispatchThreadID)
{
    const uint propertyId = dispatchId.y;
    const uint invocationIndex = dispatchId.x;

    // Loading transfer request from the pointer (can't use struct
    // with BDA on HLSL SPIRV)
    const TransferRequest transferRequest;
    transferRequest.srcAddr = vk::RawBufferLoad<uint64_t>(globals.transferCommandsAddress);
    transferRequest.dstAddr = vk::RawBufferLoad<uint64_t>(globals.transferCommandsAddress + sizeof(uint64_t));
    transferRequest.srcIndexAddr = vk::RawBufferLoad<uint64_t>(globals.transferCommandsAddress + sizeof(uint64_t) * 2);
    transferRequest.dstIndexAddr = vk::RawBufferLoad<uint64_t>(globals.transferCommandsAddress + sizeof(uint64_t) * 3);
    // TODO: These are all part of the same bitfield and shoulbe read with a single RawBufferLoad
    transferRequest.elementCount = vk::RawBufferLoad<uint64_t>(globals.transferCommandsAddress + sizeof(uint64_t) * 4);
    transferRequest.propertySize = vk::RawBufferLoad<uint64_t>(globals.transferCommandsAddress + sizeof(uint64_t) * 5);
    transferRequest.fill = vk::RawBufferLoad<uint64_t>(globals.transferCommandsAddress + sizeof(uint64_t) * 6);
    transferRequest.srcIndexSizeLog2 = vk::RawBufferLoad<uint64_t>(globals.transferCommandsAddress + sizeof(uint64_t) * 7);
    transferRequest.dstIndexSizeLog2 = vk::RawBufferLoad<uint64_t>(globals.transferCommandsAddress + sizeof(uint64_t) * 8);

    const uint dispatchSize = capability_traits::maxOptimallyResidentWorkgroupInvocations;
    const bool fill = transferRequest.fill == 1;

    if (fill) TransferLoopPermutationFill<true>.copyLoop(invocationIndex, propertyId, transferRequest, dispatchSize);
    else TransferLoopPermutationFill<false>.copyLoop(invocationIndex, propertyId, transferRequest, dispatchSize);
}

}
}
}

