#include "nbl/builtin/hlsl/jit/device_capabilities.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/property_pool/transfer.hlsl"

namespace nbl
{
namespace hlsl
{
namespace property_pools
{

[[vk::push_constant]] GlobalPushContants globals;

template<bool Fill, bool SrcIndexIota, bool DstIndexIota, uint64_t SrcIndexSizeLog2, uint64_t DstIndexSizeLog2>
struct TransferLoop
{
    void iteration(uint propertyId, uint64_t propertySize, uint64_t srcAddr, uint64_t dstAddr, uint invocationIndex)
    {
        const uint64_t srcOffset = uint64_t(invocationIndex) * (uint64_t(1) << SrcIndexSizeLog2) * propertySize;
        const uint64_t dstOffset = uint64_t(invocationIndex) * (uint64_t(1) << DstIndexSizeLog2) * propertySize;
        
        const uint64_t srcIndexAddress = Fill ? srcAddr + srcOffset : srcAddr;
        const uint64_t dstIndexAddress = Fill ? dstAddr + dstOffset : dstAddr;

        const uint64_t srcAddressMapped = SrcIndexIota ? srcIndexAddress : vk::RawBufferLoad<uint64_t>(srcIndexAddress); 
        const uint64_t dstAddressMapped = DstIndexIota ? dstIndexAddress : vk::RawBufferLoad<uint64_t>(dstIndexAddress); 

        if (SrcIndexSizeLog2 == 0) {} // we can't write individual bytes
        else if (SrcIndexSizeLog2 == 1) vk::RawBufferStore<uint16_t>(dstAddressMapped, vk::RawBufferLoad<uint16_t>(srcAddressMapped));
        else if (SrcIndexSizeLog2 == 2) vk::RawBufferStore<uint32_t>(dstAddressMapped, vk::RawBufferLoad<uint32_t>(srcAddressMapped));
        else if (SrcIndexSizeLog2 == 3) vk::RawBufferStore<uint64_t>(dstAddressMapped, vk::RawBufferLoad<uint64_t>(srcAddressMapped));
    }

    void copyLoop(uint baseInvocationIndex, uint propertyId, TransferRequest transferRequest, uint dispatchSize)
    {
        uint64_t elementCount = uint64_t(transferRequest.elementCount32)
            | uint64_t(transferRequest.elementCountExtra) << 32;
        uint lastInvocation = min(elementCount, globals.endOffset);
        for (uint invocationIndex = globals.beginOffset + baseInvocationIndex; invocationIndex < lastInvocation; invocationIndex += dispatchSize)
        {
            iteration(propertyId, transferRequest.propertySize, transferRequest.srcAddr, transferRequest.dstAddr, invocationIndex);
        }
    }
};

// For creating permutations of the functions based on parameters that are constant over the transfer request
// These branches should all be scalar, and because of how templates are compiled statically, the loops shouldn't have any
// branching within them

template<bool Fill, bool SrcIndexIota, bool DstIndexIota, uint64_t SrcIndexSizeLog2>
struct TransferLoopPermutationSrcIndexSizeLog
{
    void copyLoop(uint baseInvocationIndex, uint propertyId, TransferRequest transferRequest, uint dispatchSize)
    {
       if (transferRequest.dstIndexSizeLog2 == 0) { TransferLoop<Fill, SrcIndexIota, DstIndexIota, SrcIndexSizeLog2, 0> loop; loop.copyLoop(baseInvocationIndex, propertyId, transferRequest, dispatchSize); }
       else if (transferRequest.dstIndexSizeLog2 == 1) { TransferLoop<Fill, SrcIndexIota, DstIndexIota, SrcIndexSizeLog2, 1> loop; loop.copyLoop(baseInvocationIndex, propertyId, transferRequest, dispatchSize); }
       else if (transferRequest.dstIndexSizeLog2 == 2) { TransferLoop<Fill, SrcIndexIota, DstIndexIota, SrcIndexSizeLog2, 2> loop; loop.copyLoop(baseInvocationIndex, propertyId, transferRequest, dispatchSize); }
       else /*if (transferRequest.dstIndexSizeLog2 == 3)*/ { TransferLoop<Fill, SrcIndexIota, DstIndexIota, SrcIndexSizeLog2, 3> loop; loop.copyLoop(baseInvocationIndex, propertyId, transferRequest, dispatchSize); }
    }
};

template<bool Fill, bool SrcIndexIota, bool DstIndexIota>
struct TransferLoopPermutationDstIota
{
    void copyLoop(uint baseInvocationIndex, uint propertyId, TransferRequest transferRequest, uint dispatchSize)
    {
       if (transferRequest.srcIndexSizeLog2 == 0) { TransferLoopPermutationSrcIndexSizeLog<Fill, SrcIndexIota, DstIndexIota, 0> loop; loop.copyLoop(baseInvocationIndex, propertyId, transferRequest, dispatchSize); }
       else if (transferRequest.srcIndexSizeLog2 == 1) { TransferLoopPermutationSrcIndexSizeLog<Fill, SrcIndexIota, DstIndexIota, 1> loop; loop.copyLoop(baseInvocationIndex, propertyId, transferRequest, dispatchSize); }
       else if (transferRequest.srcIndexSizeLog2 == 2) { TransferLoopPermutationSrcIndexSizeLog<Fill, SrcIndexIota, DstIndexIota, 2> loop; loop.copyLoop(baseInvocationIndex, propertyId, transferRequest, dispatchSize); }
       else /*if (transferRequest.srcIndexSizeLog2 == 3)*/ { TransferLoopPermutationSrcIndexSizeLog<Fill, SrcIndexIota, DstIndexIota, 3> loop; loop.copyLoop(baseInvocationIndex, propertyId, transferRequest, dispatchSize); }
    }
};

template<bool Fill, bool SrcIndexIota>
struct TransferLoopPermutationSrcIota
{
    void copyLoop(uint baseInvocationIndex, uint propertyId, TransferRequest transferRequest, uint dispatchSize)
    {
        bool dstIota = transferRequest.dstAddr == 0;
        if (dstIota) { TransferLoopPermutationDstIota<Fill, SrcIndexIota, true> loop; loop.copyLoop(baseInvocationIndex, propertyId, transferRequest, dispatchSize); }
        else { TransferLoopPermutationDstIota<Fill, SrcIndexIota, false> loop; loop.copyLoop(baseInvocationIndex, propertyId, transferRequest, dispatchSize); }
    }
};

template<bool Fill>
struct TransferLoopPermutationFill
{
    void copyLoop(uint baseInvocationIndex, uint propertyId, TransferRequest transferRequest, uint dispatchSize)
    {
        bool srcIota = transferRequest.srcAddr == 0;
        if (srcIota) { TransferLoopPermutationSrcIota<Fill, true> loop; loop.copyLoop(baseInvocationIndex, propertyId, transferRequest, dispatchSize); }
        else { TransferLoopPermutationSrcIota<Fill, false> loop; loop.copyLoop(baseInvocationIndex, propertyId, transferRequest, dispatchSize); }
    }
};

template<typename device_capabilities>
void main(uint32_t3 dispatchId)
{
    const uint propertyId = dispatchId.y;
    const uint invocationIndex = dispatchId.x;

    // Loading transfer request from the pointer (can't use struct
    // with BDA on HLSL SPIRV)
    TransferRequest transferRequest;
    transferRequest.srcAddr = vk::RawBufferLoad<uint64_t>(globals.transferCommandsAddress);
    transferRequest.dstAddr = vk::RawBufferLoad<uint64_t>(globals.transferCommandsAddress + sizeof(uint64_t));
    transferRequest.srcIndexAddr = vk::RawBufferLoad<uint64_t>(globals.transferCommandsAddress + sizeof(uint64_t) * 2);
    transferRequest.dstIndexAddr = vk::RawBufferLoad<uint64_t>(globals.transferCommandsAddress + sizeof(uint64_t) * 3);
    // Remaining elements are part of the same bitfield
    // TODO: Do this only using raw buffer load?
    uint2 bitfieldType = vk::RawBufferLoad<uint2>(globals.transferCommandsAddress + sizeof(uint64_t) * 4);
    transferRequest.elementCount32 = bitfieldType;
    transferRequest.elementCountExtra = bitfieldType;
    transferRequest.propertySize = bitfieldType >> 3;
    transferRequest.fill = bitfieldType >> (3 + 24);
    transferRequest.srcIndexSizeLog2 = bitfieldType >> (3 + 24 + 1);
    transferRequest.dstIndexSizeLog2 = bitfieldType >> (3 + 24 + 1 + 2);

    const uint dispatchSize = nbl::hlsl::device_capabilities_traits<device_capabilities>::maxOptimallyResidentWorkgroupInvocations;
    const bool fill = transferRequest.fill == 1;

    if (fill) { TransferLoopPermutationFill<true> loop; loop.copyLoop(invocationIndex, propertyId, transferRequest, dispatchSize); }
    else { TransferLoopPermutationFill<false> loop; loop.copyLoop(invocationIndex, propertyId, transferRequest, dispatchSize); }
}

}
}
}

[numthreads(1,1,1)]
void main(uint32_t3 dispatchId : SV_DispatchThreadID)
{
    nbl::hlsl::property_pools::main<nbl::hlsl::jit::device_capabilities>(dispatchId);
}

