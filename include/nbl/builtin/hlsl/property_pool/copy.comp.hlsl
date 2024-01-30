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
    void iteration(uint propertyId, TransferRequest transferRequest, uint64_t invocationIndex)
    {
        const uint64_t srcIndexSize = uint64_t(1) << SrcIndexSizeLog2;
        const uint64_t dstIndexSize = uint64_t(1) << DstIndexSizeLog2;

        const uint64_t srcOffset = invocationIndex * srcIndexSize * transferRequest.propertySize;
        const uint64_t dstOffset = invocationIndex * dstIndexSize * transferRequest.propertySize;
        
        const uint64_t srcIndexAddress = Fill ? transferRequest.srcIndexAddr + srcOffset : transferRequest.srcIndexAddr;
        const uint64_t dstIndexAddress = Fill ? transferRequest.dstIndexAddr + dstOffset : transferRequest.dstIndexAddr;

        const uint64_t srcAddressBufferOffset = SrcIndexIota ? srcIndexAddress : vk::RawBufferLoad<uint32_t>(srcIndexAddress);
        const uint64_t dstAddressBufferOffset = DstIndexIota ? dstIndexAddress : vk::RawBufferLoad<uint32_t>(dstIndexAddress);

        const uint64_t srcAddressMapped = transferRequest.srcAddr + srcAddressBufferOffset * srcIndexSize; 
        const uint64_t dstAddressMapped = transferRequest.dstAddr + dstAddressBufferOffset * dstIndexSize; 

        if (SrcIndexSizeLog2 == 0) {} // we can't write individual bytes
        else if (SrcIndexSizeLog2 == 1) vk::RawBufferStore<uint16_t>(dstAddressMapped, vk::RawBufferLoad<uint16_t>(srcAddressMapped));
        else if (SrcIndexSizeLog2 == 2) vk::RawBufferStore<uint32_t>(dstAddressMapped, vk::RawBufferLoad<uint32_t>(srcAddressMapped));
        else if (SrcIndexSizeLog2 == 3) vk::RawBufferStore<uint64_t>(dstAddressMapped, vk::RawBufferLoad<uint64_t>(srcAddressMapped));
    }

    void copyLoop(uint baseInvocationIndex, uint propertyId, TransferRequest transferRequest, uint dispatchSize)
    {
        uint64_t elementCount = uint64_t(transferRequest.elementCount32)
            | uint64_t(transferRequest.elementCountExtra) << 32;
        uint64_t lastInvocation = min(elementCount, globals.endOffset);
        for (uint64_t invocationIndex = globals.beginOffset + baseInvocationIndex; invocationIndex < lastInvocation; invocationIndex += dispatchSize)
        {
            iteration(propertyId, transferRequest, invocationIndex);
        }
    }
};

// For creating permutations of the functions based on parameters that are constant over the transfer request
// These branches should all be scalar, and because of how templates are compiled statically, the loops shouldn't have any
// branching within them
// 
// Permutations:
// 2 (fill or not) * 2 (src index iota or not) * 2 (dst index iota or not) * 4 (src index size) * 4 (dst index size)
// Total amount of permutations: 128

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
        bool dstIota = transferRequest.dstIndexAddr == 0;
        if (dstIota) { TransferLoopPermutationDstIota<Fill, SrcIndexIota, true> loop; loop.copyLoop(baseInvocationIndex, propertyId, transferRequest, dispatchSize); }
        else { TransferLoopPermutationDstIota<Fill, SrcIndexIota, false> loop; loop.copyLoop(baseInvocationIndex, propertyId, transferRequest, dispatchSize); }
    }
};

template<bool Fill>
struct TransferLoopPermutationFill
{
    void copyLoop(uint baseInvocationIndex, uint propertyId, TransferRequest transferRequest, uint dispatchSize)
    {
        bool srcIota = transferRequest.srcIndexAddr == 0;
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
    transferRequest.srcAddr = vk::RawBufferLoad<uint>(globals.transferCommandsAddress) | vk::RawBufferLoad<uint>(globals.transferCommandsAddress + sizeof(uint)) << 32;
    transferRequest.dstAddr = vk::RawBufferLoad<uint64_t>(globals.transferCommandsAddress + sizeof(uint64_t));
    transferRequest.srcIndexAddr = vk::RawBufferLoad<uint64_t>(globals.transferCommandsAddress + sizeof(uint64_t) * 2);
    transferRequest.dstIndexAddr = vk::RawBufferLoad<uint64_t>(globals.transferCommandsAddress + sizeof(uint64_t) * 3);
    // Remaining elements are part of the same bitfield
    // TODO: Do this only using raw buffer load?
    uint64_t bitfieldType = vk::RawBufferLoad<uint64_t>(globals.transferCommandsAddress + sizeof(uint64_t) * 4);
    transferRequest.elementCount32 = uint32_t(bitfieldType);
    transferRequest.elementCountExtra = uint32_t(bitfieldType);
    transferRequest.propertySize = uint32_t(bitfieldType >> 3);
    transferRequest.fill = uint32_t(bitfieldType >> (3 + 24));
    transferRequest.srcIndexSizeLog2 = uint32_t(bitfieldType >> (3 + 24 + 1));
    transferRequest.dstIndexSizeLog2 = uint32_t(bitfieldType >> (3 + 24 + 1 + 2));

    const uint dispatchSize = nbl::hlsl::device_capabilities_traits<device_capabilities>::maxOptimallyResidentWorkgroupInvocations;
    const bool fill = transferRequest.fill == 1;

    vk::RawBufferStore<uint64_t>(globals.transferCommandsAddress + 40 * 3, transferRequest.srcAddr);
    vk::RawBufferStore<uint64_t>(globals.transferCommandsAddress + 40 * 4, transferRequest.dstAddr);
    //vk::RawBufferStore<uint>(globals.transferCommandsAddress + 40 * 5, vk::RawBufferLoad<uint>(transferRequest.srcAddr + sizeof(uint16_t) * 3));
    //if (fill) { TransferLoopPermutationFill<true> loop; loop.copyLoop(invocationIndex, propertyId, transferRequest, dispatchSize); }
    //else { TransferLoopPermutationFill<false> loop; loop.copyLoop(invocationIndex, propertyId, transferRequest, dispatchSize); }
}

}
}
}

[numthreads(1,1,1)]
void main(uint32_t3 dispatchId : SV_DispatchThreadID)
{
    nbl::hlsl::property_pools::main<nbl::hlsl::jit::device_capabilities>(dispatchId);
}

