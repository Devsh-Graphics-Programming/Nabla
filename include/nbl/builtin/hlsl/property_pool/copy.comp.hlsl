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

        // Fill: Always use offset 0 on src
        const uint64_t srcOffset = Fill ? 0 : invocationIndex * transferRequest.propertySize;
        const uint64_t dstOffset = invocationIndex * transferRequest.propertySize;
        
        // IOTA: Use the index as the fetching offset
        // Non IOTA: Read the address buffer ("index buffer") to select fetching offset
        const uint64_t srcAddressBufferOffset = SrcIndexIota ? srcOffset : vk::RawBufferLoad<uint32_t>(transferRequest.srcIndexAddr + srcOffset * sizeof(uint32_t));
        const uint64_t dstAddressBufferOffset = DstIndexIota ? dstOffset : vk::RawBufferLoad<uint32_t>(transferRequest.dstIndexAddr + dstOffset * sizeof(uint32_t));

        const uint64_t srcAddressMapped = transferRequest.srcAddr + srcAddressBufferOffset * srcIndexSize; 
        const uint64_t dstAddressMapped = transferRequest.dstAddr + dstAddressBufferOffset * dstIndexSize; 

        //vk::RawBufferStore<uint64_t>(transferRequest.dstAddr + invocationIndex * sizeof(uint64_t) * 2, srcAddressMapped,8);
        //vk::RawBufferStore<uint64_t>(transferRequest.dstAddr + invocationIndex * sizeof(uint64_t) * 2 + sizeof(uint64_t), dstAddressMapped,8);
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
    uint64_t transferCmdAddr = globals.transferCommandsAddress + sizeof(TransferRequest) * propertyId;
    TransferRequest transferRequest;
    transferRequest.srcAddr = vk::RawBufferLoad<uint64_t>(transferCmdAddr,8);
    transferRequest.dstAddr = vk::RawBufferLoad<uint64_t>(transferCmdAddr + sizeof(uint64_t),8);
    transferRequest.srcIndexAddr = vk::RawBufferLoad<uint64_t>(transferCmdAddr + sizeof(uint64_t) * 2,8);
    transferRequest.dstIndexAddr = vk::RawBufferLoad<uint64_t>(transferCmdAddr + sizeof(uint64_t) * 3,8);
    // Remaining elements are part of the same bitfield
    // TODO: Do this only using raw buffer load?
    uint64_t bitfieldType = vk::RawBufferLoad<uint64_t>(transferCmdAddr + sizeof(uint64_t) * 4,8);
    transferRequest.elementCount32 = uint32_t(bitfieldType);
    transferRequest.elementCountExtra = uint32_t(bitfieldType >> 32);
    transferRequest.propertySize = uint32_t(bitfieldType >> (32 + 3));
    transferRequest.fill = uint32_t(bitfieldType >> (32 + 3 + 24));
    transferRequest.srcIndexSizeLog2 = uint32_t(bitfieldType >> (32 + 3 + 24 + 1));
    transferRequest.dstIndexSizeLog2 = uint32_t(bitfieldType >> (32 + 3 + 24 + 1 + 2));

    const uint dispatchSize = nbl::hlsl::device_capabilities_traits<device_capabilities>::maxOptimallyResidentWorkgroupInvocations;
    const bool fill = transferRequest.fill == 1;

    //uint64_t debugWriteAddr = transferRequest.dstAddr + sizeof(uint64_t) * 9 * propertyId;
    //vk::RawBufferStore<uint64_t>(debugWriteAddr + sizeof(uint64_t) * 0, transferRequest.srcAddr,8);
    //vk::RawBufferStore<uint64_t>(debugWriteAddr + sizeof(uint64_t) * 1, transferRequest.dstAddr,8);
    //vk::RawBufferStore<uint64_t>(debugWriteAddr + sizeof(uint64_t) * 2, transferRequest.srcIndexAddr,8);
    //vk::RawBufferStore<uint64_t>(debugWriteAddr + sizeof(uint64_t) * 3, transferRequest.dstIndexAddr,8);
    //uint64_t elementCount = uint64_t(transferRequest.elementCount32)
    //    | uint64_t(transferRequest.elementCountExtra) << 32;
    //vk::RawBufferStore<uint64_t>(debugWriteAddr + sizeof(uint64_t) * 4, elementCount,8);
    //vk::RawBufferStore<uint32_t>(debugWriteAddr + sizeof(uint64_t) * 5, transferRequest.propertySize,4);
    //vk::RawBufferStore<uint32_t>(debugWriteAddr + sizeof(uint64_t) * 6, transferRequest.fill,4);
    //vk::RawBufferStore<uint32_t>(debugWriteAddr + sizeof(uint64_t) * 7, transferRequest.srcIndexSizeLog2,4);
    //vk::RawBufferStore<uint32_t>(debugWriteAddr + sizeof(uint64_t) * 8, transferRequest.dstIndexSizeLog2,4);
    //vk::RawBufferStore<uint64_t>(transferRequest.dstAddr + sizeof(uint64_t) * invocationIndex, invocationIndex,8);
    
    if (fill) { TransferLoopPermutationFill<true> loop; loop.copyLoop(invocationIndex, propertyId, transferRequest, dispatchSize); }
    else { TransferLoopPermutationFill<false> loop; loop.copyLoop(invocationIndex, propertyId, transferRequest, dispatchSize); }
}

}
}
}

// TODO: instead use some sort of replace function for getting optimal size?
[numthreads(512,1,1)]
void main(uint32_t3 dispatchId : SV_DispatchThreadID)
{
    nbl::hlsl::property_pools::main<nbl::hlsl::jit::device_capabilities>(dispatchId);
}

