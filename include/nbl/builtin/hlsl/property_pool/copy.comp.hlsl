#include "nbl/builtin/hlsl/jit/device_capabilities.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/property_pool/transfer.hlsl"

namespace nbl
{
namespace hlsl
{
namespace property_pools
{

[[vk::push_constant]] TransferDispatchInfo globals;

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
        uint64_t srcAddressBufferOffset;
        uint64_t dstAddressBufferOffset;

        if (SrcIndexIota) srcAddressBufferOffset = srcOffset;
        else 
        {
            if (SrcIndexSizeLog2 == 0) {} // we can't read individual byte
            else if (SrcIndexSizeLog2 == 1) srcAddressBufferOffset = vk::RawBufferLoad<uint16_t>(transferRequest.srcIndexAddr + srcOffset * sizeof(uint16_t));
            else if (SrcIndexSizeLog2 == 2) srcAddressBufferOffset = vk::RawBufferLoad<uint32_t>(transferRequest.srcIndexAddr + srcOffset * sizeof(uint32_t));
            else if (SrcIndexSizeLog2 == 3) srcAddressBufferOffset = vk::RawBufferLoad<uint64_t>(transferRequest.srcIndexAddr + srcOffset * sizeof(uint64_t));
        }

        if (DstIndexIota) dstAddressBufferOffset = dstOffset;
        else 
        {
            if (DstIndexSizeLog2 == 0) {} // we can't read individual byte
            else if (DstIndexSizeLog2 == 1) dstAddressBufferOffset = vk::RawBufferLoad<uint16_t>(transferRequest.dstIndexAddr + dstOffset * sizeof(uint16_t));
            else if (DstIndexSizeLog2 == 2) dstAddressBufferOffset = vk::RawBufferLoad<uint32_t>(transferRequest.dstIndexAddr + dstOffset * sizeof(uint32_t));
            else if (DstIndexSizeLog2 == 3) dstAddressBufferOffset = vk::RawBufferLoad<uint64_t>(transferRequest.dstIndexAddr + dstOffset * sizeof(uint64_t));
        }

        const uint64_t srcAddressMapped = transferRequest.srcAddr + srcAddressBufferOffset * srcIndexSize; 
        const uint64_t dstAddressMapped = transferRequest.dstAddr + dstAddressBufferOffset * dstIndexSize; 

        vk::RawBufferStore<uint32_t>(dstAddressMapped, vk::RawBufferLoad<uint32_t>(srcAddressMapped));
    }

    void copyLoop(NBL_CONST_REF_ARG(TransferDispatchInfo) dispatchInfo, uint baseInvocationIndex, uint propertyId, TransferRequest transferRequest, uint dispatchSize)
    {
        uint64_t elementCount = uint64_t(transferRequest.elementCount32)
            | uint64_t(transferRequest.elementCountExtra) << 32;
        uint64_t lastInvocation = min(elementCount, dispatchInfo.endOffset);
        for (uint64_t invocationIndex = dispatchInfo.beginOffset + baseInvocationIndex; invocationIndex < lastInvocation; invocationIndex += dispatchSize)
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
    void copyLoop(NBL_CONST_REF_ARG(TransferDispatchInfo) dispatchInfo, uint baseInvocationIndex, uint propertyId, TransferRequest transferRequest, uint dispatchSize)
    {
       if (transferRequest.dstIndexSizeLog2 == 0) { TransferLoop<Fill, SrcIndexIota, DstIndexIota, SrcIndexSizeLog2, 0> loop; loop.copyLoop(dispatchInfo, baseInvocationIndex, propertyId, transferRequest, dispatchSize); }
       else if (transferRequest.dstIndexSizeLog2 == 1) { TransferLoop<Fill, SrcIndexIota, DstIndexIota, SrcIndexSizeLog2, 1> loop; loop.copyLoop(dispatchInfo, baseInvocationIndex, propertyId, transferRequest, dispatchSize); }
       else if (transferRequest.dstIndexSizeLog2 == 2) { TransferLoop<Fill, SrcIndexIota, DstIndexIota, SrcIndexSizeLog2, 2> loop; loop.copyLoop(dispatchInfo, baseInvocationIndex, propertyId, transferRequest, dispatchSize); }
       else /*if (transferRequest.dstIndexSizeLog2 == 3)*/ { TransferLoop<Fill, SrcIndexIota, DstIndexIota, SrcIndexSizeLog2, 3> loop; loop.copyLoop(dispatchInfo, baseInvocationIndex, propertyId, transferRequest, dispatchSize); }
    }
};

template<bool Fill, bool SrcIndexIota, bool DstIndexIota>
struct TransferLoopPermutationDstIota
{
    void copyLoop(NBL_CONST_REF_ARG(TransferDispatchInfo) dispatchInfo, uint baseInvocationIndex, uint propertyId, TransferRequest transferRequest, uint dispatchSize)
    {
       if (transferRequest.srcIndexSizeLog2 == 0) { TransferLoopPermutationSrcIndexSizeLog<Fill, SrcIndexIota, DstIndexIota, 0> loop; loop.copyLoop(dispatchInfo, baseInvocationIndex, propertyId, transferRequest, dispatchSize); }
       else if (transferRequest.srcIndexSizeLog2 == 1) { TransferLoopPermutationSrcIndexSizeLog<Fill, SrcIndexIota, DstIndexIota, 1> loop; loop.copyLoop(dispatchInfo, baseInvocationIndex, propertyId, transferRequest, dispatchSize); }
       else if (transferRequest.srcIndexSizeLog2 == 2) { TransferLoopPermutationSrcIndexSizeLog<Fill, SrcIndexIota, DstIndexIota, 2> loop; loop.copyLoop(dispatchInfo, baseInvocationIndex, propertyId, transferRequest, dispatchSize); }
       else /*if (transferRequest.srcIndexSizeLog2 == 3)*/ { TransferLoopPermutationSrcIndexSizeLog<Fill, SrcIndexIota, DstIndexIota, 3> loop; loop.copyLoop(dispatchInfo, baseInvocationIndex, propertyId, transferRequest, dispatchSize); }
    }
};

template<bool Fill, bool SrcIndexIota>
struct TransferLoopPermutationSrcIota
{
    void copyLoop(NBL_CONST_REF_ARG(TransferDispatchInfo) dispatchInfo, uint baseInvocationIndex, uint propertyId, TransferRequest transferRequest, uint dispatchSize)
    {
        bool dstIota = transferRequest.dstIndexAddr == 0;
        if (dstIota) { TransferLoopPermutationDstIota<Fill, SrcIndexIota, true> loop; loop.copyLoop(dispatchInfo, baseInvocationIndex, propertyId, transferRequest, dispatchSize); }
        else { TransferLoopPermutationDstIota<Fill, SrcIndexIota, false> loop; loop.copyLoop(dispatchInfo, baseInvocationIndex, propertyId, transferRequest, dispatchSize); }
    }
};

template<bool Fill>
struct TransferLoopPermutationFill
{
    void copyLoop(NBL_CONST_REF_ARG(TransferDispatchInfo) dispatchInfo, uint baseInvocationIndex, uint propertyId, TransferRequest transferRequest, uint dispatchSize)
    {
        bool srcIota = transferRequest.srcIndexAddr == 0;
        if (srcIota) { TransferLoopPermutationSrcIota<Fill, true> loop; loop.copyLoop(dispatchInfo, baseInvocationIndex, propertyId, transferRequest, dispatchSize); }
        else { TransferLoopPermutationSrcIota<Fill, false> loop; loop.copyLoop(dispatchInfo, baseInvocationIndex, propertyId, transferRequest, dispatchSize); }
    }
};

// Loading transfer request from the pointer (can't use struct
// with BDA on HLSL SPIRV)
static TransferRequest TransferRequest::newFromAddress(const uint64_t transferCmdAddr)
{   
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

    return transferRequest;
}

template<typename device_capabilities>
void main(uint32_t3 dispatchId, const uint dispatchSize)
{
    const uint propertyId = dispatchId.y;
    const uint invocationIndex = dispatchId.x;

    uint64_t transferCmdAddr = globals.transferCommandsAddress + sizeof(TransferRequest) * propertyId;
    TransferRequest transferRequest = TransferRequest::newFromAddress(transferCmdAddr);

    const bool fill = transferRequest.fill == 1;

    if (fill) { TransferLoopPermutationFill<true> loop; loop.copyLoop(globals, invocationIndex, propertyId, transferRequest, dispatchSize); }
    else { TransferLoopPermutationFill<false> loop; loop.copyLoop(globals, invocationIndex, propertyId, transferRequest, dispatchSize); }
}

}
}
}

[numthreads(nbl::hlsl::property_pools::OptimalDispatchSize,1,1)]
void main(uint32_t3 dispatchId : SV_DispatchThreadID)
{
    nbl::hlsl::property_pools::main<nbl::hlsl::jit::device_capabilities>(dispatchId, nbl::hlsl::property_pools::OptimalDispatchSize);
}

