#ifndef _NBL_BUILTIN_HLSL_GLSL_PROPERTY_POOLS_TRANSFER_
#define _NBL_BUILTIN_HLSL_GLSL_PROPERTY_POOLS_TRANSFER_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

namespace nbl
{
namespace hlsl
{
namespace property_pools
{

struct TransferRequest
{
    // This represents a transfer command/request
    uint64_t srcAddr;
    uint64_t dstAddr;
    uint64_t srcIndexAddr; // IOTA default
    uint64_t dstIndexAddr; // IOTA default
    // TODO: go back to this ideal layout when things work
    // (Getting a fatal error from DXC when using 64-bit bitfields:)
    // fatal error: generated SPIR-V is invalid: [VUID-StandaloneSpirv-Base-04781] Expected 32-bit int type for Base operand: BitFieldInsert
    // %58 = OpBitFieldInsert %ulong %42 %57 %uint_0 %uint_35
    //
    //uint64_t elementCount : 35; // allow up to 64GB IGPUBuffers
    //uint64_t propertySize : 24; // all the leftover bits (just use bytes now)
    //uint64_t fill : 1;
    //// 0=uint8, 1=uint16, 2=uint32, 3=uint64
    //uint64_t srcIndexSizeLog2 : 2;
    //uint64_t dstIndexSizeLog2 : 2;
    uint32_t elementCount32; // 32 first bits
    uint32_t elementCountExtra : 3; // 3 last bits
    uint32_t propertySize : 24;
    uint32_t fill: 1;
    uint32_t srcIndexSizeLog2 : 2;
    uint32_t dstIndexSizeLog2 : 2;
    
    // Reads a TransferRequest from a BDA
    static TransferRequest newFromAddress(const uint64_t address);
};

struct TransferDispatchInfo 
{
    // BDA address (GPU pointer) into the transfer commands buffer
    uint64_t transferCommandsAddress;
    // Define the range of invocations (X axis) that will be transfered over in this dispatch
    // May be sectioned off in the case of overflow or any other situation that doesn't allow
    // for a full transfer
    uint64_t beginOffset;
    uint64_t endOffset;
};

NBL_CONSTEXPR uint32_t MaxPropertiesPerDispatch = 128;

// TODO: instead use some sort of replace function for getting optimal size?
NBL_CONSTEXPR uint32_t OptimalDispatchSize = 256;

}
}
}

#endif

