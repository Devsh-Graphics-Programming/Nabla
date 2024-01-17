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
    uint64_t srcIndexAddr = 0; // IOTA default
    uint64_t dstIndexAddr = 0; // IOTA default
    uint64_t elementCount : 35; // allow up to 64GB IGPUBuffers
    uint64_t propertySize : 24; // all the leftover bits (just use bytes now)
    uint64_t fill : 1 = false;
    // 0=uint8, 1=uint16, 2=uint32, 3=uint64
    uint64_t srcIndexSizeLog2 : 2 = 1;
    uint64_t dstIndexSizeLog2 : 2 = 1;
};

struct GlobalPushContants 
{
    // Define the range of invocations (X axis) that will be transfered over in this dispatch
    // May be sectioned off in the case of overflow or any other situation that doesn't allow
    // for a full transfer
    uint64_t beginOffset;
    uint64_t endOffset;
    // BDA address (GPU pointer) into the transfer commands buffer
    uint64_t transferCommandsAddress;
};

}
}
}
