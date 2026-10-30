// this shader has to be compiled with:
// -T cs_6_0 -E main

#include "nbl/builtin/hlsl/sort/counting.hlsl"
#include "nbl/builtin/hlsl/bda/bda_accessor.hlsl"

#define WorkgroupSize 27
#define BucketCount 27

struct CountingPushData
{
    uint64_t inputKeyAddress;
    uint64_t inputValueAddress;
    uint64_t histogramAddress;
    uint64_t outputKeyAddress;
    uint64_t outputValueAddress;
    uint32_t dataElementCount;
    uint32_t elementsPerWT;
    uint32_t minimum;
    uint32_t maximum;
};

using
namespace nbl::
    hlsl;

usingPtr = bda::__ptr<uint32_t>;
    using PtrAccessor = BdaAccessor < uint32_t >;

    groupshared uint32_t sdata[BucketCount];

    struct SharedAccessor
    {
        void get(const uint32_t index, NBL_REF_ARG( uint32_t) value)
    {
        value =
        sdata[ index];
    }

    void set(
    const uint32_t index, const uint32_tvalue)
    {
        sdata[index] =
    value;
}

uint32_t atomicAdd(const uint32_t index, const uint32_t value)
{
    return glsl::atomicAdd(sdata[index], value);
}

void workgroupExecutionAndMemoryBarrier()
{
    glsl::barrier();
}
};

uint32_t3 glsl::gl_WorkGroupSize()
{
    return uint32_t3(WorkgroupSize, 1, 1);
}

[[vk::push_constant]] CountingPushData pushData;

using DoublePtrAccessor = DoubleBdaAccessor < uint32_t >;

[numthreads(WorkgroupSize, 1, 1)]
void main(uint32_t3 ID : SV_GroupThreadID, uint32_t3 GroupID : SV_GroupID)
{
    sort::CountingParameters < uint32_t > params;
    params.dataElementCount = pushData.dataElementCount;
    params.elementsPerWT = pushData.elementsPerWT;
    params.minimum = pushData.minimum;
    params.maximum = pushData.maximum;

    using Counter = sort::counting < WorkgroupSize, BucketCount, DoublePtrAccessor, DoublePtrAccessor, PtrAccessor, SharedAccessor, PtrAccessor::type_t >;
    Counter counter = Counter::create(glsl::gl_WorkGroupID().x);

    const Ptr input_key_ptr = Ptr::create(pushData.inputKeyAddress);
    const Ptr input_value_ptr = Ptr::create(pushData.inputValueAddress);
    const Ptr histogram_ptr = Ptr::create(pushData.histogramAddress);
    const Ptr output_key_ptr = Ptr::create(pushData.outputKeyAddress);
    const Ptr output_value_ptr = Ptr::create(pushData.outputValueAddress);

    DoublePtrAccessor key_accessor = DoublePtrAccessor::create(
        input_key_ptr,
        output_key_ptr
    );
    DoublePtrAccessor value_accessor = DoublePtrAccessor::create(
        input_value_ptr,
        output_value_ptr
    );
    PtrAccessor histogram_accessor = PtrAccessor::create(histogram_ptr);
    SharedAccessor shared_accessor;
    counter.scatter(
        key_accessor,
        value_accessor,
        histogram_accessor,
        shared_accessor,
        params
    );
}
