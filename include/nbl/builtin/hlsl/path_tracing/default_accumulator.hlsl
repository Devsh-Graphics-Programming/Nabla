#ifndef _NBL_BUILTIN_HLSL_DEFAULT_ACCUMULATOR_INCLUDED_
#define _NBL_BUILTIN_HLSL_DEFAULT_ACCUMULATOR_INCLUDED_

#include <nbl/builtin/hlsl/concepts/vector.hlsl>
#include <nbl/builtin/hlsl/vector_utils/vector_traits.hlsl>

namespace nbl
{
namespace hlsl
{
namespace path_tracing
{

template<typename OutputTypeVec NBL_PRIMARY_REQUIRES(hlsl::concepts::FloatingPointVector<OutputTypeVec>)
struct DefaultAccumulator
{
    using input_sample_type = OutputTypeVec;
    using output_storage_type = OutputTypeVec;
    using this_t = DefaultAccumulator<OutputTypeVec>;
    using scalar_type = typename vector_traits<OutputTypeVec>::scalar_type;

    static this_t create()
    {
        this_t retval;
        retval.accumulation = promote<OutputTypeVec, scalar_type>(0.0f);

        return retval;
    }

    void addSample(uint32_t sampleCount, input_sample_type _sample)
    {
        scalar_type rcpSampleSize = 1.0 / (sampleCount);
        accumulation += (_sample - accumulation) * rcpSampleSize;
    }

    output_storage_type accumulation;
};

}
}
}

#endif
