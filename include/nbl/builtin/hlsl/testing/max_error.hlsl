#ifndef _NBL_BUILTIN_HLSL_TESTING_MAX_ERROR_INCLUDED_
#define _NBL_BUILTIN_HLSL_TESTING_MAX_ERROR_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/basic.h>
#include <nbl/builtin/hlsl/concepts/core.hlsl>
#include <nbl/builtin/hlsl/vector_utils/vector_traits.hlsl>
#include <nbl/builtin/hlsl/matrix_utils/matrix_traits.hlsl>
#include <nbl/builtin/hlsl/math/functions.hlsl>

namespace nbl
{
namespace hlsl
{
namespace testing
{

struct SMaxError
{
	float64_t abs = 0.0;
	float64_t rel = 0.0;

	void updateScalar(float64_t expected, float64_t tested)
	{
		abs = hlsl::max(hlsl::abs(expected - tested), abs);
		if (expected != 0.0 && tested != 0.0)
			rel = hlsl::max(hlsl::max(expected / tested, tested / expected) - 1.0, rel);
	}
};

namespace impl
{

template<typename T NBL_STRUCT_CONSTRAINABLE>
struct MaxErrorUpdater;

template<typename FloatingPoint>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointLikeScalar<FloatingPoint>)
struct MaxErrorUpdater<FloatingPoint NBL_PARTIAL_REQ_BOT(concepts::FloatingPointLikeScalar<FloatingPoint>) >
{
	static void __call(NBL_REF_ARG(SMaxError) record, NBL_CONST_REF_ARG(FloatingPoint) expected, NBL_CONST_REF_ARG(FloatingPoint) tested)
	{
		record.updateScalar(float64_t(expected), float64_t(tested));
	}
};

template<typename FloatingPointVector>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointLikeVectorial<FloatingPointVector>)
struct MaxErrorUpdater<FloatingPointVector NBL_PARTIAL_REQ_BOT(concepts::FloatingPointLikeVectorial<FloatingPointVector>) >
{
	static void __call(NBL_REF_ARG(SMaxError) record, NBL_CONST_REF_ARG(FloatingPointVector) expected, NBL_CONST_REF_ARG(FloatingPointVector) tested)
	{
		using traits = nbl::hlsl::vector_traits<FloatingPointVector>;
		for (uint32_t i = 0; i < traits::Dimension; ++i)
			MaxErrorUpdater<typename traits::scalar_type>::__call(record, expected[i], tested[i]);
	}
};

template<typename FloatingPointMatrix>
NBL_PARTIAL_REQ_TOP(concepts::Matricial<FloatingPointMatrix> && concepts::FloatingPointLikeScalar<typename nbl::hlsl::matrix_traits<FloatingPointMatrix>::scalar_type>)
struct MaxErrorUpdater<FloatingPointMatrix NBL_PARTIAL_REQ_BOT(concepts::Matricial<FloatingPointMatrix> && concepts::FloatingPointLikeScalar<typename nbl::hlsl::matrix_traits<FloatingPointMatrix>::scalar_type>) >
{
	static void __call(NBL_REF_ARG(SMaxError) record, NBL_CONST_REF_ARG(FloatingPointMatrix) expected, NBL_CONST_REF_ARG(FloatingPointMatrix) tested)
	{
		using traits = nbl::hlsl::matrix_traits<FloatingPointMatrix>;
		for (uint32_t i = 0; i < traits::RowCount; ++i)
			MaxErrorUpdater<typename traits::row_type>::__call(record, expected[i], tested[i]);
	}
};

}

template<typename T>
void updateMaxError(NBL_REF_ARG(SMaxError) record, NBL_CONST_REF_ARG(T) expected, NBL_CONST_REF_ARG(T) tested)
{
	impl::MaxErrorUpdater<T>::__call(record, expected, tested);
}

}
}
}

#endif
