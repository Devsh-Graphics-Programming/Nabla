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
	static constexpr uint32_t MaxComponents = 16;
	float64_t abs[MaxComponents] = {};
	float64_t rel[MaxComponents] = {};
	uint32_t rows = 1;
	uint32_t cols = 1;

	void updateScalar(float64_t expected, float64_t tested, uint32_t idx)
	{
		abs[idx] = hlsl::max(hlsl::abs(expected - tested), abs[idx]);
		if (expected != 0.0 && tested != 0.0)
			rel[idx] = hlsl::max(hlsl::max(expected / tested, tested / expected) - 1.0, rel[idx]);
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
	static void __call(NBL_REF_ARG(SMaxError) record, NBL_CONST_REF_ARG(FloatingPoint) expected, NBL_CONST_REF_ARG(FloatingPoint) tested, uint32_t offset)
	{
		record.updateScalar(float64_t(expected), float64_t(tested), offset);
	}
};

template<typename FloatingPointVector>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointLikeVectorial<FloatingPointVector>)
struct MaxErrorUpdater<FloatingPointVector NBL_PARTIAL_REQ_BOT(concepts::FloatingPointLikeVectorial<FloatingPointVector>) >
{
	static void __call(NBL_REF_ARG(SMaxError) record, NBL_CONST_REF_ARG(FloatingPointVector) expected, NBL_CONST_REF_ARG(FloatingPointVector) tested, uint32_t offset)
	{
		using traits = nbl::hlsl::vector_traits<FloatingPointVector>;
		for (uint32_t i = 0; i < traits::Dimension; ++i)
			MaxErrorUpdater<typename traits::scalar_type>::__call(record, expected[i], tested[i], offset + i);
	}
};

template<typename FloatingPointMatrix>
NBL_PARTIAL_REQ_TOP(concepts::Matricial<FloatingPointMatrix> && concepts::FloatingPointLikeScalar<typename nbl::hlsl::matrix_traits<FloatingPointMatrix>::scalar_type>)
struct MaxErrorUpdater<FloatingPointMatrix NBL_PARTIAL_REQ_BOT(concepts::Matricial<FloatingPointMatrix> && concepts::FloatingPointLikeScalar<typename nbl::hlsl::matrix_traits<FloatingPointMatrix>::scalar_type>) >
{
	static void __call(NBL_REF_ARG(SMaxError) record, NBL_CONST_REF_ARG(FloatingPointMatrix) expected, NBL_CONST_REF_ARG(FloatingPointMatrix) tested, uint32_t offset)
	{
		using traits = nbl::hlsl::matrix_traits<FloatingPointMatrix>;
		for (uint32_t i = 0; i < traits::RowCount; ++i)
			MaxErrorUpdater<typename traits::row_type>::__call(record, expected[i], tested[i], offset + i * traits::ColumnCount);
	}
};

}

template<typename T>
void updateMaxError(NBL_REF_ARG(SMaxError) record, NBL_CONST_REF_ARG(T) expected, NBL_CONST_REF_ARG(T) tested)
{
	if constexpr (concepts::FloatingPointLikeScalar<T>)
	{
		record.rows = 1;
		record.cols = 1;
	}
	else if constexpr (concepts::FloatingPointLikeVectorial<T>)
	{
		record.rows = 1;
		record.cols = vector_traits<T>::Dimension;
	}
	else if constexpr (concepts::Matricial<T>)
	{
		record.rows = matrix_traits<T>::RowCount;
		record.cols = matrix_traits<T>::ColumnCount;
	}
   else
      static_assert(false, "Unsupported type for max error updater");
	impl::MaxErrorUpdater<T>::__call(record, expected, tested, 0);
}

}
}
}

#endif
