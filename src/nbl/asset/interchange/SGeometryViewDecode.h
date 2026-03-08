// Internal src-only header.
// Do not include from public headers.
#ifndef _NBL_ASSET_S_GEOMETRY_VIEW_DECODE_H_INCLUDED_
#define _NBL_ASSET_S_GEOMETRY_VIEW_DECODE_H_INCLUDED_

#include "nbl/asset/ICPUPolygonGeometry.h"
#include "nbl/asset/format/decodePixels.h"
#include "nbl/builtin/hlsl/array_accessors.hlsl"
#include "nbl/builtin/hlsl/vector_utils/vector_traits.hlsl"

#include <array>
#include <tuple>
#include <type_traits>


namespace nbl::asset
{

class SGeometryViewDecode
{
	public:
		enum class EMode : uint8_t
		{
			Cooked,
			Raw
		};

		template<typename Out, EMode Mode = EMode::Cooked>
		static inline bool decodeElement(const ICPUPolygonGeometry::SDataView& view, const size_t ix, Out& out)
		{
			using scalar_t = typename STraits<Out>::scalar_type;

			out = {};
			if (!view.composed.isFormatted())
				return false;

			const void* const src = view.getPointer(ix);
			if (!src)
				return false;

			std::array<const void*, 4> srcArr = {src};
			std::array<scalar_t, 4> tmp = {};
			if (!decodePixels<scalar_t>(view.composed.format, srcArr.data(), tmp.data(), 0u, 0u))
				return false;

			const uint32_t channels = std::min<uint32_t>(STraits<Out>::Dimension, getFormatChannelCount(view.composed.format));
			if constexpr (Mode == EMode::Cooked && std::is_floating_point_v<scalar_t>)
			{
				if (isNormalizedFormat(view.composed.format))
				{
					const auto range = view.composed.getRange<hlsl::shapes::AABB<4, hlsl::float64_t>>();
					for (uint32_t i = 0u; i < channels; ++i)
						tmp[i] = static_cast<scalar_t>(tmp[i] * (range.maxVx[i] - range.minVx[i]) + range.minVx[i]);
				}
			}

			for (uint32_t i = 0u; i < channels; ++i)
				STraits<Out>::set(out, i, tmp[i]);
			return true;
		}

	private:
		template<typename T>
		struct SIsStdArray : std::false_type {};

		template<typename T, size_t N>
		struct SIsStdArray<std::array<T, N>> : std::true_type {};

		template<typename T, typename = void>
		struct SHasVectorTraits : std::false_type {};

		template<typename T>
		struct SHasVectorTraits<T, std::void_t<typename hlsl::vector_traits<T>::scalar_type>> : std::true_type {};

		template<typename Out, bool IsStdArray = SIsStdArray<Out>::value, bool IsVector = (!IsStdArray && SHasVectorTraits<Out>::value)>
		struct STraits;

		template<typename Out>
		struct STraits<Out, true, false>
		{
			using scalar_type = typename Out::value_type;
			static constexpr uint32_t Dimension = std::tuple_size_v<Out>;

			static inline void set(Out& out, const uint32_t ix, const scalar_type value)
			{
				out[ix] = value;
			}
		};

		template<typename Out>
		struct STraits<Out, false, true>
		{
			using scalar_type = typename hlsl::vector_traits<Out>::scalar_type;
			static constexpr uint32_t Dimension = hlsl::vector_traits<Out>::Dimension;

			static inline void set(Out& out, const uint32_t ix, const scalar_type value)
			{
				hlsl::array_set<Out, scalar_type> setter;
				setter(out, ix, value);
			}
		};
};

}

#endif
