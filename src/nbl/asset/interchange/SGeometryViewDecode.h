// Internal src-only header.
// Do not include from public headers.
#ifndef _NBL_ASSET_S_GEOMETRY_VIEW_DECODE_H_INCLUDED_
#define _NBL_ASSET_S_GEOMETRY_VIEW_DECODE_H_INCLUDED_
#include "nbl/asset/ICPUPolygonGeometry.h"
#include "nbl/asset/format/decodePixels.h"
#include "nbl/builtin/hlsl/concepts.hlsl"
#include "nbl/builtin/hlsl/vector_utils/vector_traits.hlsl"
#include <algorithm>
#include <array>
#include <type_traits>
namespace nbl::asset
{
class SGeometryViewDecode
{
	public:
		enum class EMode : uint8_t
		{
			Semantic,
			Stored
		};

		template<EMode Mode>
		struct Prepared
		{
			const uint8_t* data = nullptr;
			uint32_t stride = 0u;
			E_FORMAT format = EF_UNKNOWN;
			uint32_t channels = 0u;
			bool normalized = false;
			hlsl::shapes::AABB<4, hlsl::float64_t> range = hlsl::shapes::AABB<4, hlsl::float64_t>::create();

			inline explicit operator bool() const
			{
				return data != nullptr && stride != 0u && format != EF_UNKNOWN && channels != 0u;
			}

			template<typename T, size_t N>
			inline bool decode(const size_t ix, std::array<T, N>& out) const
			{
				out.fill(T{});
				return SGeometryViewDecode::template decodePrepared<Mode>(*this, ix, out.data(), static_cast<uint32_t>(N));
			}

			template<typename V> requires hlsl::concepts::Vector<V>
			inline bool decode(const size_t ix, V& out) const
			{
				out = V{};
				return SGeometryViewDecode::template decodePrepared<Mode>(*this, ix, out);
			}
		};

		template<EMode Mode>
		static inline Prepared<Mode> prepare(const ICPUPolygonGeometry::SDataView& view)
		{
			Prepared<Mode> retval = {};
			if (!view.composed.isFormatted())
				return retval;

			retval.data = reinterpret_cast<const uint8_t*>(view.getPointer());
			if (!retval.data)
				return {};

			retval.stride = view.composed.getStride();
			retval.format = view.composed.format;
			retval.channels = getFormatChannelCount(retval.format);
			if constexpr (Mode == EMode::Semantic)
			{
				retval.normalized = isNormalizedFormat(retval.format);
				if (retval.normalized)
					retval.range = view.composed.getRange<hlsl::shapes::AABB<4, hlsl::float64_t>>();
			}
			return retval;
		}

		template<typename Out, EMode Mode = EMode::Semantic>
		static inline bool decodeElement(const ICPUPolygonGeometry::SDataView& view, const size_t ix, Out& out)
		{
			return prepare<Mode>(view).decode(ix, out);
		}

	private:
		template<EMode Mode, typename T>
		static inline bool decodePreparedComponents(const Prepared<Mode>& prepared, const size_t ix, T* out, const uint32_t outDim)
		{
			if (!prepared || !out || outDim == 0u)
				return false;

			using storage_t = std::conditional_t<std::is_floating_point_v<T>, hlsl::float64_t, std::conditional_t<std::is_signed_v<T>, int64_t, uint64_t>>;
			std::array<storage_t, 4> tmp = {};
			const void* srcArr[4] = {prepared.data + ix * prepared.stride, nullptr};
			if (!decodePixels<storage_t>(prepared.format, srcArr, tmp.data(), 0u, 0u))
				return false;

			const uint32_t componentCount = std::min({prepared.channels, outDim, 4u});
			if constexpr (Mode == EMode::Semantic && std::is_floating_point_v<storage_t>)
			{
				if (prepared.normalized)
				{
					for (uint32_t i = 0u; i < componentCount; ++i)
						tmp[i] = static_cast<storage_t>(tmp[i] * (prepared.range.maxVx[i] - prepared.range.minVx[i]) + prepared.range.minVx[i]);
				}
			}

			for (uint32_t i = 0u; i < componentCount; ++i)
				out[i] = static_cast<T>(tmp[i]);
			return true;
		}

		template<EMode Mode, typename V> requires hlsl::concepts::Vector<V>
		static inline bool decodePrepared(const Prepared<Mode>& prepared, const size_t ix, V& out)
		{
			using scalar_t = typename hlsl::vector_traits<V>::scalar_type;
			constexpr uint32_t Dimension = hlsl::vector_traits<V>::Dimension;
			std::array<scalar_t, Dimension> tmp = {};
			if (!decodePreparedComponents(prepared, ix, tmp.data(), Dimension))
				return false;
			for (uint32_t i = 0u; i < Dimension; ++i)
				out[i] = tmp[i];
			return true;
		}

		template<EMode Mode, typename T>
		static inline bool decodePrepared(const Prepared<Mode>& prepared, const size_t ix, T* out, const uint32_t outDim)
		{
			return decodePreparedComponents(prepared, ix, out, outDim);
		}
};
}
#endif
