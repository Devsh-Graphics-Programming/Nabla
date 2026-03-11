// Internal src-only header. Do not include from public headers.
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
//! Shared decode helper for geometry `SDataView` read paths used by writers.
class SGeometryViewDecode
{
	public:
		//! Selects whether the output should be in logical attribute space or storage space.
		enum class EMode : uint8_t
		{
			Semantic, //!< Decode values ready for writer-side math and text/binary emission.
			Stored //!< Decode values in storage-domain form for raw integer emission.
		};

		//! Prepared decode state hoisted out of inner loops for one formatted view.
		template<EMode Mode>
		struct Prepared
		{
			const uint8_t* data = nullptr; //!< First byte of the view payload.
			uint32_t stride = 0u; //!< Byte stride between consecutive elements.
			E_FORMAT format = EF_UNKNOWN; //!< Source format used by `decodePixels`.
			uint32_t channels = 0u; //!< Channel count cached from `format`.
			bool normalized = false; //!< True when semantic decode must apply `range`.

			//! Decoded attribute range used for normalized semantic outputs.
			hlsl::shapes::AABB<4, hlsl::float64_t> range = hlsl::shapes::AABB<4, hlsl::float64_t>::create();
			inline explicit operator bool() const { return data != nullptr && stride != 0u && format != EF_UNKNOWN && channels != 0u; }

			//! Decodes one element into a fixed-size `std::array`.
			template<typename T, size_t N>
			inline bool decode(const size_t ix, std::array<T, N>& out) const { out.fill(T{}); return SGeometryViewDecode::template decodePrepared<Mode>(*this, ix, out.data(), static_cast<uint32_t>(N)); }

			//! Decodes one element into an HLSL vector type.
			template<typename V> requires hlsl::concepts::Vector<V>
			inline bool decode(const size_t ix, V& out) const { out = V{}; return SGeometryViewDecode::template decodePrepared<Mode>(*this, ix, out); }
		};

		//! Prepares one decode state that can be reused across many elements of the same view.
		template<EMode Mode>
		static inline Prepared<Mode> prepare(const ICPUPolygonGeometry::SDataView& view)
		{
			Prepared<Mode> retval = {};
			if (!view.composed.isFormatted())
				return {};
			if (!(retval.data = reinterpret_cast<const uint8_t*>(view.getPointer())))
				return {};
			retval.stride = view.composed.getStride();
			retval.format = view.composed.format;
			retval.channels = getFormatChannelCount(retval.format);
			if constexpr (Mode == EMode::Semantic)
				if (retval.normalized = isNormalizedFormat(retval.format); retval.normalized)
					retval.range = view.composed.getRange<hlsl::shapes::AABB<4, hlsl::float64_t>>();
			return retval;
		}

		//! One-shot convenience wrapper over `prepare(...).decode(...)`.
		template<typename Out, EMode Mode = EMode::Semantic>
		static inline bool decodeElement(const ICPUPolygonGeometry::SDataView& view, const size_t ix, Out& out) { return prepare<Mode>(view).decode(ix, out); }
	private:
		//! Shared scalar/vector backend that decodes one prepared element into plain components.
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

		//! Vector overload built on top of `decodePreparedComponents`.
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

		//! Pointer overload used by `std::array` and internal scratch storage.
		template<EMode Mode, typename T>
		static inline bool decodePrepared(const Prepared<Mode>& prepared, const size_t ix, T* out, const uint32_t outDim) { return decodePreparedComponents(prepared, ix, out, outDim); }
};
}
#endif
