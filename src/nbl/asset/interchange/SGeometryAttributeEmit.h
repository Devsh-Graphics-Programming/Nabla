// Internal src-only header.
// Do not include from public headers.
#ifndef _NBL_ASSET_S_GEOMETRY_ATTRIBUTE_EMIT_H_INCLUDED_
#define _NBL_ASSET_S_GEOMETRY_ATTRIBUTE_EMIT_H_INCLUDED_

#include "nbl/asset/interchange/SGeometryViewDecode.h"

#include <array>
#include <type_traits>


namespace nbl::asset
{

class SGeometryAttributeEmit
{
	public:
		template<typename Sink, typename OutT, SGeometryViewDecode::EMode Mode>
		static inline bool emit(Sink& sink, const SGeometryViewDecode::Prepared<Mode>& view, const size_t ix, const uint32_t componentCount, const bool flipVectors)
		{
			std::array<OutT, 4> decoded = {};
			if (!view.decode(ix, decoded))
				return false;
			for (uint32_t c = 0u; c < componentCount; ++c)
			{
				OutT value = decoded[c];
				if constexpr (std::is_signed_v<OutT> || std::is_floating_point_v<OutT>)
				{
					if (flipVectors && c == 0u)
						value = -value;
				}
				if (!sink.append(value))
					return false;
			}
			return true;
		}
};

}

#endif
