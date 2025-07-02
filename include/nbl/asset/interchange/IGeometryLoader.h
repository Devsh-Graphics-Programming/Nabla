// Copyright (C) 2025-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_GEOMETRY_LOADER_H_INCLUDED_
#define _NBL_ASSET_I_GEOMETRY_LOADER_H_INCLUDED_


#include "nbl/core/declarations.h"

#include "nbl/asset/ICPUPolygonGeometry.h"
#include "nbl/asset/interchange/IAssetLoader.h"
#include "nbl/asset/interchange/IImageAssetHandlerBase.h"


namespace nbl::asset
{

class IGeometryLoader : public IAssetLoader
{
	public:
		virtual inline uint64_t getSupportedAssetTypesBitfield() const override {return IAsset::ET_GEOMETRY;}

	protected:
		inline IGeometryLoader() {}

		static inline IGeometry<ICPUBuffer>::SDataView createView(const E_FORMAT format, const size_t elementCount, const void* data=nullptr)
		{
			const auto stride = getTexelOrBlockBytesize(format);
			auto buffer = ICPUBuffer::create({{stride*elementCount},const_cast<void*>(data)});
			return {
				.composed = {
					.stride = stride,
					.format = format,
					.rangeFormat = IGeometryBase::getMatchingAABBFormat(format)
				},
				.src = {.offset=0,.size=buffer->getSize(),.buffer=std::move(buffer)}
			};
		}

	private:
};

}

#endif
