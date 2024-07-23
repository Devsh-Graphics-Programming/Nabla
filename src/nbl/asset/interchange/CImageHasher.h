#ifndef _NBL_ASSET_C_IMAGE_HASHER_H_INCLUDED_
#define _NBL_ASSET_C_IMAGE_HASHER_H_INCLUDED_

#include "nbl/core/declarations.h"
#include "nbl/asset/IImage.h"

namespace nbl::asset
{
	class NBL_API2 CImageHasher : public core::Uncopyable
	{
		public:
			CImageHasher(const IImage::SCreationParams& _params) 
				: hashers(std::make_unique<blake3_hasher[]>(_params.arrayLayers * _params.mipLevels)) {}
			~CImageHasher() = default;

			std::unique_ptr<blake3_hasher[]> hashers;
	};
}

#endif // _NBL_ASSET_C_IMAGE_HASHER_H_INCLUDED_