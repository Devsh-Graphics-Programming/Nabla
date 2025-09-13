#ifndef _NBL_ASSET_C_IMAGE_HASHER_H_INCLUDED_
#define _NBL_ASSET_C_IMAGE_HASHER_H_INCLUDED_

#include "nbl/core/declarations.h"
#include "nbl/asset/IImage.h"

namespace nbl::asset
{
	class NBL_API2 CImageHasher : public core::Uncopyable
	{
		public:
			using hash_t = core::blake3_hash_t;
			using hasher_t = core::blake3_hasher;

			std::unique_ptr<hasher_t[]> hashers;
			hasher_t imageHasher;
			uint32_t arrayLayers, mipLevels;

			CImageHasher(const IImage::SCreationParams& _params) 
				: hashers(std::make_unique<hasher_t[]>(_params.arrayLayers * _params.mipLevels)),
				imageHasher({}),
				arrayLayers(_params.arrayLayers), 
				mipLevels(_params.mipLevels) 
			{}

			~CImageHasher() = default;

			inline void partialHash(uint32_t mipLevel, uint32_t level, void* data, size_t dataLenght) {
				hasher_t& layerHasher = this->hashers[mipLevel * this->arrayLayers + level];
				layerHasher.update(data, dataLenght);
			}

			inline void hashSeq(uint32_t mipLevel, uint32_t level, void* data, size_t dataLenght) {
				hasher_t& layerHasher = this->hashers[mipLevel * this->arrayLayers + level];
				layerHasher.update(data, dataLenght);
				imageHasher << static_cast<hash_t>(layerHasher);
			}

			inline void hashSeq(uint32_t mipLevel, uint32_t level) {
				hasher_t layerHasher = this->hashers[mipLevel * this->arrayLayers + level];
				imageHasher << static_cast<hash_t>(layerHasher);
			}

			inline hash_t finalizeSeq() {
				return static_cast<hash_t>(imageHasher);
			}

	};
}

#endif // _NBL_ASSET_C_IMAGE_HASHER_H_INCLUDED_