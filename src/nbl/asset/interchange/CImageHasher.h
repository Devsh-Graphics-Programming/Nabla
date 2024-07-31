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
				: hashers(std::make_unique<blake3_hasher[]>(_params.arrayLayers * _params.mipLevels)), 
				imageHasher({ _params.arrayLayers * _params.mipLevels }),
				arrayLayers(_params.arrayLayers), 
				mipLevels(_params.mipLevels) 
			{
				for(uint32_t i = 0 ; i < _params.arrayLayers * _params.mipLevels; i++){
					blake3_hasher_init(&hashers[i]);
				}
				blake3_hasher_init(&imageHasher);
			}
			~CImageHasher() = default;

			using hash_t = core::blake3_hash_t;
			using hasher_t = blake3_hasher;

			std::unique_ptr<hasher_t[]> hashers;
			hasher_t imageHasher;
			uint32_t arrayLayers, mipLevels;

			inline void partialHash(uint32_t mipLevel, uint32_t level, void* data, size_t dataLenght) {
				blake3_hasher_update(&this->hashers[mipLevel * this->arrayLayers + level], data, dataLenght);
			}

			inline void hashSeq(uint32_t mipLevel, uint32_t level, void* data, size_t dataLenght) {
				hasher_t* layerHasher = &this->hashers[mipLevel * this->arrayLayers + level];
				hash_t hash;
				blake3_hasher_update(layerHasher, data, dataLenght);
				blake3_hasher_finalize(layerHasher, reinterpret_cast<uint8_t*>(&hash), sizeof(hash_t));
				blake3_hasher_update(&imageHasher, &hash, sizeof(hash_t));
			}

			inline void hashSeq(uint32_t mipLevel, uint32_t level) {
				hasher_t* layerHasher = &this->hashers[mipLevel * this->arrayLayers + level];
				hash_t hash;
				blake3_hasher_finalize(layerHasher, reinterpret_cast<uint8_t*>(&hash), sizeof(hash_t));
				blake3_hasher_update(&imageHasher, &hash, sizeof(hash_t));
			}

			inline hash_t finalizeSeq() {
				hash_t hash;
				blake3_hasher_finalize(&imageHasher, reinterpret_cast<uint8_t*>(&hash), sizeof(hash_t));
				return hash;
			}

	};
}

#endif // _NBL_ASSET_C_IMAGE_HASHER_H_INCLUDED_