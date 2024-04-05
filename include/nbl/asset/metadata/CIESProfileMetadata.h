// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_IES_METADATA_H_INCLUDED__
#define __NBL_ASSET_C_IES_METADATA_H_INCLUDED__

#include "nbl/asset/metadata/IAssetMetadata.h"
#include "nbl/asset/utils/CIESProfile.h"

namespace nbl
{
	namespace asset
	{
		class CIESProfileMetadata final : public asset::IAssetMetadata 
		{
			public:
				CIESProfileMetadata(const CIESProfile& _profile)
					: IAssetMetadata(), profile(_profile) {}

				_NBL_STATIC_INLINE_CONSTEXPR const char* LoaderName = "CIESProfileLoader";
				const char* getLoaderName() const override { return LoaderName; }

				bool flattenIESTexture(core::smart_refctd_ptr<asset::ICPUImageView> inIESIV, const float flatten = 0.0) const
				{
                    const bool inFlattenDomain = flatten >= 0.0 && flatten < 1.0; // [0, 1) range for blend equation, 1 is invalid

                    if (!inFlattenDomain)
                        return false;

                    auto inIES = inIESIV->getCreationParameters().image;

                    const auto& creationParams = inIES->getCreationParameters();
                    const bool validFormat = creationParams.format == CIESProfile::IES_TEXTURE_STORAGE_FORMAT;

                    if (!validFormat)
                        return false;

                    if (flatten > 0.0) // skip if 0 because its just copying texels
                    {
                        // TODO: this thing could be done with filters

                        char* bufferPtr = reinterpret_cast<char*>(inIES->getBuffer()->getPointer());
                        const auto texelBytesz = asset::getTexelOrBlockBytesize(creationParams.format);
                        const size_t bufferRowLength = asset::IImageAssetHandlerBase::calcPitchInBlocks(creationParams.extent.width, texelBytesz);

                        const auto avgEmmision = profile.getAvgEmmision();

                        for (size_t i = 0; i < creationParams.extent.height; i++)
                            for (size_t j = 0; j < creationParams.extent.width; j++)
                            {
                                auto* bufferP = reinterpret_cast<uint16_t*>(bufferPtr + i * bufferRowLength * texelBytesz + j * texelBytesz);
                                const auto decodeV = (double)(*bufferP) / CIESProfile::UI16_MAX_D;
                                
                                // blend the IES texture with "flatten"
                                const auto blendV = decodeV * (1.0 - flatten) + avgEmmision * flatten;

                                const uint16_t encodeV = static_cast<uint16_t>(std::clamp(blendV * CIESProfile::UI16_MAX_D, 0.0, CIESProfile::UI16_MAX_D));
                                *bufferP = encodeV;
                            }
                    }

                    return true;
				}

				const CIESProfile profile;
		};
	}
}

#endif // __NBL_ASSET_C_IES_METADATA_H_INCLUDED__
