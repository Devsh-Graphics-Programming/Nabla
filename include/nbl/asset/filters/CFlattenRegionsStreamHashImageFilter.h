// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_FLATTEN_REGIONS_STREAM_HASH_IMAGE_FILTER_H_INCLUDED__
#define __NBL_ASSET_C_FLATTEN_REGIONS_STREAM_HASH_IMAGE_FILTER_H_INCLUDED__

#include <type_traits>

#include "nbl/core/declarations.h"

#include "nbl/asset/filters/CMatchedSizeInOutImageFilterCommon.h"
#include "nbl/asset/filters/CFlattenRegionsImageFilter.h"
#include "blake/c/blake3.h"

namespace nbl
{
namespace asset
{
class CFlattenRegionsStreamHashImageFilter : public CMatchedSizeInOutImageFilterCommon
{
	public:
		virtual ~CFlattenRegionsStreamHashImageFilter() {}

		struct ScratchMemory
		{
			core::smart_refctd_ptr<asset::ICPUImage> flatten; // for flattening input regions & prefilling with 0-value texels not covered by regions 
			core::smart_refctd_ptr<asset::ICPUBuffer> heap; // for storing hashes, single hash is obtained from given miplevel & layer, full hash for an image is a hash of this stack
		};
		
		class CState : public IImageFilter::IState
		{
			public:
				virtual ~CState() {}

				using hash_t = std::array<size_t, 4u>;

				const ICPUImage*					inImage = nullptr;
				hash_t								outHash = { {0} };
				ScratchMemory						scratchMemory;
		};

		using state_type = CState;

		static inline ScratchMemory allocateScratchMemory(const asset::ICPUImage* inImage)
		{
			ScratchMemory scratch;
			scratch.flatten = asset::IAsset::castDown<asset::ICPUImage>(inImage->clone());

			const auto& parameters = scratch.flatten->getCreationParameters();
			scratch.heap = core::make_smart_refctd_ptr<asset::ICPUBuffer>(parameters.mipLevels * parameters.arrayLayers * sizeof(CState::outHash));

			return scratch;
		}

		static inline bool validate(state_type* state)
		{
			if (!state)
				return false;

			if(!state->inImage)
				return false;

			if(!state->scratchMemory.flatten)
				return false;

			if (state->scratchMemory.flatten->getBuffer()->getSize() != state->inImage->getBuffer()->getSize())
				return false;

			if (!state->scratchMemory.heap)
				return false;

			const auto& parameters = state->inImage->getCreationParameters();

			if (state->scratchMemory.heap->getSize() != parameters.mipLevels * parameters.arrayLayers * sizeof(CState::outHash))
				return false;

			CFlattenRegionsImageFilter::state_type flatten;
			{
				flatten.inImage = state->inImage;
				flatten.outImage = state->scratchMemory.flatten;

				if (!CFlattenRegionsImageFilter::validate(&flatten)) // just to not DRY some of extra common validation steps
					return false;
			}

			return true;
		}

		template<class ExecutionPolicy>
		static inline bool execute(ExecutionPolicy&& policy, state_type* state)
		{
			if (!validate(state))
				return false;

			struct
			{
				CFlattenRegionsImageFilter::state_type flatten;			// CFlattenRegionsImageFilter's state, we use scratch memory to respecify regions & prefill with 0-value non-covered texels
			} proxy;

			/*
				first we need to ensure that texels which are not covered by regions 
				are filled with 0 value, flatten will also respecify regions to make
				sure we don't have overlaping regions - we will override them anyway
			*/

			{
				auto& flatten = proxy.flatten;
				flatten.inImage = state->inImage;
				flatten.outImage = state->scratchMemory.flatten;
				flatten.preFill = true;
				memset(flatten.fillValue.pointer, 0, sizeof(flatten.fillValue.pointer));

				assert(CFlattenRegionsImageFilter::execute(policy, &proxy.flatten)); // this should never fail, at this point we are already validated
			}

			/*
				now when the output is prepared we ignore respecified regions and go
				with single region covering all texels for a given mip level & layers 
			*/

			const auto& parameters = proxy.flatten.outImage->getCreationParameters();
			const uint8_t* inData = reinterpret_cast<const uint8_t*>(proxy.flatten.outImage->getBuffer()->getPointer());
			const TexelBlockInfo info(parameters.format);
			const auto bytesPerPixel = proxy.flatten.outImage->getBytesPerPixel();
			const auto texelOrBlockByteSize = asset::getTexelOrBlockBytesize(parameters.format);

			// override regions, we need to cover all texels
			auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<IImage::SBufferCopy>>(parameters.mipLevels);
			{
				size_t bufferSize = 0ull;

				for (auto rit = regions->begin(); rit != regions->end(); rit++)
				{
					auto miplevel = static_cast<uint32_t>(std::distance(regions->begin(), rit));
					auto localExtent = proxy.flatten.outImage->getMipSize(miplevel);
					rit->bufferOffset = bufferSize;
					rit->bufferRowLength = localExtent.x; // could round up to multiple of 8 bytes in the future
					rit->bufferImageHeight = localExtent.y;
					rit->imageSubresource.aspectMask = static_cast<IImage::E_ASPECT_FLAGS>(0u);
					rit->imageSubresource.mipLevel = miplevel;
					rit->imageSubresource.baseArrayLayer = 0u;
					rit->imageSubresource.layerCount = parameters.arrayLayers;
					rit->imageOffset = { 0u,0u,0u };
					rit->imageExtent = { localExtent.x,localExtent.y,localExtent.z };
					auto levelSize = info.roundToBlockSize(localExtent);
					auto memsize = levelSize[0] * levelSize[1] * levelSize[2] * parameters.arrayLayers * bytesPerPixel;

					assert(memsize.getNumerator() % memsize.getDenominator() == 0u);
					bufferSize += memsize.getIntegerApprox();
				}
			}

			auto executePerMipLevel = [&](const uint32_t miplevel)
			{
				/*
					we stream-hash texels per given mip level & layer
				*/

				auto* hashers = _NBL_NEW_ARRAY(blake3_hasher, parameters.arrayLayers);

				for (auto layer = 0u; layer < parameters.arrayLayers; ++layer)
					blake3_hasher_init(&hashers[layer]);

				auto hash = [&hashers, &inData, &texelOrBlockByteSize](uint32_t readBlockArrayOffset, core::vectorSIMDu32 readBlockPos) -> void
				{
					auto* hasher = hashers + readBlockPos.w;
					blake3_hasher_update(hasher, inData + readBlockArrayOffset, texelOrBlockByteSize);
				};

				IImage::SSubresourceLayers subresource = { .aspectMask = static_cast<IImage::E_ASPECT_FLAGS>(0u), .mipLevel = miplevel, .baseArrayLayer = 0u, .layerCount = parameters.arrayLayers }; // stick to given mip level and take all layers
				CMatchedSizeInOutImageFilterCommon::state_type::TexelRange range = { .offset = {}, .extent = { parameters.extent.width, parameters.extent.height, parameters.extent.depth } }; // cover all texels within layer range, take 0th mip level size to not clip anything at all
				CBasicImageFilterCommon::clip_region_functor_t clipFunctor(subresource, range, parameters.format);

				CBasicImageFilterCommon::executePerRegion(policy, proxy.flatten.outImage.get(), hash, regions->begin(), regions->end(), clipFunctor); // fire the hasher for layers with specified execution policy, yes you can use parallel policy here if you want at it will work

				for (auto layer = 0u; layer < parameters.arrayLayers; ++layer)
				{
					auto* out = reinterpret_cast<uint8_t*>(reinterpret_cast<CState::hash_t*>(state->scratchMemory.heap->getPointer()) + (miplevel * parameters.arrayLayers) + layer);
					blake3_hasher_finalize(hashers + layer, out, sizeof(CState::hash_t)); // finalize hash for layer + put it to heap for given mip level
				}

				_NBL_DELETE_ARRAY(hashers, parameters.arrayLayers);
			};

			std::vector<uint32_t> levels(parameters.mipLevels);
			std::iota(levels.begin(), levels.end(), 0);

			std::for_each(policy, levels.begin(), levels.end(), executePerMipLevel); // fire per block of layers for given mip level with specified execution policy, yes you can use parallel policy here if you want at it will work

			/*
				scratch's heap is filled with all hashes, 
				time to use them and compute final hash
			*/

			blake3_hasher hasher;
			blake3_hasher_init(&hasher);
			{
				for (auto miplevel = 0u; miplevel < parameters.mipLevels; ++miplevel)
					for (auto layer = 0u; layer < parameters.arrayLayers; ++layer)
					{
						auto* hash = reinterpret_cast<CState::hash_t*>(state->scratchMemory.heap->getPointer()) + miplevel * parameters.mipLevels + layer;
						blake3_hasher_update(&hasher, hash, sizeof(CState::hash_t));
					}

				blake3_hasher_finalize(&hasher, reinterpret_cast<uint8_t*>(state->outHash.data()), sizeof(CState::hash_t)); // finalize output hash for whole image given all hashes
			}

			return true;
		}
		static inline bool execute(state_type* state)
		{
			return execute(core::execution::seq, state);
		}
};

} // end namespace asset
} // end namespace nbl

#endif // __NBL_ASSET_C_FLATTEN_REGIONS_STREAM_HASH_IMAGE_FILTER_H_INCLUDED__