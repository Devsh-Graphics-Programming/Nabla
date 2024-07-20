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
		
		class CState : public IImageFilter::IState
		{
			public:
				virtual ~CState() {}

				using hash_t = std::array<size_t, 4u>;

				const ICPUImage*					inImage = nullptr;
				hash_t								outHash = { {0} };

				struct ScratchMemory
				{
					void* memory;	// pointer to scratch memory
					size_t size;	// size of scratch memory in bytes
				} scratch;

				static inline size_t getRequiredScratchByteSize(const ICPUImage* input)
				{
					assert(input);

					const auto& parameters = input->getCreationParameters();
					const auto product = parameters.mipLevels * parameters.arrayLayers;

					size_t bufferSize = product * sizeof(CState::outHash);
					bufferSize += product * sizeof(blake3_hasher);
					bufferSize += getFlattenBufferSize(input);

					return bufferSize;
				}

				private:	
					static size_t getFlattenBufferSize(const ICPUImage* input)
					{
						bool isFullyFlatten = true;
						size_t bufferSize = 0ull;

						const auto& parameters = input->getCreationParameters();
						const TexelBlockInfo info(parameters.format);
						const auto bytesPerPixel = input->getBytesPerPixel();

						for (uint32_t i = 0u; i < parameters.mipLevels; ++i)
						{
							const auto regions = input->getRegions(i);

							if (!regions.empty())
							{
								const auto& region = regions[0];
								const auto mipExtent = input->getMipSize(i);

								const bool isFlatten = regions.size() == 1 && region.getDstOffset() == nbl::asset::VkOffset3D{ 0, 0, 0 } && region.getExtent() == nbl::asset::VkExtent3D{ mipExtent.x, mipExtent.y, mipExtent.z };

								if (!isFlatten)
									isFullyFlatten = false;

								const auto levelSize = info.roundToBlockSize(mipExtent);
								const auto memsize = levelSize[0] * levelSize[1] * levelSize[2] * parameters.arrayLayers * bytesPerPixel;

								assert(memsize.getNumerator() % memsize.getDenominator() == 0u);
								bufferSize += memsize.getIntegerApprox();
							}
						}

						return isFullyFlatten ? 0ull : bufferSize;
					}

				friend class CFlattenRegionsStreamHashImageFilter;
		};

		using state_type = CState;

		static inline bool validate(state_type* state)
		{
			if (!state)
				return false;

			if(!state->inImage)
				return false;

			if(!state->scratch.memory)
				return false;

			const auto& parameters = state->inImage->getCreationParameters();

			if (state->scratch.size != state_type::getRequiredScratchByteSize(state->inImage))
				return false;

			CFlattenRegionsImageFilter::state_type flatten;
			{
				flatten.inImage = state->inImage;

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

			const auto& parameters = state->inImage->getCreationParameters();
			const TexelBlockInfo info(parameters.format);
			const auto bytesPerPixel = state->inImage->getBytesPerPixel();

			auto getScratchAsBuffer = [&memory = state->scratch.memory](size_t size, size_t offset = 0ull)
			{
				return core::make_smart_refctd_ptr<asset::CCustomAllocatorCPUBuffer<core::null_allocator<uint8_t>, true> >(size, (uint8_t*)memory + offset, core::adopt_memory); // adopt memory & don't free it on exit
			};
			
			/*
				we use scratch memory and map it with follwing [hashes][hashers][tight flatten buffer] memory layout,
				we guarantee that hashes are at the beginning of the memory and it won't change within API updates
			*/
			
			ScratchMap scratch;
			{
				auto buffer = getScratchAsBuffer(state->scratch.size);
				const auto product = parameters.mipLevels * parameters.arrayLayers;

				scratch.hashes = {.offset = 0u, .size = product * sizeof(CState::outHash), .buffer = buffer };
				scratch.hashers = { .offset = scratch.hashes.size, .size = product * sizeof(blake3_hasher), .buffer = buffer};
				scratch.flatten = { .offset = scratch.hashers.offset + scratch.hashers.size, .size = state->scratch.size - scratch.hashers.size - scratch.hashes.size, .buffer = buffer };
			}

			const auto isFullyFlatten = scratch.flatten.size == 0ull;

			struct
			{
				CFlattenRegionsImageFilter::state_type flatten;	// CFlattenRegionsImageFilter's state, we use scratch memory to respecify regions & prefill with 0-value non-covered texels
			} proxy;

			/*
				first we need to ensure that texels which are not covered by regions 
				are filled with 0 value, we use flatten with tight buffer for the 
				texel copy & prefill if we don't have fully flatten input
			*/

			if (!isFullyFlatten) // TODO: we may think of even better optimization if we have mixed regions for mips (eg. a few flattened, some others not and left empty)
			{
				// create own regions & hook tight buffer with no gaps from scratch
				auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<IImage::SBufferCopy>>(parameters.mipLevels);
				{
					size_t bufferSize = 0ull;

					for (auto rit = regions->begin(); rit != regions->end(); rit++)
					{
						auto miplevel = static_cast<uint32_t>(std::distance(regions->begin(), rit));
						auto localExtent = state->inImage->getMipSize(miplevel);
						rit->bufferOffset = bufferSize;
						rit->bufferRowLength = localExtent.x; // could round up to multiple of 8 bytes in the future
						rit->bufferImageHeight = localExtent.y;
						rit->imageSubresource.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT; // otherwise won't pass validaiton 
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

				auto out = ICPUImage::create(IImage::SCreationParams(parameters));
				out->setBufferAndRegions(core::smart_refctd_ptr(getScratchAsBuffer(scratch.flatten.size, scratch.flatten.offset)), std::move(regions));

				auto& flatten = proxy.flatten;
				flatten.inImage = state->inImage;
				flatten.outImage = out;
				flatten.preFill = true;
				memset(flatten.fillValue.pointer, 0, sizeof(flatten.fillValue.pointer));

				assert(CFlattenRegionsImageFilter::execute(policy, &proxy.flatten)); // this should never fail, at this point we are already validated
			}

			/*
				now when the output is prepared we go with single 
				region covering all texels for a given mip level & layers,
				note if we have optimized (fully flattened) input we skip 
				flattening by hand to avoid extra allocations & copies
			*/

			const auto* const image = isFullyFlatten ? state->inImage : proxy.flatten.outImage.get();
			const auto texelOrBlockByteSize = asset::getTexelOrBlockBytesize(parameters.format);
			const uint8_t* inData = reinterpret_cast<const uint8_t*>(image->getBuffer()->getPointer());

			std::vector<uint32_t> layers(parameters.arrayLayers);
			std::iota(layers.begin(), layers.end(), 0);

			std::vector<uint32_t> levels(parameters.mipLevels);
			std::iota(levels.begin(), levels.end(), 0);

			/*
				we stream-hash texels per given mip level & layer
			*/

            auto* const hashes = reinterpret_cast<CState::hash_t*>(getScratchAsBuffer(scratch.hashes.size, scratch.hashes.offset)->getPointer());
            auto* const hashers = reinterpret_cast<blake3_hasher*>(getScratchAsBuffer(scratch.hashers.size, scratch.hashers.offset)->getPointer());

			auto executePerMipLevel = [&](const uint32_t miplevel)
			{
				const auto mipOffset = (miplevel * parameters.arrayLayers);

				auto executePerLayer = [&](const uint32_t layer)
				{
					const auto pOffset = mipOffset + layer;
					auto* const hasher = hashers + pOffset;
					auto* const hash = hashes + pOffset;

					blake3_hasher_init(hasher);

					IImage::SSubresourceLayers subresource = { .aspectMask = static_cast<IImage::E_ASPECT_FLAGS>(0u), .mipLevel = miplevel, .baseArrayLayer = layer, .layerCount = 1u }; // stick to given mip level and single layer
					CMatchedSizeInOutImageFilterCommon::state_type::TexelRange range = { .offset = {}, .extent = { parameters.extent.width, parameters.extent.height, parameters.extent.depth } }; // cover all texels within layer range, take 0th mip level size to not clip anything at all
					CBasicImageFilterCommon::clip_region_functor_t clipFunctor(subresource, range, parameters.format);

					auto executePerTexelOrBlock = [&](uint32_t readBlockArrayOffset, core::vectorSIMDu32 readBlockPos) -> void
					{
						blake3_hasher_update(hasher, inData + readBlockArrayOffset, texelOrBlockByteSize);
					};

					const auto regions = image->getRegions(miplevel);
					const bool performNullHash = regions.empty();

					if (performNullHash)
					{
						const auto mipExtentInBlocks = info.convertTexelsToBlocks(image->getMipSize(miplevel));
						const auto zeroLength = info.getBlockByteSize() * mipExtentInBlocks.x;
						auto zeroArray = std::make_unique<uint8_t[]>(zeroLength);
						for (auto z = 0; z < mipExtentInBlocks.z; z++)
							for (auto y = 0; y < mipExtentInBlocks.y; y++)
								blake3_hasher_update(hasher, zeroArray.get(), zeroLength);
					}
					else
						CBasicImageFilterCommon::executePerRegion(std::execution::seq, image, executePerTexelOrBlock, regions.begin(), regions.end(), clipFunctor); // fire the hasher for a layer, note we forcing seq policy because texels/blocks cannot be handled with par policies when we hash them
				
					blake3_hasher_finalize(hasher, reinterpret_cast<uint8_t*>(hash), sizeof(CState::hash_t)); // finalize hash for layer + put it to heap for given mip level	
				};

				std::for_each(policy, layers.begin(), layers.end(), executePerLayer); // fire per layer for given given mip level with specified execution policy, yes you can use parallel policy here if you want at it will work
			};

			std::for_each(policy, levels.begin(), levels.end(), executePerMipLevel); // fire per block of layers for given mip level with specified execution policy, yes you can use parallel policy here if you want at it will work

			/*
				scratch's heap is filled with all hashes, 
				time to use them and compute final hash
			*/

			blake3_hasher hasher;
			blake3_hasher_init(&hasher);
			{
				for (auto miplevel = 0u; miplevel < parameters.mipLevels; ++miplevel)
				{
					const auto mipOffset = (miplevel * parameters.arrayLayers);

					for (auto layer = 0u; layer < parameters.arrayLayers; ++layer)
					{
						auto* hash = hashes + mipOffset + layer;
						blake3_hasher_update(&hasher, hash->data(), sizeof(CState::hash_t));
					}
				}

				blake3_hasher_finalize(&hasher, reinterpret_cast<uint8_t*>(state->outHash.data()), sizeof(CState::hash_t)); // finalize output hash for whole image given all hashes
			}

			return true;
		}
		static inline bool execute(state_type* state)
		{
			return execute(core::execution::seq, state);
		}

		private:

			struct ScratchMap
			{
				asset::SBufferRange<asset::ICPUBuffer> hashes; // hashes, single hash is obtained from given miplevel & layer, full hash for an image is a hash of this hash buffer
				asset::SBufferRange<asset::ICPUBuffer> hashers; // hashers, used to produce a hash
				asset::SBufferRange<asset::ICPUBuffer> flatten; // tightly packed texels from input, no memory gaps
			};
};

} // end namespace asset
} // end namespace nbl

#endif // __NBL_ASSET_C_FLATTEN_REGIONS_STREAM_HASH_IMAGE_FILTER_H_INCLUDED__