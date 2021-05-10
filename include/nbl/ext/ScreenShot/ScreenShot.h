// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXT_SCREEN_SHOT_INCLUDED_
#define _NBL_EXT_SCREEN_SHOT_INCLUDED_

#include "nabla.h"

#include "../../../../source/Nabla/COpenGLBuffer.h"
#include "../../../../source/Nabla/COpenGLExtensionHandler.h"
#include "nbl/asset/interchange/IImageAssetHandlerBase.h"

namespace nbl
{
	namespace ext
	{
		namespace ScreenShot
		{
			/*
				Creates useful FBO that may be used for instance 
				to fetch default render target color data attachment 
				after rendering scene.
				The usage is following:
				- create frame buffer using the function
				- blit framebuffers, driver->blitRenderTargets(nullptr, frameBuffer, false, false);
				Note that in the call above we don't want to copy depth and stencil buffer, but event though default
				FBO contains depth buffer.
				- pass frame buffer to performScreenShot(video::IFrameBuffer*)
				Notes:
				- color buffer is placed under video::EFAP_COLOR_ATTACHMENT0 attachment
				- depth buffer is placed under video::EFAP_DEPTH_ATTACHMENT attachment
			*/

			nbl::video::IFrameBuffer* createDefaultFBOForScreenshoting(core::smart_refctd_ptr<IrrlichtDevice> device, asset::E_FORMAT colorFormat=asset::EF_R8G8B8A8_SRGB)
			{
				auto driver = device->getVideoDriver();

				auto createAttachement = [&](bool colorBuffer)
				{
					asset::ICPUImage::SCreationParams imgInfo;
					imgInfo.format = colorBuffer ? colorFormat:asset::EF_D24_UNORM_S8_UINT;
					imgInfo.type = asset::ICPUImage::ET_2D;
					imgInfo.extent.width = driver->getScreenSize().Width;
					imgInfo.extent.height = driver->getScreenSize().Height;
					imgInfo.extent.depth = 1u;
					imgInfo.mipLevels = 1u;
					imgInfo.arrayLayers = 1u;
					imgInfo.samples = asset::ICPUImage::ESCF_1_BIT;
					imgInfo.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);

					auto image = asset::ICPUImage::create(std::move(imgInfo));
					const auto texelFormatBytesize = getTexelOrBlockBytesize(image->getCreationParameters().format);

					auto texelBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(image->getImageDataSizeInBytes());
					auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::ICPUImage::SBufferCopy>>(1u);
					asset::ICPUImage::SBufferCopy& region = regions->front();

					region.imageSubresource.mipLevel = 0u;
					region.imageSubresource.baseArrayLayer = 0u;
					region.imageSubresource.layerCount = 1u;
					region.bufferOffset = 0u;
					region.bufferRowLength = image->getCreationParameters().extent.width;
					region.bufferImageHeight = 0u;
					region.imageOffset = { 0u, 0u, 0u };
					region.imageExtent = image->getCreationParameters().extent;

					image->setBufferAndRegions(std::move(texelBuffer), regions);

					asset::ICPUImageView::SCreationParams imgViewInfo;
					imgViewInfo.image = std::move(image);
					imgViewInfo.format = colorBuffer ? colorFormat:asset::EF_D24_UNORM_S8_UINT;
					imgViewInfo.viewType = asset::IImageView<asset::ICPUImage>::ET_2D;
					imgViewInfo.flags = static_cast<asset::ICPUImageView::E_CREATE_FLAGS>(0u);
					imgViewInfo.subresourceRange.baseArrayLayer = 0u;
					imgViewInfo.subresourceRange.baseMipLevel = 0u;
					imgViewInfo.subresourceRange.layerCount = imgInfo.arrayLayers;
					imgViewInfo.subresourceRange.levelCount = imgInfo.mipLevels;

					auto imageView = asset::ICPUImageView::create(std::move(imgViewInfo));
					auto gpuImageView = driver->getGPUObjectsFromAssets(&imageView.get(), &imageView.get() + 1)->front();

					return std::move(gpuImageView);
				};

				auto gpuImageViewDepthBuffer = createAttachement(false);
				auto gpuImageViewColorBuffer = createAttachement(true);

				auto frameBuffer = driver->addFrameBuffer();
				frameBuffer->attach(video::EFAP_DEPTH_ATTACHMENT, std::move(gpuImageViewDepthBuffer));
				frameBuffer->attach(video::EFAP_COLOR_ATTACHMENT0, std::move(gpuImageViewColorBuffer));

				return frameBuffer;
			};

			/*
				Download mip level image with gpu image usage and save it to IGPUBuffer.
				Because of the fence placed by driver the function stalls the CPU 
				to wait on the GPU to finish, beware of that.
				@see video::IDriverFence
			*/

			//! TODO: HANDLE UNPACK ALIGNMENT
			[[nodiscard]] core::smart_refctd_ptr<video::IDriverFence> downloadImageMipLevel(video::IDriver* driver, video::IGPUImage* source, video::IGPUBuffer* destination, uint32_t sourceMipLevel = 0u, size_t destOffset = 0ull, bool implicitflush = true)
			{
				// will change this, https://github.com/buildaworldnet/IrrlichtBAW/issues/148
				if (isBlockCompressionFormat(source->getCreationParameters().format))
					return nullptr;

				auto extent = source->getMipSize(sourceMipLevel);
				video::IGPUImage::SBufferCopy pRegions[1u] = { {destOffset,extent.x,extent.y,{static_cast<asset::IImage::E_ASPECT_FLAGS>(0u),sourceMipLevel,0u,1u},{0u,0u,0u},{extent.x,extent.y,extent.z}} };
				driver->copyImageToBuffer(source, destination, 1u, pRegions);

				return driver->placeFence(implicitflush);
			}

			/*
				Create a ScreenShot with gpu image usage and save it to a file.
			*/
			bool createScreenShot(video::IVideoDriver* driver, asset::IAssetManager* assetManager, const video::IGPUImageView* gpuImageView, const std::string& outFileName, asset::E_FORMAT convertToFormat=asset::EF_UNKNOWN)
			{
				auto fetchedImageViewParmas = gpuImageView->getCreationParameters();
				auto gpuImage = fetchedImageViewParmas.image;
				auto fetchedImageParams = gpuImage->getCreationParameters();
				auto image = asset::ICPUImage::create(std::move(fetchedImageParams));

				auto texelBufferRowLength = asset::IImageAssetHandlerBase::calcPitchInBlocks(fetchedImageParams.extent.width * asset::getBlockDimensions(fetchedImageParams.format).X, asset::getTexelOrBlockBytesize(fetchedImageParams.format));

				auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::ICPUImage::SBufferCopy>>(1u);
				asset::ICPUImage::SBufferCopy& region = regions->front();

				region.imageSubresource.mipLevel = 0u;
				region.imageSubresource.baseArrayLayer = 0u;
				region.imageSubresource.layerCount = 1u;
				region.bufferOffset = 0u;
				region.bufferRowLength = texelBufferRowLength;
				region.bufferImageHeight = 0u;
				region.imageOffset = { 0u, 0u, 0u };
				region.imageExtent = image->getCreationParameters().extent;

				video::IDriverMemoryBacked::SDriverMemoryRequirements memoryRequirements;
				memoryRequirements.vulkanReqs.alignment = 64u;
				memoryRequirements.vulkanReqs.memoryTypeBits = 0xffffffffu;
				memoryRequirements.memoryHeapLocation = video::IDriverMemoryAllocation::ESMT_NOT_DEVICE_LOCAL;
				memoryRequirements.mappingCapability = video::IDriverMemoryAllocation::EMCF_CAN_MAP_FOR_READ | video::IDriverMemoryAllocation::EMCF_COHERENT | video::IDriverMemoryAllocation::EMCF_CACHED;
				memoryRequirements.vulkanReqs.size = image->getImageDataSizeInBytes();
				auto destinationBuffer = driver->createGPUBufferOnDedMem(memoryRequirements);

				auto mapPointerGetterFence = downloadImageMipLevel(driver, gpuImage.get(), destinationBuffer.get());

				auto destinationBoundMemory = destinationBuffer->getBoundMemory();
				destinationBoundMemory->mapMemoryRange(video::IDriverMemoryAllocation::EMCAF_READ, { 0u, memoryRequirements.vulkanReqs.size });
				auto texelBuffer = core::make_smart_refctd_ptr<asset::CCustomAllocatorCPUBuffer<core::null_allocator<uint8_t>>>(memoryRequirements.vulkanReqs.size, destinationBoundMemory->getMappedPointer(), core::adopt_memory);

				image->setBufferAndRegions(std::move(texelBuffer), regions);
				
				while (mapPointerGetterFence->waitCPU(1000ull, mapPointerGetterFence->canDeferredFlush()) == video::EDFR_TIMEOUT_EXPIRED) {}

				core::smart_refctd_ptr<asset::ICPUImage> convertedImage;
				if (convertToFormat != asset::EF_UNKNOWN)
				{
					auto referenceImageParams = image->getCreationParameters();
					auto referenceBuffer = image->getBuffer();
					auto referenceRegions = image->getRegions();
					auto referenceRegion = referenceRegions.begin();
					const auto newTexelOrBlockByteSize = asset::getTexelOrBlockBytesize(convertToFormat);

					auto newImageParams = referenceImageParams;
					auto newCpuBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(referenceBuffer->getSize() * newTexelOrBlockByteSize);
					auto newRegions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::ICPUImage::SBufferCopy>>(referenceRegions.size());

					for (auto newRegion = newRegions->begin(); newRegion != newRegions->end(); ++newRegion)
					{
						*newRegion = *(referenceRegion++);
						newRegion->bufferOffset = newRegion->bufferOffset * newTexelOrBlockByteSize;
					}

					newImageParams.format = convertToFormat;
					convertedImage = asset::ICPUImage::create(std::move(newImageParams));
					convertedImage->setBufferAndRegions(std::move(newCpuBuffer), newRegions);

					//asset::CConvertFormatImageFilter TODO: use this one instead with a nice dither @Anastazluk, we could also get rid of a lot of code here, since there's a bunch of constraints
					asset::CSwizzleAndConvertImageFilter<> convertFilter;
					asset::CSwizzleAndConvertImageFilter<>::state_type state;

					state.swizzle = {};
					state.inImage = image.get();
					state.outImage = convertedImage.get();
					state.inOffset = { 0, 0, 0 };
					state.inBaseLayer = 0;
					state.outOffset = { 0, 0, 0 };
					state.outBaseLayer = 0;
					//state.dither = ;

					for (auto itr = 0; itr < convertedImage->getCreationParameters().mipLevels; ++itr)
					{
						auto regionWithMipMap = convertedImage->getRegions(itr).begin();

						state.extent = regionWithMipMap->getExtent();
						state.layerCount = regionWithMipMap->imageSubresource.layerCount;
						state.inMipLevel = regionWithMipMap->imageSubresource.mipLevel;
						state.outMipLevel = regionWithMipMap->imageSubresource.mipLevel;

						const bool ok = convertFilter.execute(std::execution::par_unseq,&state);
						assert(ok);
					}
				}
				else
					convertedImage = image;
				auto newCreationParams = convertedImage->getCreationParameters();
				
				asset::ICPUImageView::SCreationParams viewParams;
				viewParams.flags = static_cast<asset::ICPUImageView::E_CREATE_FLAGS>(0u);
				viewParams.image = convertedImage;
				viewParams.format = newCreationParams.format;
				viewParams.viewType = asset::ICPUImageView::ET_2D;
				viewParams.subresourceRange.baseArrayLayer = 0u;
				viewParams.subresourceRange.layerCount = newCreationParams.arrayLayers;
				viewParams.subresourceRange.baseMipLevel = 0u;
				viewParams.subresourceRange.levelCount = newCreationParams.mipLevels;

				auto imageView = asset::ICPUImageView::create(std::move(viewParams));

				auto tryToWrite = [&](asset::IAsset* asset)
				{
					asset::IAssetWriter::SAssetWriteParams wparams(asset);
					return assetManager->writeAsset(outFileName, wparams);
				};

				bool status = tryToWrite(convertedImage.get());
				if (!status)
					status = tryToWrite(imageView.get());

				destinationBoundMemory->unmapMemory();
				return status;

			}
			inline bool createScreenShot(core::smart_refctd_ptr<IrrlichtDevice> device, const video::IGPUImageView* gpuImageView, const std::string& outFileName)
			{
				auto driver = device->getVideoDriver();
				auto assetManager = device->getAssetManager();
				return createScreenShot(driver, assetManager, gpuImageView, outFileName);
			}
		} // namespace ScreenShot
	} // namespace ext
} // namespace nbl

#endif