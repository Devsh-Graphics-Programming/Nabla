// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXT_SCREEN_SHOT_INCLUDED_
#define _NBL_EXT_SCREEN_SHOT_INCLUDED_

#include <nabla.h>

namespace nbl::ext::ScreenShot
{
/*
	Create a ScreenShot with gpu image usage and save it to a file.
	The queue being passed must have TRANSFER capability.

	TODO: Add support for downloading a region of a specific subresource
*/

inline core::smart_refctd_ptr<asset::ICPUImageView> createScreenShot(
	video::ILogicalDevice* logicalDevice,
	video::IGPUQueue* queue,
	video::IGPUSemaphore* semaphore,
	const video::IGPUImageView* gpuImageView,
	const asset::E_ACCESS_FLAGS accessMask,
	const asset::E_IMAGE_LAYOUT imageLayout)
{
	assert(logicalDevice->getPhysicalDevice()->getQueueFamilyProperties().begin()[queue->getFamilyIndex()].queueFlags.value&video::IPhysicalDevice::EQF_TRANSFER_BIT);

	auto fetchedImageViewParmas = gpuImageView->getCreationParameters();
	auto gpuImage = fetchedImageViewParmas.image;
	auto fetchedGpuImageParams = gpuImage->getCreationParameters();

	if(!fetchedGpuImageParams.usage.hasFlags(asset::IImage::EUF_TRANSFER_SRC_BIT))
	{
		assert(false);
		return nullptr;
	}

	if (asset::isBlockCompressionFormat(fetchedGpuImageParams.format))
		return nullptr;

	core::smart_refctd_ptr<video::IGPUBuffer> gpuTexelBuffer;
	
	core::smart_refctd_ptr<video::IGPUCommandBuffer> gpuCommandBuffer;
	{
		// commandbuffer should refcount the pool, so it should be 100% legal to drop at the end of the scope
		auto gpuCommandPool = logicalDevice->createCommandPool(queue->getFamilyIndex(),static_cast<video::IGPUCommandPool::E_CREATE_FLAGS>(0u));
		logicalDevice->createCommandBuffers(gpuCommandPool.get(), video::IGPUCommandBuffer::EL_PRIMARY, 1u, &gpuCommandBuffer);
		assert(gpuCommandBuffer);
	}
	gpuCommandBuffer->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
	{
		auto extent = gpuImage->getMipSize();

		const uint32_t mipLevelToScreenshot = fetchedImageViewParmas.subresourceRange.baseMipLevel;

		video::IGPUImage::SBufferCopy region = {};
		region.imageSubresource.aspectMask = fetchedImageViewParmas.subresourceRange.aspectMask; 
		region.imageSubresource.mipLevel = mipLevelToScreenshot;
		region.imageSubresource.baseArrayLayer = fetchedImageViewParmas.subresourceRange.baseArrayLayer;
		region.imageSubresource.layerCount = fetchedImageViewParmas.subresourceRange.layerCount;
		region.imageExtent = { extent.x, extent.y, extent.z };

		video::IGPUBuffer::SCreationParams bufferCreationParams = {};
		bufferCreationParams.usage = asset::IBuffer::EUF_TRANSFER_DST_BIT;

		auto deviceLocalGPUMemoryReqs = logicalDevice->getDownStreamingMemoryReqs();
		deviceLocalGPUMemoryReqs.vulkanReqs.size = extent.x*extent.y*extent.z*asset::getTexelOrBlockBytesize(fetchedGpuImageParams.format);
		gpuTexelBuffer = logicalDevice->createGPUBufferOnDedMem(bufferCreationParams,deviceLocalGPUMemoryReqs);

		video::IGPUCommandBuffer::SImageMemoryBarrier barrier = {};
		barrier.barrier.srcAccessMask = accessMask;
		barrier.barrier.dstAccessMask = asset::EAF_TRANSFER_READ_BIT;
		barrier.oldLayout = imageLayout;
		barrier.newLayout = asset::EIL_TRANSFER_SRC_OPTIMAL;
		barrier.srcQueueFamilyIndex = ~0u;
		barrier.dstQueueFamilyIndex = ~0u;
		barrier.image = gpuImage;
		barrier.subresourceRange.aspectMask = fetchedImageViewParmas.subresourceRange.aspectMask;
		barrier.subresourceRange.baseMipLevel = mipLevelToScreenshot;
		barrier.subresourceRange.levelCount = 1u;
		barrier.subresourceRange.baseArrayLayer = fetchedImageViewParmas.subresourceRange.baseArrayLayer;
		barrier.subresourceRange.layerCount = fetchedImageViewParmas.subresourceRange.layerCount;
		gpuCommandBuffer->pipelineBarrier(
			asset::EPSF_TOP_OF_PIPE_BIT,
			asset::EPSF_TRANSFER_BIT,
			static_cast<asset::E_DEPENDENCY_FLAGS>(0u),
			0u, nullptr,
			0u, nullptr,
			1u, &barrier);

		gpuCommandBuffer->copyImageToBuffer(gpuImage.get(),asset::EIL_TRANSFER_SRC_OPTIMAL,gpuTexelBuffer.get(),1,&region);

		barrier.barrier.srcAccessMask = asset::EAF_TRANSFER_READ_BIT;
		barrier.barrier.dstAccessMask = static_cast<asset::E_ACCESS_FLAGS>(0u);
		barrier.oldLayout = asset::EIL_TRANSFER_SRC_OPTIMAL;
		barrier.newLayout = imageLayout;
		gpuCommandBuffer->pipelineBarrier(
			asset::EPSF_TRANSFER_BIT,
			asset::EPSF_BOTTOM_OF_PIPE_BIT,
			static_cast<asset::E_DEPENDENCY_FLAGS>(0u),
			0u, nullptr,
			0u, nullptr,
			1u, &barrier);
	}
	gpuCommandBuffer->end();

	auto fence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

	video::IGPUQueue::SSubmitInfo info;
	info.commandBufferCount = 1u;
	info.commandBuffers = &gpuCommandBuffer.get();
	info.pSignalSemaphores = nullptr;
	info.signalSemaphoreCount = 0u;
	info.pWaitSemaphores = nullptr;
	info.waitSemaphoreCount = 0u;
	auto stageflags = asset::EPSF_ALL_COMMANDS_BIT; // assume the image we're trying to download could be touched by anything before (host manipulation is implicitly visibile because of submit's guarantees)
	info.pWaitDstStageMask = &stageflags;
	queue->submit(1u, &info, fence.get());

	if (!logicalDevice->blockForFences(1u, &fence.get()))
		return nullptr;

	core::smart_refctd_ptr<asset::ICPUImageView> cpuImageView;
	{
		const auto gpuTexelBufferSize = gpuTexelBuffer->getSize(); // If you get validation errors from the `invalidateMappedMemoryRanges` we need to expose VK_WHOLE_BUFFER equivalent constant
		video::IDriverMemoryAllocation::MappedMemoryRange mappedMemoryRange(gpuTexelBuffer->getBoundMemory(),0u,gpuTexelBufferSize);
		logicalDevice->mapMemory(mappedMemoryRange, video::IDriverMemoryAllocation::EMCAF_READ);

		if (gpuTexelBuffer->getBoundMemory()->haveToMakeVisible())
			logicalDevice->invalidateMappedMemoryRanges(1u,&mappedMemoryRange);

		auto cpuNewImage = asset::ICPUImage::create(std::move(fetchedGpuImageParams));
		auto texelBufferRowLength = asset::IImageAssetHandlerBase::calcPitchInBlocks(fetchedGpuImageParams.extent.width * asset::getBlockDimensions(fetchedGpuImageParams.format).X, asset::getTexelOrBlockBytesize(fetchedGpuImageParams.format));

		auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::ICPUImage::SBufferCopy>>(1u);
		asset::ICPUImage::SBufferCopy& region = regions->front();

		region.imageSubresource.mipLevel = 0u;
		region.imageSubresource.baseArrayLayer = 0u;
		region.imageSubresource.layerCount = 1u;
		region.bufferOffset = 0u;
		region.bufferRowLength = texelBufferRowLength;
		region.bufferImageHeight = 0u;
		region.imageOffset = { 0u, 0u, 0u };
		region.imageExtent = cpuNewImage->getCreationParameters().extent;

		auto cpuNewTexelBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(gpuTexelBufferSize);
		{
			memcpy(cpuNewTexelBuffer->getPointer(), gpuTexelBuffer->getBoundMemory()->getMappedPointer(), gpuTexelBuffer->getSize());
		}
		cpuNewImage->setBufferAndRegions(core::smart_refctd_ptr(cpuNewTexelBuffer), regions);
		logicalDevice->unmapMemory(gpuTexelBuffer->getBoundMemory());
		{
			auto newCreationParams = cpuNewImage->getCreationParameters();

			asset::ICPUImageView::SCreationParams viewParams;
			viewParams.flags = static_cast<asset::ICPUImageView::E_CREATE_FLAGS>(0u);
			viewParams.image = cpuNewImage;
			viewParams.format = newCreationParams.format;
			viewParams.viewType = asset::ICPUImageView::ET_2D;
			viewParams.subresourceRange.baseArrayLayer = 0u;
			viewParams.subresourceRange.layerCount = newCreationParams.arrayLayers;
			viewParams.subresourceRange.baseMipLevel = 0u;
			viewParams.subresourceRange.levelCount = newCreationParams.mipLevels;

			cpuImageView = asset::ICPUImageView::create(std::move(viewParams));
		}
	}
	return cpuImageView;
}

inline bool createScreenShot(
	video::ILogicalDevice* logicalDevice,
	video::IGPUQueue* queue,
	video::IGPUSemaphore* semaphore,
	const video::IGPUImageView* gpuImageView,
	asset::IAssetManager* assetManager,
	system::IFile* outFile,
	const asset::E_ACCESS_FLAGS accessMask,
	const asset::E_IMAGE_LAYOUT imageLayout)
{
	assert(outFile->getFlags()&system::IFile::ECF_WRITE);
	auto cpuImageView = createScreenShot(logicalDevice,queue,semaphore,gpuImageView,accessMask,imageLayout);
	asset::IAssetWriter::SAssetWriteParams writeParams(cpuImageView.get());
	return assetManager->writeAsset(outFile,writeParams);
}

inline bool createScreenShot(
	video::ILogicalDevice* logicalDevice,
	video::IGPUQueue* queue,
	video::IGPUSemaphore* semaphore,
	const video::IGPUImageView* gpuImageView,
	asset::IAssetManager* assetManager,
	const std::filesystem::path& filename,
	const asset::E_IMAGE_LAYOUT imageLayout,
	const asset::E_ACCESS_FLAGS accessMask = asset::EAF_ALL_IMAGE_ACCESSES_DEVSH)
{
	auto cpuImageView = createScreenShot(logicalDevice,queue,semaphore,gpuImageView,accessMask,imageLayout);
	asset::IAssetWriter::SAssetWriteParams writeParams(cpuImageView.get());
	return assetManager->writeAsset(filename.string(),writeParams); // TODO: Use std::filesystem::path
}

} // namespace nbl::ext::ScreenShot

#endif

#ifdef OLD_CODE // code from `master` branch:
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

				auto correctedScreenShotTexelBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(memoryRequirements.vulkanReqs.size);
				bool flipImage = true;
				if(flipImage)
				{
					auto extent = gpuImage->getMipSize(0u);
					uint32_t rowByteSize = extent.x * asset::getTexelOrBlockBytesize(gpuImage->getCreationParameters().format);
					for(uint32_t y = 0; y < extent.y; ++y)
					{
						uint32_t flipped_y = extent.y - y - 1;
						memcpy(reinterpret_cast<uint8_t*>(correctedScreenShotTexelBuffer->getPointer()) + rowByteSize * y, reinterpret_cast<uint8_t*>(destinationBoundMemory->getMappedPointer()) + rowByteSize * flipped_y, rowByteSize);
					}
				}
				else
				{
					memcpy(correctedScreenShotTexelBuffer->getPointer(), destinationBoundMemory->getMappedPointer(), memoryRequirements.vulkanReqs.size);
				}

				destinationBoundMemory->unmapMemory();

				image->setBufferAndRegions(std::move(correctedScreenShotTexelBuffer), regions);
				
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

						const bool ok = convertFilter.execute(core::execution::par_unseq,&state);
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

				return status;

			}
#endif