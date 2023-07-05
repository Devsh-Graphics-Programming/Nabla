// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXT_SCREEN_SHOT_INCLUDED_
#define _NBL_EXT_SCREEN_SHOT_INCLUDED_

#include <nabla.h>

namespace nbl::ext::ScreenShot
{

using namespace nbl::asset;
using namespace nbl::video;
/*
	Create a ScreenShot with gpu image usage and save it to a file.
	The queue being passed must have TRANSFER capability.

	TODO: Add support for downloading a region of a specific subresource
*/

inline core::smart_refctd_ptr<ICPUImageView> createScreenShot(
	ILogicalDevice* logicalDevice,
	IQueue* queue,
	IGPUSemaphore* semaphore,
	const IGPUImageView* gpuImageView,
	const ACCESS_FLAGS accessMask,
	const IImage::LAYOUT imageLayout)
{
	assert(logicalDevice->getPhysicalDevice()->getQueueFamilyProperties().begin()[queue->getFamilyIndex()].queueFlags.value&IQueue::FAMILY_FLAGS::TRANSFER_BIT);

	auto fetchedImageViewParmas = gpuImageView->getCreationParameters();
	auto gpuImage = fetchedImageViewParmas.image;
	auto fetchedGpuImageParams = gpuImage->getCreationParameters();

	if(!fetchedGpuImageParams.usage.hasFlags(IImage::EUF_TRANSFER_SRC_BIT))
	{
		assert(false);
		return nullptr;
	}

	if (isBlockCompressionFormat(fetchedGpuImageParams.format))
		return nullptr;

	core::smart_refctd_ptr<IGPUBuffer> gpuTexelBuffer;
	
	core::smart_refctd_ptr<IGPUCommandBuffer> gpuCommandBuffer;
	{
		// commandbuffer should refcount the pool, so it should be 100% legal to drop at the end of the scope
		auto gpuCommandPool = logicalDevice->createCommandPool(queue->getFamilyIndex(),IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);
		logicalDevice->createCommandBuffers(gpuCommandPool.get(), IGPUCommandBuffer::LEVEL::PRIMARY, 1u, &gpuCommandBuffer);
		assert(gpuCommandBuffer);
	}
	gpuCommandBuffer->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
	{
		auto extent = gpuImage->getMipSize();

		const uint32_t mipLevelToScreenshot = fetchedImageViewParmas.subresourceRange.baseMipLevel;

		IGPUImage::SBufferCopy region = {};
		region.imageSubresource.aspectMask = fetchedImageViewParmas.subresourceRange.aspectMask; 
		region.imageSubresource.mipLevel = mipLevelToScreenshot;
		region.imageSubresource.baseArrayLayer = fetchedImageViewParmas.subresourceRange.baseArrayLayer;
		region.imageSubresource.layerCount = fetchedImageViewParmas.subresourceRange.layerCount;
		region.imageExtent = { extent.x, extent.y, extent.z };

		IGPUBuffer::SCreationParams bufferCreationParams = {};
		bufferCreationParams.size = extent.x*extent.y*extent.z*getTexelOrBlockBytesize(fetchedGpuImageParams.format);
		bufferCreationParams.usage = IBuffer::EUF_TRANSFER_DST_BIT;
		gpuTexelBuffer = logicalDevice->createBuffer(std::move(bufferCreationParams));
		auto gpuTexelBufferMemReqs = gpuTexelBuffer->getMemoryReqs();
		gpuTexelBufferMemReqs.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDownStreamingMemoryTypeBits();
		auto gpuTexelBufferMem = logicalDevice->allocate(gpuTexelBufferMemReqs, gpuTexelBuffer.get());

		IGPUCommandBuffer::SPipelineBarrierDependencyInfo info = {};
		decltype(info)::image_barrier_t barrier = {};
		info.imgBarrierCount = 1u;
		info.imgBarriers = &barrier;

		{
			barrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS;
			barrier.barrier.dep.srcAccessMask = accessMask;
			barrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT;
			barrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::TRANSFER_READ_BIT;
			barrier.oldLayout = imageLayout;
			barrier.newLayout = IImage::LAYOUT::TRANSFER_SRC_OPTIMAL;
			barrier.image = gpuImage;
			barrier.subresourceRange.aspectMask = fetchedImageViewParmas.subresourceRange.aspectMask;
			barrier.subresourceRange.baseMipLevel = mipLevelToScreenshot;
			barrier.subresourceRange.levelCount = 1u;
			barrier.subresourceRange.baseArrayLayer = fetchedImageViewParmas.subresourceRange.baseArrayLayer;
			barrier.subresourceRange.layerCount = fetchedImageViewParmas.subresourceRange.layerCount;
			gpuCommandBuffer->pipelineBarrier(EDF_NONE,info);
		}
		gpuCommandBuffer->copyImageToBuffer(gpuImage.get(),IImage::LAYOUT::TRANSFER_SRC_OPTIMAL,gpuTexelBuffer.get(),1,&region);
		{
			barrier.barrier.dep.srcStageMask = barrier.barrier.dep.dstStageMask;
			barrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::NONE;
			barrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS;
			barrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::NONE;
			barrier.oldLayout = barrier.newLayout;
			barrier.newLayout = imageLayout;
			gpuCommandBuffer->pipelineBarrier(EDF_NONE,info);
		}
	}
	gpuCommandBuffer->end();

	auto fence = logicalDevice->createFence(static_cast<IGPUFence::E_CREATE_FLAGS>(0));

	IQueue::SSubmitInfo info;
	info.commandBufferCount = 1u;
	info.commandBuffers = &gpuCommandBuffer.get();
	info.pSignalSemaphores = nullptr;
	info.signalSemaphoreCount = 0u;
	info.pWaitSemaphores = &semaphore;
	info.waitSemaphoreCount = semaphore ? 1u:0u;
	auto stageflags = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS; // assume the image we're trying to download could be touched by anything before (host manipulation is implicitly visibile because of submit's guarantees)
	info.pWaitDstStageMask = &stageflags;
	queue->submit(1u, &info, fence.get());

	if (!logicalDevice->blockForFences(1u, &fence.get()))
		return nullptr;

	core::smart_refctd_ptr<ICPUImageView> cpuImageView;
	{
		const auto gpuTexelBufferSize = gpuTexelBuffer->getSize(); // If you get validation errors from the `invalidateMappedMemoryRanges` we need to expose VK_WHOLE_BUFFER equivalent constant
		IDeviceMemoryAllocation::MappedMemoryRange mappedMemoryRange(gpuTexelBuffer->getBoundMemory(),0u,gpuTexelBufferSize);
		logicalDevice->mapMemory(mappedMemoryRange, IDeviceMemoryAllocation::EMCAF_READ);

		if (gpuTexelBuffer->getBoundMemory()->haveToMakeVisible())
			logicalDevice->invalidateMappedMemoryRanges(1u,&mappedMemoryRange);

		auto cpuNewImage = ICPUImage::create(std::move(fetchedGpuImageParams));

		auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(1u);
		ICPUImage::SBufferCopy& region = regions->front();

		region.imageSubresource.aspectMask = IImage::EAF_COLOR_BIT;
		region.imageSubresource.mipLevel = 0u;
		region.imageSubresource.baseArrayLayer = 0u;
		region.imageSubresource.layerCount = 1u;
		region.bufferOffset = 0u;
		region.bufferRowLength = fetchedGpuImageParams.extent.width;
		region.bufferImageHeight = 0u;
		region.imageOffset = { 0u, 0u, 0u };
		region.imageExtent = cpuNewImage->getCreationParameters().extent;

		auto cpuNewTexelBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(gpuTexelBufferSize);
		{
			memcpy(cpuNewTexelBuffer->getPointer(), gpuTexelBuffer->getBoundMemory()->getMappedPointer(), gpuTexelBuffer->getSize());
		}
		cpuNewImage->setBufferAndRegions(core::smart_refctd_ptr(cpuNewTexelBuffer), regions);
		logicalDevice->unmapMemory(gpuTexelBuffer->getBoundMemory());
		{
			auto newCreationParams = cpuNewImage->getCreationParameters();

			ICPUImageView::SCreationParams viewParams = {};
			viewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
			viewParams.image = cpuNewImage;
			viewParams.format = newCreationParams.format;
			viewParams.viewType = ICPUImageView::ET_2D;
			viewParams.subresourceRange.baseArrayLayer = 0u;
			viewParams.subresourceRange.layerCount = newCreationParams.arrayLayers;
			viewParams.subresourceRange.baseMipLevel = 0u;
			viewParams.subresourceRange.levelCount = newCreationParams.mipLevels;

			cpuImageView = ICPUImageView::create(std::move(viewParams));
		}
	}
	return cpuImageView;
}

inline bool createScreenShot(
	ILogicalDevice* logicalDevice,
	IQueue* queue,
	IGPUSemaphore* semaphore,
	const IGPUImageView* gpuImageView,
	IAssetManager* assetManager,
	system::IFile* outFile,
	const ACCESS_FLAGS accessMask,
	const IImage::LAYOUT imageLayout)
{
	assert(outFile->getFlags()&system::IFile::ECF_WRITE);
	auto cpuImageView = createScreenShot(logicalDevice,queue,semaphore,gpuImageView,accessMask,imageLayout);
	IAssetWriter::SAssetWriteParams writeParams(cpuImageView.get());
	return assetManager->writeAsset(outFile,writeParams);
}

inline bool createScreenShot(
	ILogicalDevice* logicalDevice,
	IQueue* queue,
	IGPUSemaphore* semaphore,
	const IGPUImageView* gpuImageView,
	IAssetManager* assetManager,
	const std::filesystem::path& filename,
	const IImage::LAYOUT imageLayout,
	const ACCESS_FLAGS accessMask = ACCESS_FLAGS::MEMORY_WRITE_BITS)
{
	auto cpuImageView = createScreenShot(logicalDevice,queue,semaphore,gpuImageView,accessMask,imageLayout);
	IAssetWriter::SAssetWriteParams writeParams(cpuImageView.get());
	return assetManager->writeAsset(filename.string(),writeParams); // TODO: Use std::filesystem::path
}

} // namespace nbl::ext::ScreenShot

#endif

#ifdef OLD_CODE // code from old `ditt` branch:
			/*
				Download mip level image with gpu image usage and save it to IGPUBuffer.
				Because of the fence placed by driver the function stalls the CPU 
				to wait on the GPU to finish, beware of that.
				@see IDriverFence
			*/

			//! TODO: HANDLE UNPACK ALIGNMENT
			[[nodiscard]] core::smart_refctd_ptr<IDriverFence> downloadImageMipLevel(IDriver* driver, IGPUImage* source, IGPUBuffer* destination, uint32_t sourceMipLevel = 0u, size_t destOffset = 0ull, bool implicitflush = true)
			{
				// will change this, https://github.com/buildaworldnet/IrrlichtBAW/issues/148
				if (isBlockCompressionFormat(source->getCreationParameters().format))
					return nullptr;

				auto extent = source->getMipSize(sourceMipLevel);
				IGPUImage::SBufferCopy pRegions[1u] = { {destOffset,extent.x,extent.y,{static_cast<IImage::E_ASPECT_FLAGS>(0u),sourceMipLevel,0u,1u},{0u,0u,0u},{extent.x,extent.y,extent.z}} };
				driver->copyImageToBuffer(source, destination, 1u, pRegions);

				return driver->placeFence(implicitflush);
			}

			/*
				Create a ScreenShot with gpu image usage and save it to a file.
			*/
			bool createScreenShot(IVideoDriver* driver, IAssetManager* assetManager, const IGPUImageView* gpuImageView, const std::string& outFileName, E_FORMAT convertToFormat=EF_UNKNOWN)
			{
				auto fetchedImageViewParmas = gpuImageView->getCreationParameters();
				auto gpuImage = fetchedImageViewParmas.image;
				auto fetchedImageParams = gpuImage->getCreationParameters();
				auto image = ICPUImage::create(std::move(fetchedImageParams));

				auto texelBufferRowLength = IImageAssetHandlerBase::calcPitchInBlocks(fetchedImageParams.extent.width * getBlockDimensions(fetchedImageParams.format).X, getTexelOrBlockBytesize(fetchedImageParams.format));

				auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(1u);
				ICPUImage::SBufferCopy& region = regions->front();

				region.imageSubresource.mipLevel = 0u;
				region.imageSubresource.baseArrayLayer = 0u;
				region.imageSubresource.layerCount = 1u;
				region.bufferOffset = 0u;
				region.bufferRowLength = texelBufferRowLength;
				region.bufferImageHeight = 0u;
				region.imageOffset = { 0u, 0u, 0u };
				region.imageExtent = image->getCreationParameters().extent;

				IDeviceMemoryBacked::SDeviceMemoryRequirements memoryRequirements;
				memoryRequirements.vulkanReqs.alignment = 64u;
				memoryRequirements.vulkanReqs.memoryTypeBits = 0xffffffffu;
				memoryRequirements.memoryHeapLocation = IDeviceMemoryAllocation::ESMT_NOT_DEVICE_LOCAL;
				memoryRequirements.mappingCapability = IDeviceMemoryAllocation::EMCF_CAN_MAP_FOR_READ | IDeviceMemoryAllocation::EMCF_COHERENT | IDeviceMemoryAllocation::EMCF_CACHED;
				memoryRequirements.vulkanReqs.size = image->getImageDataSizeInBytes();
				auto destinationBuffer = driver->createGPUBufferOnDedMem(memoryRequirements);

				auto mapPointerGetterFence = downloadImageMipLevel(driver, gpuImage.get(), destinationBuffer.get());

				auto destinationBoundMemory = destinationBuffer->getBoundMemory();
				destinationBoundMemory->mapMemoryRange(IDeviceMemoryAllocation::EMCAF_READ, { 0u, memoryRequirements.vulkanReqs.size });

				auto correctedScreenShotTexelBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(memoryRequirements.vulkanReqs.size);
				bool flipImage = true;
				if(flipImage)
				{
					auto extent = gpuImage->getMipSize(0u);
					uint32_t rowByteSize = extent.x * getTexelOrBlockBytesize(gpuImage->getCreationParameters().format);
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
				
				while (mapPointerGetterFence->waitCPU(1000ull, mapPointerGetterFence->canDeferredFlush()) == EDFR_TIMEOUT_EXPIRED) {}

				core::smart_refctd_ptr<ICPUImage> convertedImage;
				if (convertToFormat != EF_UNKNOWN)
				{
					auto referenceImageParams = image->getCreationParameters();
					auto referenceBuffer = image->getBuffer();
					auto referenceRegions = image->getRegions();
					auto referenceRegion = referenceRegions.begin();
					const auto newTexelOrBlockByteSize = getTexelOrBlockBytesize(convertToFormat);

					auto newImageParams = referenceImageParams;
					auto newCpuBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(referenceBuffer->getSize() * newTexelOrBlockByteSize);
					auto newRegions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(referenceRegions.size());

					for (auto newRegion = newRegions->begin(); newRegion != newRegions->end(); ++newRegion)
					{
						*newRegion = *(referenceRegion++);
						newRegion->bufferOffset = newRegion->bufferOffset * newTexelOrBlockByteSize;
					}

					newImageParams.format = convertToFormat;
					convertedImage = ICPUImage::create(std::move(newImageParams));
					convertedImage->setBufferAndRegions(std::move(newCpuBuffer), newRegions);

					//CConvertFormatImageFilter TODO: use this one instead with a nice dither @Anastazluk, we could also get rid of a lot of code here, since there's a bunch of constraints
					CSwizzleAndConvertImageFilter<> convertFilter;
					CSwizzleAndConvertImageFilter<>::state_type state;

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
				
				ICPUImageView::SCreationParams viewParams;
				viewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
				viewParams.image = convertedImage;
				viewParams.format = newCreationParams.format;
				viewParams.viewType = ICPUImageView::ET_2D;
				viewParams.subresourceRange.baseArrayLayer = 0u;
				viewParams.subresourceRange.layerCount = newCreationParams.arrayLayers;
				viewParams.subresourceRange.baseMipLevel = 0u;
				viewParams.subresourceRange.levelCount = newCreationParams.mipLevels;

				auto imageView = ICPUImageView::create(std::move(viewParams));

				auto tryToWrite = [&](IAsset* asset)
				{
					IAssetWriter::SAssetWriteParams wparams(asset);
					return assetManager->writeAsset(outFileName, wparams);
				};

				bool status = tryToWrite(convertedImage.get());
				if (!status)
					status = tryToWrite(imageView.get());

				return status;

			}
#endif