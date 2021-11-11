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
		logicalDevice->mapMemory(mappedMemoryRange,video::IDriverMemoryAllocation::EMCAF_READ);

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