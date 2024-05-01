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

	TODO (Mihailo): Add support for downloading a region of a specific subresource
*/

#if 0 // TODO (Mihailo): port
inline core::smart_refctd_ptr<ICPUImageView> createScreenShot(
	ILogicalDevice* logicalDevice,
	IQueue* queue,
	IGPUSemaphore* semaphore,
	const IGPUImageView* gpuImageView,
	const ACCESS_FLAGS accessMask,
	const IImage::LAYOUT imageLayout)
{
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
#endif

} // namespace nbl::ext::ScreenShot

#endif