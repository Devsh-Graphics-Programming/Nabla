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

	TODO (Ali): Add support for downloading a region of a specific subresource
*/


inline core::smart_refctd_ptr<ICPUImageView> createScreenShot(
	ILogicalDevice* logicalDevice,
	IQueue* queue,
	ISemaphore* semaphore,
	const IGPUImageView* gpuImageView,
	const ACCESS_FLAGS accessMask,
	const IImage::LAYOUT imageLayout)
{
	{
		const auto queueFlags = logicalDevice->getPhysicalDevice()->getQueueFamilyProperties().begin()[queue->getFamilyIndex()].queueFlags;
		const auto required = core::bitflag<IQueue::FAMILY_FLAGS>(IQueue::FAMILY_FLAGS::TRANSFER_BIT) | IQueue::FAMILY_FLAGS::GRAPHICS_BIT | IQueue::FAMILY_FLAGS::COMPUTE_BIT;
		if (!queueFlags.hasAnyFlag(required))
			logicalDevice->getLogger()->log("ScreenShot: queue family %u lacks transfer/graphics/compute flags; continuing anyway.", system::ILogger::ELL_WARNING, queue->getFamilyIndex());
	}

	auto fetchedImageViewParmas = gpuImageView->getCreationParameters();
	auto gpuImage = fetchedImageViewParmas.image;
	auto fetchedGpuImageParams = gpuImage->getCreationParameters();

	if(!fetchedGpuImageParams.usage.hasFlags(IImage::EUF_TRANSFER_SRC_BIT))
	{
		if (auto* logger = logicalDevice->getLogger())
			logger->log("ScreenShot: source image missing TRANSFER_SRC usage.", system::ILogger::ELL_ERROR);
		return nullptr;
	}

	if (isBlockCompressionFormat(fetchedGpuImageParams.format))
	{
		if (auto* logger = logicalDevice->getLogger())
			logger->log("ScreenShot: block-compressed formats are not supported.", system::ILogger::ELL_ERROR);
		return nullptr;
	}

	core::smart_refctd_ptr<IGPUBuffer> gpuTexelBuffer;
	
	core::smart_refctd_ptr<IGPUCommandBuffer> gpuCommandBuffer;
	{
		// commandbuffer should refcount the pool, so it should be 100% legal to drop at the end of the scope
		auto gpuCommandPool = logicalDevice->createCommandPool(queue->getFamilyIndex(),IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);
		if (!gpuCommandPool)
		{
			if (auto* logger = logicalDevice->getLogger())
				logger->log("ScreenShot: failed to create command pool.", system::ILogger::ELL_ERROR);
			return nullptr;
		}
		gpuCommandPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &gpuCommandBuffer);
		if (!gpuCommandBuffer)
		{
			if (auto* logger = logicalDevice->getLogger())
				logger->log("ScreenShot: failed to create command buffer.", system::ILogger::ELL_ERROR);
			return nullptr;
		}
	}
	if (auto* logger = logicalDevice->getLogger())
		logger->log("ScreenShot: recording command buffer.", system::ILogger::ELL_INFO);
	if (!gpuCommandBuffer->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT))
	{
		if (auto* logger = logicalDevice->getLogger())
			logger->log("ScreenShot: failed to begin command buffer.", system::ILogger::ELL_ERROR);
		return nullptr;
	}
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
		if (!gpuTexelBuffer)
		{
			if (auto* logger = logicalDevice->getLogger())
				logger->log("ScreenShot: failed to create GPU texel buffer.", system::ILogger::ELL_ERROR);
			return nullptr;
		}
		auto gpuTexelBufferMemReqs = gpuTexelBuffer->getMemoryReqs();
		gpuTexelBufferMemReqs.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDownStreamingMemoryTypeBits();
		if (!gpuTexelBufferMemReqs.memoryTypeBits)
		{
			if (auto* logger = logicalDevice->getLogger())
				logger->log("ScreenShot: no down-streaming memory type for texel buffer.", system::ILogger::ELL_ERROR);
			return nullptr;
		}
		auto gpuTexelBufferMem = logicalDevice->allocate(gpuTexelBufferMemReqs, gpuTexelBuffer.get());
		if (!gpuTexelBufferMem.isValid())
		{
			if (auto* logger = logicalDevice->getLogger())
				logger->log("ScreenShot: failed to allocate texel buffer memory.", system::ILogger::ELL_ERROR);
			return nullptr;
		}

		IGPUCommandBuffer::SPipelineBarrierDependencyInfo info = {};
		decltype(info)::image_barrier_t barrier = {};
		info.imgBarriers = { &barrier, &barrier + 1 };

		{
			barrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS;
			barrier.barrier.dep.srcAccessMask = accessMask;
			barrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT;
			barrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::TRANSFER_READ_BIT;
			barrier.oldLayout = imageLayout;
			barrier.newLayout = IImage::LAYOUT::TRANSFER_SRC_OPTIMAL;
			barrier.image = gpuImage.get();
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
	if (!gpuCommandBuffer->end())
	{
		if (auto* logger = logicalDevice->getLogger())
			logger->log("ScreenShot: failed to end command buffer.", system::ILogger::ELL_ERROR);
		return nullptr;
	}

	auto signalSemaphore = logicalDevice->createSemaphore(0);

	IQueue::SSubmitInfo info;
	IQueue::SSubmitInfo::SCommandBufferInfo cmdBufferInfo{ gpuCommandBuffer.get() };
	IQueue::SSubmitInfo::SSemaphoreInfo signalSemaphoreInfo;
	IQueue::SSubmitInfo::SSemaphoreInfo waitSemaphoreInfo;
	signalSemaphoreInfo.semaphore = signalSemaphore.get();
	signalSemaphoreInfo.value = 1;
	signalSemaphoreInfo.stageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS;
	info.commandBuffers = { &cmdBufferInfo, &cmdBufferInfo + 1 };
	info.signalSemaphores = { &signalSemaphoreInfo, &signalSemaphoreInfo + 1 };

	if (semaphore)
	{
		waitSemaphoreInfo.semaphore = semaphore;
		waitSemaphoreInfo.value = 1;
		waitSemaphoreInfo.stageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS;
		info.waitSemaphores = { &waitSemaphoreInfo, &waitSemaphoreInfo + 1 };
	}

	if (auto* logger = logicalDevice->getLogger())
		logger->log("ScreenShot: submitting copy command buffer.", system::ILogger::ELL_INFO);
	if (queue->submit({ &info, &info + 1}) != IQueue::RESULT::SUCCESS)
	{
		if (auto* logger = logicalDevice->getLogger())
			logger->log("ScreenShot: failed to submit copy command buffer.", system::ILogger::ELL_ERROR);
		return nullptr;
	}

	ISemaphore::SWaitInfo waitInfo{ signalSemaphore.get(), 1u};

	if (auto* logger = logicalDevice->getLogger())
		logger->log("ScreenShot: waiting for copy completion.", system::ILogger::ELL_INFO);
	if (logicalDevice->blockForSemaphores({&waitInfo, &waitInfo + 1}) != ISemaphore::WAIT_RESULT::SUCCESS)
	{
		if (auto* logger = logicalDevice->getLogger())
			logger->log("ScreenShot: failed to wait for copy completion.", system::ILogger::ELL_ERROR);
		return nullptr;
	}

	core::smart_refctd_ptr<ICPUImageView> cpuImageView;
	{
		const auto gpuTexelBufferSize = gpuTexelBuffer->getSize(); // If you get validation errors from the `invalidateMappedMemoryRanges` we need to expose VK_WHOLE_BUFFER equivalent constant
		auto* allocation = gpuTexelBuffer->getBoundMemory().memory;
		if (!allocation)
			return nullptr;

		bool mappedHere = false;
		if (!allocation->getMappedPointer())
		{
			const IDeviceMemoryAllocation::MemoryRange range = { 0u, gpuTexelBufferSize };
			if (!allocation->map(range, IDeviceMemoryAllocation::EMCAF_READ))
			{
				if (auto* logger = logicalDevice->getLogger())
					logger->log("ScreenShot: failed to map texel buffer memory.", system::ILogger::ELL_ERROR);
				return nullptr;
			}
			mappedHere = true;
		}

		ILogicalDevice::MappedMemoryRange mappedMemoryRange(allocation,0u,gpuTexelBufferSize);
		if (allocation->haveToMakeVisible())
		{
			if (auto* logger = logicalDevice->getLogger())
				logger->log("ScreenShot: invalidating mapped range.", system::ILogger::ELL_INFO);
			logicalDevice->invalidateMappedMemoryRanges(1u,&mappedMemoryRange);
		}

		auto cpuNewImage = ICPUImage::create(std::move(fetchedGpuImageParams));
		if (!cpuNewImage)
		{
			if (auto* logger = logicalDevice->getLogger())
				logger->log("ScreenShot: failed to create CPU image.", system::ILogger::ELL_ERROR);
			if (mappedHere)
				allocation->unmap();
			return nullptr;
		}

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

		auto cpuNewTexelBuffer = ICPUBuffer::create({ gpuTexelBufferSize });
		if (!cpuNewTexelBuffer)
		{
			if (auto* logger = logicalDevice->getLogger())
				logger->log("ScreenShot: failed to create CPU buffer.", system::ILogger::ELL_ERROR);
			if (mappedHere)
				allocation->unmap();
			return nullptr;
		}
		if (auto* logger = logicalDevice->getLogger())
			logger->log("ScreenShot: copying GPU data to CPU buffer.", system::ILogger::ELL_INFO);
		{
			memcpy(cpuNewTexelBuffer->getPointer(), allocation->getMappedPointer(), gpuTexelBuffer->getSize());
		}
		cpuNewImage->setBufferAndRegions(core::smart_refctd_ptr(cpuNewTexelBuffer), regions);
		if (mappedHere)
			allocation->unmap();
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
	ISemaphore* semaphore,
	const IGPUImageView* gpuImageView,
	IAssetManager* assetManager,
	system::IFile* outFile,
	const ACCESS_FLAGS accessMask,
	const IImage::LAYOUT imageLayout)
{
	assert(outFile->getFlags()&system::IFile::ECF_WRITE);
	auto cpuImageView = createScreenShot(logicalDevice,queue,semaphore,gpuImageView,accessMask,imageLayout);
	if (!cpuImageView)
	{
		if (auto* logger = logicalDevice->getLogger())
			logger->log("ScreenShot: GPU readback failed, no image to write.", system::ILogger::ELL_ERROR);
		return false;
	}
	IAssetWriter::SAssetWriteParams writeParams(cpuImageView.get());
	return assetManager->writeAsset(outFile,writeParams);
}

inline bool createScreenShot(
	ILogicalDevice* logicalDevice,
	IQueue* queue,
	ISemaphore* semaphore,
	const IGPUImageView* gpuImageView,
	IAssetManager* assetManager,
	const std::filesystem::path& filename,
	const IImage::LAYOUT imageLayout,
	const ACCESS_FLAGS accessMask = ACCESS_FLAGS::MEMORY_WRITE_BITS)
{
	auto cpuImageView = createScreenShot(logicalDevice,queue,semaphore,gpuImageView,accessMask,imageLayout);
	if (!cpuImageView)
	{
		if (auto* logger = logicalDevice->getLogger())
			logger->log("ScreenShot: GPU readback failed, no image to write.", system::ILogger::ELL_ERROR);
		return false;
	}
	IAssetWriter::SAssetWriteParams writeParams(cpuImageView.get());
	return assetManager->writeAsset(filename.string(),writeParams); // TODO: Use std::filesystem::path
}


} // namespace nbl::ext::ScreenShot

#endif
