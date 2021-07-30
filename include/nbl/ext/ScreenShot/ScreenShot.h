// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXT_SCREEN_SHOT_INCLUDED_
#define _NBL_EXT_SCREEN_SHOT_INCLUDED_

#include <nabla.h>
#include "../source/Nabla/CFileSystem.h"

namespace nbl
{
	namespace ext
	{
		namespace ScreenShot
		{
			/*
				Create a ScreenShot with gpu image usage and save it to a file.
				The queue being passed must have TRANSFER capability.
			*/

			inline bool createScreenShot(nbl::video::ILogicalDevice* logicalDevice, nbl::video::IGPUQueue* queue, nbl::video::IGPUSemaphore* semaphore, const nbl::video::IGPUImageView* gpuImageView, const std::string& outFileName)
			{
				auto fetchedImageViewParmas = gpuImageView->getCreationParameters();
				auto gpuImage = fetchedImageViewParmas.image;
				auto fetchedGpuImageParams = gpuImage->getCreationParameters();

				if (nbl::asset::isBlockCompressionFormat(fetchedGpuImageParams.format))
					return false;

				auto gpuCommandPool = logicalDevice->createCommandPool(queue->getFamilyIndex(), nbl::video::IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT);

				video::IDriverMemoryBacked::SDriverMemoryRequirements driverMemoryRequirements;
				core::smart_refctd_ptr<video::IGPUCommandBuffer> gpuCommandBuffer;
				logicalDevice->createCommandBuffers(gpuCommandPool.get(), video::IGPUCommandBuffer::EL_PRIMARY, 1u, &gpuCommandBuffer);
				assert(gpuCommandBuffer);

				nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer> gpuTexelBuffer;

				gpuCommandBuffer->begin(nbl::video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
				{
					auto extent = gpuImage->getMipSize();
					video::IGPUImage::SBufferCopy pRegions[1u] = { {0u,extent.x,extent.y,{static_cast<asset::IImage::E_ASPECT_FLAGS>(0u),0,0u,1u},{0u,0u,0u},{extent.x,extent.y,extent.z}} };

					auto deviceLocalGPUMemoryReqs = logicalDevice->getDownStreamingMemoryReqs();
					deviceLocalGPUMemoryReqs.vulkanReqs.size = extent.x * extent.y * extent.z * nbl::asset::getTexelOrBlockBytesize(fetchedGpuImageParams.format);
					gpuTexelBuffer = logicalDevice->createGPUBufferOnDedMem(deviceLocalGPUMemoryReqs, true);

					gpuCommandBuffer->copyImageToBuffer(gpuImage.get(), nbl::asset::EIL_GENERAL, gpuTexelBuffer.get(), 1, pRegions);
				}
				gpuCommandBuffer->end();

				auto fence = logicalDevice->createFence(static_cast<nbl::video::IGPUFence::E_CREATE_FLAGS>(0));

				video::IGPUQueue::SSubmitInfo info;
				info.commandBufferCount = 1u;
				info.commandBuffers = &gpuCommandBuffer.get();
				info.pSignalSemaphores = nullptr;
				info.signalSemaphoreCount = 0u;
				info.pWaitSemaphores = &semaphore;
				info.waitSemaphoreCount = 1u;
				nbl::asset::E_PIPELINE_STAGE_FLAGS stageflags = nbl::asset::EPSF_TRANSFER_BIT;
				info.pWaitDstStageMask = &stageflags;
				queue->submit(1u, &info, fence.get());

				logicalDevice->waitForFences(1u, &fence.get(), false, 999999999ull);

				nbl::core::smart_refctd_ptr<nbl::asset::ICPUImageView> cpuImageView;
				{
					const auto gpuTexelBufferSize = gpuTexelBuffer->getSize();
					nbl::video::IDriverMemoryAllocation::MappedMemoryRange mappedMemoryRange(gpuTexelBuffer->getBoundMemory(), 0u, gpuTexelBufferSize);
					logicalDevice->mapMemory(mappedMemoryRange, nbl::video::IDriverMemoryAllocation::EMCAF_READ);

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

					auto cpuNewTexelBuffer = nbl::core::make_smart_refctd_ptr<nbl::asset::ICPUBuffer>(gpuTexelBufferSize);
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

				nbl::core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
				{
					nbl::core::smart_refctd_ptr<nbl::io::IFileSystem> fileSystem = nbl::core::make_smart_refctd_ptr<nbl::io::CFileSystem>("");
					assetManager = core::make_smart_refctd_ptr<nbl::asset::IAssetManager>(std::move(fileSystem));
				}

				nbl::asset::IAssetWriter::SAssetWriteParams writeParams(cpuImageView.get());
				return assetManager->writeAsset(outFileName, writeParams);
			}
		} // namespace ScreenShot
	} // namespace ext
} // namespace nbl

#endif