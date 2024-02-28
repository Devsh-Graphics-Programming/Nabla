#include "nbl/video/utilities/ISimpleManagedSurface.h"

using namespace nbl;
using namespace video;


bool ISimpleManagedSurface::immediateBlit(const image_barrier_t& contents, const IQueue::SSubmitInfo::SSemaphoreInfo& waitBeforeBlit, CThreadSafeQueueAdapter* blitAndPresentQueue)
{
	auto& swapchainResources = getSwapchainResources();
	if (!contents.image || swapchainResources.getStatus()!=ISwapchainResources::STATUS::USABLE)
		return false;

	auto* swapchain = swapchainResources.getSwapchain();
	assert(swapchain); // because status is usable
	auto device = const_cast<ILogicalDevice*>(swapchain->getOriginDevice());
	
	// check queue provided
	{
		const auto qFamProps = device->getPhysicalDevice()->getQueueFamilyProperties();
		auto compatibleQueue = [&](const uint8_t qFam)->bool
		{
			return qFamProps[qFam].queueFlags.hasFlags(IQueue::FAMILY_FLAGS::GRAPHICS_BIT) && m_surface->isSupportedForPhysicalDevice(device->getPhysicalDevice(),qFam);
		};
		// pick if default wanted
		if (!blitAndPresentQueue)
		{
			for (uint8_t qFam=0; qFam<ILogicalDevice::MaxQueueFamilies; qFam++)
			{
				const auto qCount = device->getQueueCount(qFam);
				if (qCount && qFamProps[qFam].queueFlags.hasFlags(IQueue::FAMILY_FLAGS::GRAPHICS_BIT))
				{
					// pick a different queue than we'd pick for a regular present
					blitAndPresentQueue = device->getThreadSafeQueue(qFam,0);
					if (blitAndPresentQueue==m_queue)
						blitAndPresentQueue = device->getThreadSafeQueue(qFam,qCount-1);
					break;
				}
			}
		}

		if (!blitAndPresentQueue || compatibleQueue(blitAndPresentQueue->getFamilyIndex()))
			return false;
	}

	// create a different semaphore so we don't increase the acquire counter in `this`
	auto semaphore = device->createSemaphore(0);
	if (!semaphore)
		return false;

	// transient commandbuffer and pool to perform the blit
	core::smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
	{
		auto pool = device->createCommandPool(blitAndPresentQueue->getFamilyIndex(),IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);
		if (!pool || !pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,{&cmdbuf,1}) || !cmdbuf)
			return false;

		if (!cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT))
			return false;
	}
	
	const IQueue::SSubmitInfo::SSemaphoreInfo acquired = {
		.semaphore=semaphore.get(),
		.value=1
	};
	// acquire
	uint32_t imageIndex;
	switch (swapchain->acquireNextImage({.queue=blitAndPresentQueue,.signalSemaphores={&acquired,1}},&imageIndex))
	{
		case ISwapchain::ACQUIRE_IMAGE_RESULT::SUBOPTIMAL: [[fallthrough]];
		case ISwapchain::ACQUIRE_IMAGE_RESULT::SUCCESS:
			break;
		case ISwapchain::ACQUIRE_IMAGE_RESULT::TIMEOUT: [[fallthrough]];
		case ISwapchain::ACQUIRE_IMAGE_RESULT::NOT_READY: // don't throw our swapchain away just because of a timeout XD
			assert(false); // shouldn't happen though cause we use uint64_t::max() as the timeout
			return false;
		case ISwapchain::ACQUIRE_IMAGE_RESULT::OUT_OF_DATE:
			swapchainResources.invalidate();
			return false;
		default:
			swapchainResources.becomeIrrecoverable();
			return false;
	}
	// once image is acquired, WE HAVE TO present it
	bool retval = true;

	// now record the blit commands
	auto acquiredImage = swapchainResources.getImage(imageIndex);
	{
		const auto blitSrcLayout = IGPUImage::LAYOUT::TRANSFER_SRC_OPTIMAL;
		const auto blitDstLayout = IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL;
		IGPUCommandBuffer::SPipelineBarrierDependencyInfo depInfo = {};

		// barrier before
		const image_barrier_t preBarriers[2] = {
			{
				.barrier = {
					.dep = {
						.srcStageMask = asset::PIPELINE_STAGE_FLAGS::NONE, // acquire isn't a stage
						.srcAccessMask = asset::ACCESS_FLAGS::NONE, // performs no accesses
						.dstStageMask = asset::PIPELINE_STAGE_FLAGS::BLIT_BIT,
						.dstAccessMask = asset::ACCESS_FLAGS::TRANSFER_WRITE_BIT
					}
				},
				.image = acquiredImage,
				.subresourceRange = {
					.aspectMask = IGPUImage::EAF_COLOR_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1
				},
				.oldLayout = IGPUImage::LAYOUT::UNDEFINED, // I do not care about previous contents
				.newLayout = blitDstLayout
			},
			{
				.barrier = {
					.dep = {
						.srcStageMask = contents.barrier.dep.srcStageMask,
						.srcAccessMask = contents.barrier.dep.srcAccessMask,
						.dstStageMask = asset::PIPELINE_STAGE_FLAGS::BLIT_BIT,
						.dstAccessMask = asset::ACCESS_FLAGS::TRANSFER_READ_BIT
					},
					.ownershipOp = IGPUCommandBuffer::SOwnershipTransferBarrier::OWNERSHIP_OP::ACQUIRE,
					.otherQueueFamilyIndex = contents.barrier.otherQueueFamilyIndex
				},
				.image = contents.image,
				.subresourceRange = contents.subresourceRange,
				.oldLayout = contents.oldLayout,
				.newLayout = blitSrcLayout
			}
		};
		depInfo.imgBarriers = preBarriers;
		retval &= cmdbuf->pipelineBarrier(asset::EDF_NONE,depInfo);
		
		// TODO: Implement scaling modes other than plain STRETCH, and allow for using subrectangles of the initial contents
		{
			const auto srcExtent = contents.image->getCreationParameters().extent;
			const auto dstExtent = acquiredImage->getCreationParameters().extent;
			const IGPUCommandBuffer::SImageBlit regions[1] = {{
				.srcMinCoord = {0,0,0},
				.srcMaxCoord = {srcExtent.width,srcExtent.height,1},
				.dstMinCoord = {0,0,0},
				.dstMaxCoord = {dstExtent.width,dstExtent.height,1},
				.layerCount = acquiredImage->getCreationParameters().arrayLayers,
				.srcBaseLayer = 0, // TODO
				.dstBaseLayer = 0,
				.srcMipLevel = 0 // TODO
			}};
			retval &= cmdbuf->blitImage(contents.image,blitSrcLayout,acquiredImage,blitDstLayout,regions,IGPUSampler::ETF_LINEAR);
		}

		// barrier after
		const image_barrier_t postBarriers[2] = {
			{
				.barrier = {
					// When transitioning the image to VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR or VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, there is no need to delay subsequent processing,
					// or perform any visibility operations (as vkQueuePresentKHR performs automatic visibility operations).
					// To achieve this, the dstAccessMask member of the VkImageMemoryBarrier should be set to 0, and the dstStageMask parameter should be set to VK_PIPELINE_STAGE_2_NONE
					.dep = preBarriers[0].barrier.dep.nextBarrier(asset::PIPELINE_STAGE_FLAGS::NONE,asset::ACCESS_FLAGS::NONE)
				},
				.image = preBarriers[0].image,
				.subresourceRange = preBarriers[0].subresourceRange,
				.oldLayout = preBarriers[0].newLayout,
				.newLayout = IGPUImage::LAYOUT::PRESENT_SRC
			},
			{
				.barrier = {
					.dep = preBarriers[1].barrier.dep.nextBarrier(contents.barrier.dep.dstStageMask,contents.barrier.dep.dstAccessMask),
					.ownershipOp = IGPUCommandBuffer::SOwnershipTransferBarrier::OWNERSHIP_OP::RELEASE,
					.otherQueueFamilyIndex = contents.barrier.otherQueueFamilyIndex
				},
				.image = preBarriers[1].image,
				.subresourceRange = preBarriers[1].subresourceRange,
				.oldLayout = preBarriers[1].newLayout,
				.newLayout = contents.newLayout
			}
		};
		depInfo.imgBarriers = postBarriers;
		retval &= cmdbuf->pipelineBarrier(asset::EDF_NONE,depInfo);

		retval &= cmdbuf->end();
	}

	const IQueue::SSubmitInfo::SSemaphoreInfo blitted[1] = {
		{
			.semaphore = semaphore.get(),
			.value = 2,
			.stageMask = asset::PIPELINE_STAGE_FLAGS::BLIT_BIT
		}
	};
	// submit
	{
		const IQueue::SSubmitInfo::SSemaphoreInfo wait[2] = {acquired,waitBeforeBlit};
		const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[1] = {{.cmdbuf=cmdbuf.get()}};
		IQueue::SSubmitInfo infos[1] = {
			{
				.waitSemaphores = wait,
				.commandBuffers = cmdbufs,
				.signalSemaphores = blitted
			}
		};
		retval &= blitAndPresentQueue->submit(infos)==IQueue::RESULT::SUCCESS;
	}

	// present			
	switch (swapchainResources.swapchain->present({.queue=blitAndPresentQueue,.imgIndex=imageIndex,.waitSemaphores=blitted},std::move(cmdbuf)))
	{
		case ISwapchain::PRESENT_RESULT::SUBOPTIMAL: [[fallthrough]];
		case ISwapchain::PRESENT_RESULT::SUCCESS:
			// all resources can be dropped, the swapchain will hold onto them
			return retval;
		case ISwapchain::PRESENT_RESULT::OUT_OF_DATE:
			swapchainResources.invalidate();
			break;
		default:
			swapchainResources.becomeIrrecoverable();
			break;
	}
	// swapchain won't hold onto anything, so just block till resources not used anymore
	if (retval) // only if queue has submitted you have anything to wait on
	{
		ISemaphore::SWaitInfo infos[1] = {{
				.semaphore = blitted[0].semaphore,
				.value = blitted[0].value
		}};
		device->blockForSemaphores(infos);
	}
	return false;
}