#include "nbl/video/utilities/ISimpleManagedSurface.h"

using namespace nbl;
using namespace video;


bool ISimpleManagedSurface::immediateBlit(const image_barrier_t& contents, IQueue* blitQueue)
{
	if (!contents.image || !m_queue)
		return false;

	auto device = const_cast<ILogicalDevice*>(m_queue->getOriginDevice());
	const auto qFamProps = device->getPhysicalDevice()->getQueueFamilyProperties();
	if (!blitQueue)
	{
		// default to using the presentation queue if it can blit
		if (qFamProps[m_queue->getFamilyIndex()].queueFlags.hasFlags(IQueue::FAMILY_FLAGS::GRAPHICS_BIT))
			blitQueue = m_queue;
		else // just pick first compatible
		for (uint8_t qFam=0; qFam<ILogicalDevice::MaxQueueFamilies; qFam++)
		if (device->getQueueCount(qFam) && qFamProps[qFam].queueFlags.hasFlags(IQueue::FAMILY_FLAGS::GRAPHICS_BIT))
		{
			blitQueue = device->getThreadSafeQueue(qFam,0);
			break;
		}
	}

	if (!blitQueue || qFamProps[blitQueue->getFamilyIndex()].queueFlags.hasFlags(IQueue::FAMILY_FLAGS::GRAPHICS_BIT))
		return false;

	// create a different semaphore so we don't increase the acquire counter in `this`
	auto semaphore = device->createSemaphore(0);
	if (!semaphore)
		return false;

	// transient commandbuffer and pool to perform the blit
	core::smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
	{
		auto pool = device->createCommandPool(blitQueue->getFamilyIndex(),IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);
		if (!pool || !pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,{&cmdbuf,1}) || !cmdbuf)
			return false;
	}
	
	const IQueue::SSubmitInfo::SSemaphoreInfo acquired[1] = {
		{
			.semaphore=semaphore.get(),
			.value=1
		}
	};
	// acquire
	;

	// now record the blit commands
	auto acquiredImage = getSwapchainResources().getImage(0xffu);
	{
		if (!cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT))
			return false;

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
		if (!cmdbuf->pipelineBarrier(asset::EDF_NONE,depInfo))
			return false;
		
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
			if (!cmdbuf->blitImage(contents.image,blitSrcLayout,acquiredImage,blitDstLayout,regions,IGPUSampler::ETF_LINEAR))
				return false;
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
		if (!cmdbuf->pipelineBarrier(asset::EDF_NONE,depInfo))
			return false;
	}

	// submit

	// present
	;

	return true;
}