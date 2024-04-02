// Copyright (C) 2023-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_C_DEFAULT_SWAPCHAIN_FRAMEBUFFERS_HPP_INCLUDED_
#define _NBL_VIDEO_C_DEFAULT_SWAPCHAIN_FRAMEBUFFERS_HPP_INCLUDED_


// Build on top of the previous one
#include "nabla.h"


namespace nbl::video
{
	
// Just a class to create a Default single Subpass Renderpass and hold framebuffers derived from swapchain images.
// WARNING: It assumes the format won't change between swapchain recreates!
class CDefaultSwapchainFramebuffers : public ISimpleManagedSurface::ISwapchainResources
{
	public:
		inline CDefaultSwapchainFramebuffers(ILogicalDevice* device, const asset::E_FORMAT format, const IGPURenderpass::SCreationParams::SSubpassDependency* dependencies)
		{
			// If we create the framebuffers by default, we also need to default the renderpass (except dependencies)
			const IGPURenderpass::SCreationParams::SColorAttachmentDescription colorAttachments[] = {
				{{
					{
						.format = format,
						.samples = IGPUImage::ESCF_1_BIT,
						.mayAlias = false
					},
					/*.loadOp = */IGPURenderpass::LOAD_OP::CLEAR,
					/*.storeOp = */IGPURenderpass::STORE_OP::STORE,
					/*.initialLayout = */IGPUImage::LAYOUT::UNDEFINED, // because we clear we don't care about contents
					/*.finalLayout = */ IGPUImage::LAYOUT::PRESENT_SRC // transition to presentation right away so we can skip a barrier
				}},
				IGPURenderpass::SCreationParams::ColorAttachmentsEnd
			};
			IGPURenderpass::SCreationParams::SSubpassDescription subpasses[] = {
				{},
				IGPURenderpass::SCreationParams::SubpassesEnd
			};
			subpasses[0].colorAttachments[0] = {.render={.attachmentIndex=0,.layout=IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL}};
				
			IGPURenderpass::SCreationParams params = {};
			params.colorAttachments = colorAttachments;
			params.subpasses = subpasses;
			params.dependencies = dependencies;
			m_renderpass = device->createRenderpass(params);
		}

		inline IGPURenderpass* getRenderpass() {return m_renderpass.get();}

		inline IGPUFramebuffer* getFrambuffer(const uint8_t imageIx)
		{
			if (imageIx<m_framebuffers.size())
				return m_framebuffers[imageIx].get();
			return nullptr;
		}

	protected:
		virtual inline void invalidate_impl()
		{
			std::fill(m_framebuffers.begin(),m_framebuffers.end(),nullptr);
		}

		// For creating extra per-image or swapchain resources you might need
		virtual inline bool onCreateSwapchain_impl(const uint8_t qFam)
		{
			auto device = const_cast<ILogicalDevice*>(m_renderpass->getOriginDevice());

			const auto swapchain = getSwapchain();
			const auto count = swapchain->getImageCount();
			const auto& sharedParams = swapchain->getCreationParameters().sharedParams;
			for (uint8_t i=0u; i<count; i++)
			{
				auto imageView = device->createImageView({
					.flags = IGPUImageView::ECF_NONE,
					.subUsages = IGPUImage::EUF_RENDER_ATTACHMENT_BIT,
					.image = core::smart_refctd_ptr<IGPUImage>(getImage(i)),
					.viewType = IGPUImageView::ET_2D,
					.format = getImage(i)->getCreationParameters().format
				});
				m_framebuffers[i] = device->createFramebuffer({{
					.renderpass = core::smart_refctd_ptr(m_renderpass),
					.colorAttachments = &imageView.get(),
					.width = sharedParams.width,
					.height = sharedParams.height
				}});
				if (!m_framebuffers[i])
					return false;
			}
			return true;
		}

		core::smart_refctd_ptr<IGPURenderpass> m_renderpass;
		// Per-swapchain
		std::array<core::smart_refctd_ptr<IGPUFramebuffer>,ISwapchain::MaxImages> m_framebuffers;
};

}
#endif