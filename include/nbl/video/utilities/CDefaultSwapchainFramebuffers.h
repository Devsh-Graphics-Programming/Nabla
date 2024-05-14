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
		inline CDefaultSwapchainFramebuffers(ILogicalDevice* device, const asset::E_FORMAT format, const IGPURenderpass::SCreationParams::SSubpassDependency* dependencies) : m_device(device)
		{
			// If we create the framebuffers by default, we also need to default the renderpass (except dependencies)
			static const IGPURenderpass::SCreationParams::SColorAttachmentDescription colorAttachments[] = {
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
			static IGPURenderpass::SCreationParams::SSubpassDescription subpasses[] = {
				{},
				IGPURenderpass::SCreationParams::SubpassesEnd
			};
			subpasses[0].colorAttachments[0] = {.render={.attachmentIndex=0,.layout=IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL}};
				
			m_params.colorAttachments = colorAttachments;
			m_params.subpasses = subpasses;
			m_params.dependencies = dependencies;
		}

		inline IGPURenderpass* getRenderpass()
		{
			if (!m_renderpass)
				m_renderpass = m_device->createRenderpass(m_params);
			return m_renderpass.get();
		}

		inline IGPUFramebuffer* getFramebuffer(const uint8_t imageIx)
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
			const auto swapchain = getSwapchain();
			const auto count = swapchain->getImageCount();
			const auto& sharedParams = swapchain->getCreationParameters().sharedParams;
			for (uint32_t i=0u; i<count; i++)
			{
				auto imageView = m_device->createImageView({
					.flags = IGPUImageView::ECF_NONE,
					.subUsages = IGPUImage::EUF_RENDER_ATTACHMENT_BIT,
					.image = core::smart_refctd_ptr<IGPUImage>(getImage(i)),
					.viewType = IGPUImageView::ET_2D,
					.format = getImage(i)->getCreationParameters().format
				});
				std::string debugName = "Swapchain "+std::to_string(ptrdiff_t(swapchain));
				debugName += " Image View #"+std::to_string(i);
				imageView->setObjectDebugName(debugName.c_str());
				m_framebuffers[i] = createFramebuffer({{
					.renderpass = core::smart_refctd_ptr<IGPURenderpass>(getRenderpass()),
					.colorAttachments = &imageView.get(),
					.width = sharedParams.width,
					.height = sharedParams.height
				}});
				if (!m_framebuffers[i])
					return false;
			}
			return true;
		}

		virtual inline core::smart_refctd_ptr<IGPUFramebuffer> createFramebuffer(IGPUFramebuffer::SCreationParams&& params)
		{
			return m_device->createFramebuffer(std::move(params));
		}


		ILogicalDevice* const m_device;
		IGPURenderpass::SCreationParams m_params = {};
		core::smart_refctd_ptr<IGPURenderpass> m_renderpass;
		// Per-swapchain
		std::array<core::smart_refctd_ptr<IGPUFramebuffer>,ISwapchain::MaxImages> m_framebuffers;
};

}
#endif