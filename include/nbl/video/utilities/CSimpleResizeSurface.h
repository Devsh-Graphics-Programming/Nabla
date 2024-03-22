#ifndef _NBL_VIDEO_C_SIMPLE_RESIZE_SURFACE_H_INCLUDED_
#define _NBL_VIDEO_C_SIMPLE_RESIZE_SURFACE_H_INCLUDED_


#include "nbl/video/utilities/ISimpleManagedSurface.h"


namespace nbl::video
{

// For this whole thing to work, you CAN ONLY ACQUIRE ONE IMAGE AT A TIME BEFORE CALLING PRESENT!
// The use of this class is supposed to be externally synchronized
template<typename SwapchainResources> requires std::is_base_of_v<ISimpleManagedSurface::ISwapchainResources,SwapchainResources>
class CSimpleResizeSurface final : public ISimpleManagedSurface
{
	public:
		using this_t = CSimpleResizeSurface<SwapchainResources>;

		// Factory method so we can fail, requires a `_surface` created from a window and with a callback that inherits from `ICallback` declared just above
		template<typename Surface> requires std::is_base_of_v<CSurface<typename Surface::window_t,typename Surface::immediate_base_t>,Surface>
		static inline core::smart_refctd_ptr<this_t> create(core::smart_refctd_ptr<Surface>&& _surface)
		{
			if (!_surface)
				return nullptr;

			auto _window = _surface->getWindow();
			ICallback* cb = nullptr;
			if (_window)
				cb = dynamic_cast<ICallback*>(_window->getEventCallback());

			return core::smart_refctd_ptr<this_t>(new this_t(std::move(_surface),cb),core::dont_grab);
		}

		//
		inline bool init(CThreadSafeQueueAdapter* queue, std::unique_ptr<SwapchainResources>&& scResources, const ISwapchain::SSharedCreationParams& sharedParams={})
		{
			if (!scResources || !base_init(queue))
				return init_fail();

			m_sharedParams = sharedParams;
			if (!m_sharedParams.deduce(queue->getOriginDevice()->getPhysicalDevice(),getSurface()))
				return init_fail();

			m_swapchainResources = std::move(scResources);
			return true;
		}
		
		// Can be public because we don't need to worry about mutexes unlike the Smooth Resize class
		inline ISwapchainResources* getSwapchainResources() override {return m_swapchainResources.get();}

		// need to see if the swapchain is invalidated (e.g. because we're starting from 0-area old Swapchain) and try to recreate the swapchain
		inline SAcquireResult acquireNextImage()
		{
			if (!isWindowOpen())
			{
				becomeIrrecoverable();
				return {};
			}
			
			if (!m_swapchainResources || (m_swapchainResources->getStatus()!=ISwapchainResources::STATUS::USABLE && !recreateSwapchain()))
				return {};

			return ISimpleManagedSurface::acquireNextImage();
		}

		// its enough to just foward though
		inline bool present(const uint8_t imageIndex, const std::span<const IQueue::SSubmitInfo::SSemaphoreInfo> waitSemaphores)
		{
			return ISimpleManagedSurface::present(imageIndex,waitSemaphores);
		}

		//
		inline bool recreateSwapchain()
		{
			assert(m_swapchainResources);
			// dont assign straight to `m_swapchainResources` because of complex refcounting and cycles
			core::smart_refctd_ptr<ISwapchain> newSwapchain;
			// TODO: This block of code could be rolled up into `ISimpleManagedSurface::ISwapchainResources` eventually
			{
				auto* surface = getSurface();
				auto device = const_cast<ILogicalDevice*>(getAssignedQueue()->getOriginDevice());
				// 0s are invalid values, so they indicate we want them deduced
				m_sharedParams.width = 0;
				m_sharedParams.height = 0;
				// Question: should we re-query the supported queues, formats, present modes, etc. just-in-time??
				auto* swapchain = m_swapchainResources->getSwapchain();
				if (swapchain ? swapchain->deduceRecreationParams(m_sharedParams):m_sharedParams.deduce(device->getPhysicalDevice(),surface))
				{
					// super special case, we can't re-create the swapchain but its possible to recover later on
					if (m_sharedParams.width==0 || m_sharedParams.height==0)
					{
						// we need to keep the old-swapchain around, but can drop the rest
						m_swapchainResources->invalidate();
						return false;
					}
					// now lets try to create a new swapchain
					if (swapchain)
						newSwapchain = swapchain->recreate(m_sharedParams);
					else
					{
						ISwapchain::SCreationParams params = {
							.surface = core::smart_refctd_ptr<ISurface>(surface),
							.surfaceFormat = {},
							.sharedParams = m_sharedParams
							// we're not going to support concurrent sharing in this simple class
						};
						const bool success = params.deduceFormat(
							device->getPhysicalDevice(),
							m_swapchainResources->getPreferredFormats(),
							m_swapchainResources->getPreferredEOTFs(),
							m_swapchainResources->getPreferredColorPrimaries()
						);
						if (success)
							newSwapchain = CVulkanSwapchain::create(core::smart_refctd_ptr<const ILogicalDevice>(device),std::move(params));
					}
				}
				else // parameter deduction failed
					return false;
			}

			if (newSwapchain)
			{
				m_swapchainResources->invalidate();
				return m_swapchainResources->onCreateSwapchain(getAssignedQueue()->getFamilyIndex(),std::move(newSwapchain));
			}
			else
				becomeIrrecoverable();

			return false;
		}

	protected:
		using ISimpleManagedSurface::ISimpleManagedSurface;

		//
		inline void deinit_impl() override final
		{
			becomeIrrecoverable();
		}

		//
		inline void becomeIrrecoverable() override { m_swapchainResources = nullptr; }

		// gets called when OUT_OF_DATE upon an acquire
		inline SAcquireResult handleOutOfDate() override final
		{
			// recreate swapchain and try to acquire again
			if (recreateSwapchain())
				return ISimpleManagedSurface::acquireNextImage();
			return {};
		}

	private:
		// Because the surface can start minimized (extent={0,0}) we might not be able to create the swapchain right away, so store creation parameters until we can create it.
		ISwapchain::SSharedCreationParams m_sharedParams = {};
		// The swapchain might not be possible to create or recreate right away, so this might be
		// either nullptr before the first successful acquire or the old to-be-retired swapchain.
		std::unique_ptr<SwapchainResources> m_swapchainResources = {};
};

}
#endif