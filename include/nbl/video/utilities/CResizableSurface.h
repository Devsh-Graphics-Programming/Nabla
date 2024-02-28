#ifndef _NBL_VIDEO_C_RESIZABLE_SURFACE_H_INCLUDED_
#define _NBL_VIDEO_C_RESIZABLE_SURFACE_H_INCLUDED_


#include "nbl/video/utilities/ISimpleManagedSurface.h"


namespace nbl::video
{

// For this whole thing to work, you CAN ONLY ACQUIRE ONE IMAGE AT A TIME BEFORE CALLING PRESENT!
class NBL_API2 IResizableSurface : public ISimpleManagedSurface
{
	public:
		// Simple callback to facilitate detection of window being closed
		class ICallback : public ISimpleManagedSurface::ICallback
		{
			protected:
				friend class IResizableSurface;
				// `recreator` owns the `ISurface`, which refcounts the `IWindow` which refcounts the callback, so fumb pointer to avoid cycles 
				inline void setSwapchainRecreator(IResizableSurface* recreator) {m_recreator = recreator;}

				// remember to call this when overriding it further down!
				inline virtual bool onWindowResized_impl(uint32_t w, uint32_t h) override
				{
					m_recreator->explicitRecreateSwapchain(w,h);
					return true;
				}

				IResizableSurface* m_recreator;
		};

		//
		class NBL_API2 ISwapchainResources : public core::IReferenceCounted, public ISimpleManagedSurface::ISwapchainResources
		{
			protected:
				// A callback to do a regular copy to image
				struct SPresentInfo
				{
					inline operator bool() const {return sourceImage.get();}

					core::smart_refctd_ptr<IGPUImage> sourceImage = nullptr;
					// TODO: add sourceRegion
					IQueue::SSubmitInfo::SSemaphoreInfo signal;
				};
				virtual SPresentInfo tripleBufferPresent() = 0;
				// - wait for source image rendering to be finished & transition acquired image to correct layout
				// - enqueue a blit or something from the source image to acquired image
				// - transition acquired image to correct layout
				// - submit to a queue
				// - tell us what semaphore to signal and wait on

				// The `cmdbuf` is already begun when given to the callback, and will be ended inside
				// You can count on the `something.signal...` to be awaited before the `cmdbuf` is submitted, so image layout of `something.sourceImage` should match that.
				// Returns what stage the submit signal semaphore should signal from for the presentation to wait on 
				inline asset::PIPELINE_STAGE_FLAGS raceToPresent_impl(const IGPUImage* lastSource, IGPUCommandBuffer* cmdbuf);

				// if `tripleBufferToImageCopy` is expected to close up commandbuffers or something
				virtual void presentFailed() = 0;

			private:
				friend class IResizableSurface;
				inline void setStatus(const STATUS _status) {status=_status;}

				// Returns whether `presentSemaphore` will actually be signalled
				inline bool raceToPresent(CThreadSafeQueueAdapter* queue, const IQueue::SSubmitInfo::SSemaphoreInfo& wait, const IGPUImage* lastSource, IGPUCommandBuffer* cmdbuf, ISemaphore* presentSemaphore, const uint64_t presentValue)
				{
					const IQueue::SSubmitInfo::SSemaphoreInfo readyToPresent[1] = {{
						.semaphore = presentSemaphore,
						.value = presentValue,
						.stageMask = raceToPresent_impl(lastSource,cmdbuf)
					}};
					const std::span<const IQueue::SSubmitInfo::SSemaphoreInfo> waitSemaphores = {&wait,1};

					bool willBlit = false;
					// successful blit command enqueue if we have to wait on something
					if (readyToPresent->stageMask && cmdbuf->end())
					{
						const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[1] = {{
							.cmdbuf=cmdbuf
						}};
						const IQueue::SSubmitInfo submits[1] = {{
							.waitSemaphores = waitSemaphores,
							.commandBuffers = commandBuffers,
							.signalSemaphores = readyToPresent
						}};
						willBlit = queue->submit(submits)==IQueue::RESULT::SUCCESS;
					}
					
					// need to present either way
					if (willBlit)
					{
						switch (swapchain->present({.queue=queue,.imgIndex=acquiredIx,.waitSemaphores=readyToPresent},core::smart_refctd_ptr<core::IReferenceCounted>(cmdbuf)))
						{
							case ISwapchain::PRESENT_RESULT::SUBOPTIMAL: [[fallthrough]];
							case ISwapchain::PRESENT_RESULT::SUCCESS:
								// all resources can be dropped, the swapchain will hold onto them
								return true;
							case ISwapchain::PRESENT_RESULT::OUT_OF_DATE:
								invalidate();
								break;
							default:
								becomeIrrecoverable();
								break;
						}
						// now in a weird situation where we have pending blit work on the GPU but no present and refcounting to keep the submitted cmdbuf and semaphores alive
						const ISemaphore::SWaitInfo infos[1] = {{.semaphore=presentSemaphore,.value=presentValue}};
						const_cast<ILogicalDevice*>(queue->getOriginDevice())->blockForSemaphores(infos);
					}
					else
						swapchain->present({.queue=queue,.imgIndex=acquiredIx,.waitSemaphores=waitSemaphores});
					return false;
				}

				inline uint8_t immediateAcquire(CThreadSafeQueueAdapter* queue, const IQueue::SSubmitInfo::SSemaphoreInfo& acquireSignal)
				{
					assert(acquiredIx==ISwapchain::MaxImages);
					uint32_t imageIndex = ISwapchain::MaxImages;
					switch (swapchain->acquireNextImage({.queue=queue,.signalSemaphores={&acquireSignal,1}},&imageIndex))
					{
						case ISwapchain::ACQUIRE_IMAGE_RESULT::SUBOPTIMAL: [[fallthrough]];
						case ISwapchain::ACQUIRE_IMAGE_RESULT::SUCCESS:
							break;
						case ISwapchain::ACQUIRE_IMAGE_RESULT::TIMEOUT: [[fallthrough]];
						case ISwapchain::ACQUIRE_IMAGE_RESULT::NOT_READY: // don't throw our swapchain away just because of a timeout XD
							assert(false); // shouldn't happen though cause we use uint64_t::max() as the timeout
							return false;
						case ISwapchain::ACQUIRE_IMAGE_RESULT::OUT_OF_DATE:
							invalidate();
							return false;
						default:
							becomeIrrecoverable();
							return false;
					}
					acquiredIx = static_cast<uint8_t>(imageIndex);
					return true;
				}

				uint8_t acquiredIx = ISwapchain::MaxImages;
		};
		
		// TODO: explanation
		inline core::smart_refctd_ptr<ISwapchainResources> acquireNextImage()
		{
			std::unique_lock guard(m_swapchainResourcesMutex);
			uint8_t imageIx = ISimpleManagedSurface::acquireNextImage();
			if (imageIx==ISwapchain::MaxImages)
				return nullptr;
			// rememver to get this AFTER the `acquireNextImage` because it might re-create the swapchain
			auto retval = obtainSwapchainResources();
			retval->acquiredIx = imageIx;
			return retval;
		}
		
		// We expect that the user handles the blit/copy from the `ISwapchainResources::tripleBufferImages[imageIndex]` to `ISwapchainResources::images[imageIndex]` themselves before the present.
		// For the resize we NEED to know when the rendering to the TripleBuffer is complete so we can have the Swapchain Recreation Immediate Blit wait for it to complete!
		inline bool present(ISwapchainResources* acquired, core::IReferenceCounted* frameResources)
		{
			if (!acquired || !frameResources)
				return false;

			std::unique_lock guard(m_swapchainResourcesMutex);
			// `acquired` might have been retired in the meantime
			if (acquired->acquiredIx==ISwapchain::MaxImages)
				acquired = obtainSwapchainResources().get();

			const auto imageIx = acquired->acquiredIx;
			acquired->acquiredIx = ISwapchain::MaxImages;
			// now we have a problem and can't present (this really shouldn't happen)
			if (acquired->getStatus()!=ISwapchainResources::STATUS::USABLE || imageIx==ISwapchain::MaxImages)
			{
				acquired->presentFailed();
				return false;
			}

			m_lastPresent = acquired->tripleBufferPresent();
			return ISimpleManagedSurface::present(imageIx,{&m_lastPresent.signal,1},frameResources);
		}

		// Call this when you want to recreate the swapchain with new extents
		inline bool explicitRecreateSwapchain(const uint32_t w, const uint32_t h, CThreadSafeQueueAdapter* blitAndPresentQueue=nullptr)
		{
			// quick sanity check
			{
				std::unique_lock guard(m_swapchainResourcesMutex);
				if (getSwapchainResources().getStatus()==ISwapchainResources::STATUS::USABLE)
				{
					const auto& params = getSwapchainResources().getSwapchain()->getCreationParameters().sharedParams;
					if (w==params.width && h==params.height)
						return true;
				}
			}

			// Need a good queue to present all the acquired images of the old swapchain
			auto device = const_cast<ILogicalDevice*>(getAssignedQueue()->getOriginDevice());
			{
				const auto qFamProps = device->getPhysicalDevice()->getQueueFamilyProperties();
				auto compatibleQueue = [&](const uint8_t qFam)->bool
				{
					return qFamProps[qFam].queueFlags.hasFlags(IQueue::FAMILY_FLAGS::GRAPHICS_BIT) && getSurface()->isSupportedForPhysicalDevice(device->getPhysicalDevice(),qFam);
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
							if (blitAndPresentQueue==getAssignedQueue())
								blitAndPresentQueue = device->getThreadSafeQueue(qFam,qCount-1);
							break;
						}
					}
				}

				if (!blitAndPresentQueue || compatibleQueue(blitAndPresentQueue->getFamilyIndex()))
					return false;
			}

			// create a different semaphore so we don't increase the `m_acquireCount`
			auto semaphore = device->createSemaphore(0);
			if (!semaphore)
				return false;

			// transient commandbuffer and pool to perform the blit, one for old SC image and one for new
			core::smart_refctd_ptr<IGPUCommandBuffer> cmdbufs[2];
			{
				auto pool = device->createCommandPool(blitAndPresentQueue->getFamilyIndex(),IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);
				if (!pool || !pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,cmdbufs))
					return false;

				for (auto cmdbuf : cmdbufs)
				if (!cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT))
					return false;
			}

			// recreate the swapchain under a mutex
			std::unique_lock guard(m_swapchainResourcesMutex);
			auto old = obtainSwapchainResources();
			// If we had swapchain_maintenance1 we'd just release the images, instead we need to present them.
			// So we blit the last frame enqueued for a present to any already acquired image and present it.
			const bool oldNeedsPresent = old->acquiredIx!=ISwapchain::MaxImages;
			core::smart_refctd_ptr<ISwapchain> oldSwapchain = old->swapchain;

			bool retval = recreateSwapchain(w,h);
			auto current = obtainSwapchainResources();
			// no point racing to present to old SC
			if (current->getSwapchain()==old->getSwapchain())
				return false;
			// The blit enqueue operations are fast enough to be done under a mutex, this is safer on some platforms

			// First blit and present the old SC, whether we can present successfully doesn't matter
			if (oldNeedsPresent && old->raceToPresent(blitAndPresentQueue,m_lastPresent.signal,m_lastPresent.sourceImage.get(),cmdbufs[0].get(),semaphore.get(),1))
				m_lastPresent.signal = {.semaphore=semaphore.get(),.value=1};

			// You need to "race to present" to avoid a flicker
			uint8_t imageIx = current->immediateAcquire(blitAndPresentQueue,acquired);
			if (imageIx==ISwapchain::MaxImages && current->raceToPresent(blitAndPresentQueue,m_lastPresent.signal,m_lastPresent.sourceImage.get(),cmdbufs[1].get(),semaphore.get(),2))
				return current->immediateAcquire(blitAndPresentQueue)!=ISwapchain::MaxImages; // leave an image acquired so that we can redirect the next present

			return false;
		}

	protected:
		inline IResizableSurface(core::smart_refctd_ptr<ISurface>&& _surface, ICallback* _cb) : ISimpleManagedSurface(std::move(_surface),_cb)
		{
			_cb->setSwapchainRecreator(this);
		}
		virtual inline ~IResizableSurface()
		{
			static_cast<ICallback*>(m_cb)->setSwapchainRecreator(nullptr);
		}

		// This hands out a smartpointer for a good reason, we might end up recreating it on a thread at literally any point
		virtual core::smart_refctd_ptr<ISwapchainResources> obtainSwapchainResources() = 0;

		// basic parameter checks
		inline bool init_impl(CThreadSafeQueueAdapter* queue, const ISwapchain::SSharedCreationParams& sharedParams) override final
		{
			m_sharedParams = sharedParams;
			auto device = const_cast<ILogicalDevice*>(queue->getOriginDevice());
			if (!m_sharedParams.deduce(device->getPhysicalDevice(),getSurface()))
				return false;

			obtainSwapchainResources()->setStatus(ISwapchainResources::STATUS::NOT_READY);
			return true;
		}

		// handlers for acquisition exceptions (will get called under mutexes)
		inline bool handleNotReady() override final
		{
			return recreateSwapchain();
		}
		inline uint8_t handleOutOfDate() override final
		{
			// try again, will re-create swapchain
			return ISimpleManagedSurface::acquireNextImage();
		}

		// Assume it will execute under a mutex
		virtual bool recreateSwapchain(const uint32_t w=0, const uint32_t h=0) = 0;

		// Because the surface can start minimized (extent={0,0}) we might not be able to create the swapchain right away, so store creation parameters until we can create it.
		ISwapchain::SSharedCreationParams m_sharedParams = {};

	private:
		// used to protect access to swapchain resources during acquire, present and immediateBlit
		std::mutex m_swapchainResourcesMutex;
		// Why do we delay the swapchain recreate present until the rendering of the most recent present source is done? Couldn't we present whatever latest Triple Buffer is done?
		// No because there can be presents enqueued whose wait semaphores have not signalled yet, meaning there could be images presented in the future.
		// Unless you like your frames to go backwards in time in a special "rewind glitch" you need to blit the frame that has not been presented yet or is the same as most recently enqueued.
		ISwapchainResources::SPresentInfo m_lastPresent = {};
};

// The use of this class is supposed to be externally synchronized
template<typename SwapchainResources> requires std::is_base_of_v<IResizableSurface::ISwapchainResources,SwapchainResources>
class CResizableSurface final : public IResizableSurface
{
	public:
		using this_t = CResizableSurface<SwapchainResources>;		
		// Factory method so we can fail, requires a `_surface` created from a window and with a callback that inherits from `ICallback` declared just above
		static inline core::smart_refctd_ptr<this_t> create(core::smart_refctd_ptr<ISurface>&& _surface)
		{
			if (!_surface)
				return nullptr;

			auto _window = _surface->getWindow();
			if (!_window)
				return nullptr;

			auto cb = dynamic_cast<ICallback*>(_window->getEventCallback());
			if (!cb)
				return nullptr;

			return core::smart_refctd_ptr<this_t>(new this_t(std::move(_surface),cb),core::dont_grab);
		}

	protected:
		inline CResizableSurface(core::smart_refctd_ptr<ISurface>&& _surface, ICallback* _cb) : IResizableSurface(std::move(_surface),_cb)
		{
			// need to create an initial one thats NOT_READY for the initialization to complete
			m_swapchainResources = core::make_smart_refctd_ptr<SwapchainResources>();
		}

		// All of the below are called under a mutex
		inline ISimpleManagedSurface::ISwapchainResources& getSwapchainResources() override {return *m_swapchainResources.get();}
		inline core::smart_refctd_ptr<ISwapchainResources> obtainSwapchainResources() override {return core::smart_refctd_ptr(m_swapchainResources);}

		inline bool recreateSwapchain(const uint32_t w, const uint32_t h) override
		{
			auto* surface = getSurface();
			auto device = const_cast<ILogicalDevice*>(getAssignedQueue()->getOriginDevice());

			// dont assign straight to `m_swapchainResources` because of complex refcounting and cycles
			core::smart_refctd_ptr<ISwapchain> newSwapchain;
			{
				m_sharedParams.width = w;
				m_sharedParams.height = h;
				// Question: should we re-query the supported queues, formats, present modes, etc. just-in-time??
				auto& swapchain = m_swapchainResources->swapchain;
				if (swapchain ? swapchain->deduceRecreationParams(m_sharedParams):m_sharedParams.deduce(device->getPhysicalDevice(),surface))
				{
					// super special case, we can't re-create the swapchain but its possible to recover later on
					if (m_sharedParams.width==0 || m_sharedParams.height==0)
					{
						// we need to keep the old-swapchain around, but can drop the rest
						m_swapchainResources.invalidate();
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
						if (params.deduceFormat(device->getPhysicalDevice()))
							newSwapchain = CVulkanSwapchain::create(core::smart_refctd_ptr<const ILogicalDevice>(device),std::move(params));
					}
				}
				else // parameter deduction failed
					return false;
			}

			if (newSwapchain)
			{
				m_swapchainResources.invalidate();
				m_swapchainResources = core::make_smart_refctd_ptr<SwapchainResources>();
				m_swapchainResources->swapchain = std::move(newSwapchain);
			}
			else
			{
				m_swapchainResources->becomeIrrecoverable();
				return false;
			}
			
			if (!m_swapchainResources->onCreateSwapchain())
			{
				m_swapchainResources->invalidate();
				return false;
			}

			swapchainResources->setStatus(ISwapchainResources::STATUS::USABLE);
			return true;
		}

	private:
		// TODO: Explanation
		core::smart_refctd_ptr<SwapchainResources> m_swapchainResources = {};
};

}
#endif