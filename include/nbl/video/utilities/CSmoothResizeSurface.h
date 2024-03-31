#ifndef _NBL_VIDEO_C_SMOOTH_RESIZE_SURFACE_H_INCLUDED_
#define _NBL_VIDEO_C_SMOOTH_RESIZE_SURFACE_H_INCLUDED_


#include "nbl/video/utilities/ISimpleManagedSurface.h"


namespace nbl::video
{

// For this whole thing to work, you CAN ONLY ACQUIRE ONE IMAGE AT A TIME BEFORE CALLING PRESENT!
class NBL_API2 ISmoothResizeSurface : public ISimpleManagedSurface
{
	public:
		// Simple callback to facilitate detection of window being closed
		class ICallback : public ISimpleManagedSurface::ICallback
		{
			protected:
				// remember to call this when overriding it further down!
				inline virtual bool onWindowResized_impl(uint32_t w, uint32_t h) override
				{
					if (m_recreator)
						m_recreator->explicitRecreateSwapchain(w,h);
					return true;
				}

			private:
				friend class ISmoothResizeSurface;
				// `recreator` owns the `ISurface`, which refcounts the `IWindow` which refcounts the callback, so dumb pointer to avoid cycles 
				ISmoothResizeSurface* m_recreator = nullptr;
		};

		//
		struct SPresentSource
		{
			IGPUImage* image;
			VkRect2D rect;
		};

		//
		class NBL_API2 ISwapchainResources : public ISimpleManagedSurface::ISwapchainResources
		{
			protected:
				friend class ISmoothResizeSurface;

				// Returns what Pipeline Stages will be used for performing the `tripleBufferPresent`
				// We use this to set the waitStage for the Semaphore Wait, but only when ownership does not need to be acquired.
				virtual core::bitflag<asset::PIPELINE_STAGE_FLAGS> getTripleBufferPresentStages() const = 0;

				// Returns what if the commands to present `source` into `dstIx` Swapchain Image were successfully recorded to `cmdbuf` and can be submitted
				// The `cmdbuf` is already begun when given to the callback, and will be ended outside.
				// User is responsible for transitioning the image layouts (most notably the swapchain), acquiring ownership etc.
				// Performance Tip: DO NOT transition the layout of `source` inside this callback, have it already in the correct Layout you need!
				// However, if `qFamToAcquireSrcFrom!=IQueue::FamilyIgnored`, you need to acquire the ownership of the `source.image`
				virtual bool tripleBufferPresent(IGPUCommandBuffer* cmdbuf, const SPresentSource& source, const uint8_t dstIx, const uint32_t qFamToAcquireSrcFrom) = 0;
		};

		//
		inline CThreadSafeQueueAdapter* pickQueue(ILogicalDevice* device) const override final
		{
			const auto fam = pickQueueFamily(device);
			return device->getThreadSafeQueue(fam,device->getQueueCount(fam)-1);
		}

		// This is basically a poll, the extent CAN change between a call to this and `present`
		inline VkExtent2D getCurrentExtent()
		{
			std::unique_lock guard(m_swapchainResourcesMutex);
			// because someone might skip an acquire when area==0, handle window closing
			if (m_cb->isWindowOpen())
			while (true)
			{
				auto resources = getSwapchainResources();
				if (resources && resources->getStatus()==ISwapchainResources::STATUS::USABLE)
				{
					auto swapchain = resources->getSwapchain();
					if (swapchain)
					{
						const auto& params = swapchain->getCreationParameters().sharedParams;
						if (params.width>0 && params.height>0)
							return {params.width,params.height};
					}
				}
				// if got some weird invalid extent, try to recreate and retry once
				if (!recreateSwapchain())
					break;
			}
			else
				becomeIrrecoverable();
			return {0,0};
		}

		//
		inline ISemaphore* getPresentSemaphore() {return m_presentSemaphore.get();}

		// We need to prevent a spontaneous Triple Buffer Present during a resize between when:
		// - we check the semaphore value we need to wait on before rendering to the `SPresentSource::image`
		// - and the present from `SPresentSource::image
		// This is so that the value we would need to wait on, does not change.
		// Ofcourse if the target addresses of the atomic counters of wait values differ, no lock needed!
		inline std::unique_lock<std::mutex> pseudoAcquire(std::atomic_uint64_t* pPresentSemaphoreWaitValue)
		{
			if (pPresentSemaphoreWaitValue!=m_lastPresent.pPresentSemaphoreWaitValue)
				return {}; // no lock
			return std::unique_lock<std::mutex>(m_swapchainResourcesMutex);
		}

		struct SCachedPresentInfo
		{
			inline operator bool() const {return source.image && waitSemaphore && waitValue && pPresentSemaphoreWaitValue;}

			SPresentSource source = {};
			// only allow waiting for one semaphore, because there's only one source to present!
			ISemaphore* waitSemaphore = nullptr;
			uint64_t waitValue = 0;
			// what value will be signalled by the enqueued Triple Buffer Presents so far
			std::atomic_uint64_t* pPresentSemaphoreWaitValue = nullptr;
		};
		struct SPresentInfo : SCachedPresentInfo
		{
			uint32_t mostRecentFamilyOwningSource; // TODO: change to uint8_t
		};		
		// This is a present that you should regularly use from the main rendering thread or something.
		// Due to the constraints and mutexes on everything, its impossible to split this into a separate acquire and present call so this does both.
		// So CAN'T USE `acquireNextImage` for frame pacing, it was bad Vulkan practice anyway!
		inline bool present(std::unique_lock<std::mutex>&& acquireLock, const SPresentInfo& presentInfo)
		{
			if (!presentInfo)
				return false;

			std::unique_lock<std::mutex> guard;
			if (acquireLock)
				guard = std::move(acquireLock);
			else
			{
				guard = std::unique_lock<std::mutex>(m_swapchainResourcesMutex);
				assert(presentInfo.pPresentSemaphoreWaitValue!=m_lastPresent.pPresentSemaphoreWaitValue);
			}
			// The only thing we want to do under the mutex, is just enqueue a triple buffer present and a swapchain present, its not a lot.
			// Only acquire ownership if the Present queue is different to the current one and not concurrent sharing
			bool needFamilyOwnershipTransfer = getAssignedQueue()->getFamilyIndex()!=presentInfo.mostRecentFamilyOwningSource;
			if (presentInfo.source.image->getCachedCreationParams().isConcurrentSharing()) // in reality should also return false if the assigned queue is NOT in the Concurrent Sharing Set
				needFamilyOwnershipTransfer = false;
			return present_impl(presentInfo,needFamilyOwnershipTransfer);
		}

		// Call this when you want to recreate the swapchain with new extents
		inline bool explicitRecreateSwapchain(const uint32_t w, const uint32_t h)
		{
			// recreate the swapchain under a mutex
			std::unique_lock guard(m_swapchainResourcesMutex);

			// quick sanity check
			core::smart_refctd_ptr<ISwapchain> oldSwapchain(getSwapchainResources() ? getSwapchainResources()->getSwapchain():nullptr);
			if (oldSwapchain)
			{
				const auto& params = oldSwapchain->getCreationParameters().sharedParams;
				if (w==params.width && h==params.height)
					return true;
			}

			bool retval = recreateSwapchain(w,h);
			auto current = getSwapchainResources();
			// no point racing to present to old SC
			if (current->getSwapchain()==oldSwapchain.get())
				return true;

			// The triple present enqueue operations are fast enough to be done under a mutex, this is safer on some platforms. You need to "race to present" to avoid a flicker.
			// Queue family ownership acquire not needed, done by the the very first present when `m_lastPresentSource` wasset.
			return present_impl({m_lastPresent},false);
		}

	protected:
		inline ISmoothResizeSurface(core::smart_refctd_ptr<ISurface>&& _surface, ICallback* _cb) : ISimpleManagedSurface(std::move(_surface),_cb)
		{
			auto api = getSurface()->getAPIConnection();
			auto dcb = api->getDebugCallback();
			if (api->getEnabledFeatures().synchronizationValidation && dcb)
			{
				auto logger = dcb->getLogger();
				if (logger)
					logger->log("You're about to get a ton of False Positive Synchronization Errors from the Validation Layer due it Ignoring Queue Family Ownership Transfers (https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/7024)",system::ILogger::ELL_WARNING);
			}
		}
		virtual inline ~ISmoothResizeSurface()
		{
			// stop any calls into explicit resizes
			deinit_impl();
		}
		
		//
		inline void setSwapchainRecreator() {static_cast<ICallback*>(m_cb)->m_recreator=this;}

		//
		inline void deinit_impl() override final
		{
			// stop any calls into explicit resizes
			std::unique_lock guard(m_swapchainResourcesMutex);
			static_cast<ICallback*>(m_cb)->m_recreator = nullptr;

			m_lastPresent.source = {};
			m_lastPresentSourceImage = nullptr;
			m_lastPresent.waitSemaphore = nullptr;
			m_lastPresent.waitValue = 0;
			m_lastPresentSemaphore = nullptr;

			if (m_presentSemaphore)
			{
				auto device = const_cast<ILogicalDevice*>(m_presentSemaphore->getOriginDevice());
				const ISemaphore::SWaitInfo info[1] = {{
					.semaphore = m_presentSemaphore.get(),.value=getAcquireCount()
				}};
				device->blockForSemaphores(info);
			}

			std::fill(m_cmdbufs.begin(),m_cmdbufs.end(),nullptr);
			m_lastPresent.pPresentSemaphoreWaitValue = {};
			m_presentSemaphore = nullptr;
		}

		//
		inline bool recreateSwapchain(const uint32_t w=0, const uint32_t h=0)
		{
			auto* surface = getSurface();
			auto device = const_cast<ILogicalDevice*>(getAssignedQueue()->getOriginDevice());

			auto swapchainResources = getSwapchainResources();
			// dont assign straight to `m_swapchainResources` because of complex refcounting
			core::smart_refctd_ptr<ISwapchain> newSwapchain;
			{
				m_sharedParams.width = w;
				m_sharedParams.height = h;
				// Question: should we re-query the supported queues, formats, present modes, etc. just-in-time??
				auto* swapchain = swapchainResources->getSwapchain();
				if (swapchain ? swapchain->deduceRecreationParams(m_sharedParams):m_sharedParams.deduce(device->getPhysicalDevice(),surface))
				{
					// super special case, we can't re-create the swapchain but its possible to recover later on
					if (m_sharedParams.width==0 || m_sharedParams.height==0)
					{
						// we need to keep the old-swapchain around, but can drop the rest
						swapchainResources->invalidate();
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
							swapchainResources->getPreferredFormats(),
							swapchainResources->getPreferredEOTFs(),
							swapchainResources->getPreferredColorPrimaries()
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
				swapchainResources->invalidate();
				return swapchainResources->onCreateSwapchain(getAssignedQueue()->getFamilyIndex(),std::move(newSwapchain));
			}
			else
			{
				becomeIrrecoverable();
				return false;
			}
		}

		// handlers for acquisition exceptions (will get called under mutexes)
		inline SAcquireResult handleOutOfDate() override final
		{
			// try again, will re-create swapchain
			return ISimpleManagedSurface::acquireNextImage();
		}

		//
		inline bool present_impl(const SCachedPresentInfo& presentInfo, const bool acquireOwnership)
		{
			// irrecoverable or bad input
			if (!presentInfo || !getSwapchainResources())
				return false;

			// delayed init of our swapchain
			if (getSwapchainResources()->getStatus()!=ISwapchainResources::STATUS::USABLE && !recreateSwapchain())
				return false;

			// now pointer won't change until we get out from under the lock
			auto swapchainResources = static_cast<ISwapchainResources*>(getSwapchainResources());
			assert(swapchainResources);
			
			const auto acquire = acquireNextImage();
			if (!acquire)
				return false;

			// now that an image is acquired, we HAVE TO present
			m_lastPresentSourceImage = core::smart_refctd_ptr<IGPUImage>(presentInfo.source.image);
			m_lastPresentSemaphore = core::smart_refctd_ptr<ISemaphore>(presentInfo.waitSemaphore);
			m_lastPresent = presentInfo;
			// in case of failure, most we can do is just not submit an invalidated commandbuffer with the triple buffer present
			bool willBlit = true;

			const auto acquireCount = acquire.acquireCount;
			const IQueue::SSubmitInfo::SSemaphoreInfo waitSemaphores[2] = {
				{
					.semaphore = presentInfo.waitSemaphore,
					.value = presentInfo.waitValue,
					// If we need to Acquire Ownership of the Triple Buffer Source we need a Happens-Before between the Semaphore wait and the Acquire Ownership
					// https://github.com/KhronosGroup/Vulkan-Docs/issues/2319
					// else we need to know what stage starts reading from the Triple Buffer Source
					.stageMask = acquireOwnership ? asset::PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS:swapchainResources->getTripleBufferPresentStages()
				},
				{
					.semaphore = acquire.semaphore,
					.value = acquire.acquireCount,
					// There 99% surely probably be a Layout Transition of the acquired image away from PRESENT_SRC,
					// so need an ALL->NONE dependency between acquire and semaphore wait
					// https://github.com/KhronosGroup/Vulkan-Docs/issues/2319
					.stageMask = asset::PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS
				}
			};
			
			//
			auto queue = getAssignedQueue();
			auto device = const_cast<ILogicalDevice*>(queue->getOriginDevice());
			
			// need to wait before resetting a commandbuffer
			const auto maxFramesInFlight = getMaxFramesInFlight();
			if (acquireCount>maxFramesInFlight)
			{
				const ISemaphore::SWaitInfo cmdbufDonePending[1] = {
					{ 
						.semaphore = m_presentSemaphore.get(),
						.value = acquireCount-maxFramesInFlight
					}
				};
				device->blockForSemaphores(cmdbufDonePending);
			}


			// Maybe tie the cmbufs to the Managed Surface instead?
			const auto cmdBufIx = acquireCount%maxFramesInFlight;
			auto cmdbuf = m_cmdbufs[cmdBufIx].get();

			willBlit &= cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			willBlit &= swapchainResources->tripleBufferPresent(cmdbuf,presentInfo.source,acquire.imageIndex,acquireOwnership ? queue->getFamilyIndex():IQueue::FamilyIgnored);
			willBlit &= cmdbuf->end();
			
			const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[1] = {{.cmdbuf=cmdbuf}};
			// now enqueue the Triple Buffer Present
			const IQueue::SSubmitInfo::SSemaphoreInfo presented[1] = {{
				.semaphore = m_presentSemaphore.get(),
				.value = acquireCount,
				// There 99% surely probably be a Layout Transition of the acquired image to PRESENT_SRC,
				// so need a NONE->ALL dependency between that and semaphore signal
				// https://github.com/KhronosGroup/Vulkan-Docs/issues/2319
				.stageMask = asset::PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS
			}};
			const IQueue::SSubmitInfo submitInfos[1] = {{
				.waitSemaphores = waitSemaphores,
				.commandBuffers = commandBuffers,
				.signalSemaphores = presented
			}};
			willBlit &= queue->submit(submitInfos)==IQueue::RESULT::SUCCESS;

			// handle two cases of present
			if (willBlit)
			{
				// let the others know we've enqueued another TBP
				const auto oldVal = m_lastPresent.pPresentSemaphoreWaitValue->exchange(acquireCount);
				assert(oldVal<acquireCount);

				return ISimpleManagedSurface::present(acquire.imageIndex,presented);
			}
			else
				return ISimpleManagedSurface::present(acquire.imageIndex,{waitSemaphores+1,1});
		}

		// Because the surface can start minimized (extent={0,0}) we might not be able to create the swapchain right away, so store creation parameters until we can create it.
		ISwapchain::SSharedCreationParams m_sharedParams = {};
		// Have to use a second semaphore to make acquire-present pairs independent of each other, also because there can be no ordering ensured between present->acquire
		core::smart_refctd_ptr<ISemaphore> m_presentSemaphore;
		// Command Buffers for blitting/copying the Triple Buffers to Swapchain Images
		std::array<core::smart_refctd_ptr<IGPUCommandBuffer>,ISwapchain::MaxImages> m_cmdbufs = {};

	private:
		// used to protect access to swapchain resources during present and recreateExplicit
		std::mutex m_swapchainResourcesMutex;
		// Why do we delay the swapchain recreate present until the rendering of the most recent present source is done? Couldn't we present whatever latest Triple Buffer is done?
		// No because there can be presents enqueued whose wait semaphores have not signalled yet, meaning there could be images presented in the future.
		// Unless you like your frames to go backwards in time in a special "rewind glitch" you need to blit the frame that has not been presented yet or is the same as most recently enqueued.
		SCachedPresentInfo m_lastPresent = {};
		// extras for lifetime preservation till next immediate spontaneous triple buffer present
		core::smart_refctd_ptr<ISemaphore> m_lastPresentSemaphore = {};
		core::smart_refctd_ptr<IGPUImage> m_lastPresentSourceImage = {};
};

// The use of this class is supposed to be externally synchronized
template<typename SwapchainResources> requires std::is_base_of_v<ISmoothResizeSurface::ISwapchainResources,SwapchainResources>
class CSmoothResizeSurface final : public ISmoothResizeSurface
{
	public:
		using this_t = CSmoothResizeSurface<SwapchainResources>;

		// Factory method so we can fail, requires a `_surface` created from a window and with a callback that inherits from `ICallback` declared just above
		template<typename Surface> requires std::is_base_of_v<CSurface<typename Surface::window_t,typename Surface::immediate_base_t>,Surface>
		static inline core::smart_refctd_ptr<this_t> create(core::smart_refctd_ptr<Surface>&& _surface)
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

		//
		inline bool init(CThreadSafeQueueAdapter* queue, std::unique_ptr<SwapchainResources>&& scResources, const ISwapchain::SSharedCreationParams& sharedParams={})
		{
			// swapchain callback already deinitialized, so no mutex needed here
			if (!scResources || !base_init(queue))
				return init_fail();

			auto device = const_cast<ILogicalDevice*>(queue->getOriginDevice());

			m_sharedParams = sharedParams;
			if (!m_sharedParams.deduce(device->getPhysicalDevice(),getSurface()))
				return init_fail();

			// want to keep using the same semaphore throughout the lifetime to not run into sync issues
			if (!m_presentSemaphore)
			{
				m_presentSemaphore = device->createSemaphore(0u);
				if (!m_presentSemaphore)
					return init_fail();
			}

			// transient commandbuffer and pool to perform the blits or other copies to SC images
			auto pool = device->createCommandPool(queue->getFamilyIndex(),IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
			if (!pool || !pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,{m_cmdbufs.data(),getMaxFramesInFlight()}))
				return init_fail();

			m_swapchainResources = std::move(scResources);
			
			setSwapchainRecreator();
			return true;
		}

	protected:
		using ISmoothResizeSurface::ISmoothResizeSurface;

		inline bool checkQueueFamilyProps(const IPhysicalDevice::SQueueFamilyProperties& props) const override {return props.queueFlags.hasFlags(SwapchainResources::RequiredQueueFlags);}

		// All of the below are called from under a mutex
		inline ISimpleManagedSurface::ISwapchainResources* getSwapchainResources() override {return m_swapchainResources.get();}
		inline void becomeIrrecoverable() override {m_swapchainResources = nullptr;}

	private:
		// As per the above, the swapchain might not be possible to create or recreate right away, so this might be
		// either nullptr before the first successful acquire or the old to-be-retired swapchain.
		std::unique_ptr<SwapchainResources> m_swapchainResources = {};
};

}
#endif