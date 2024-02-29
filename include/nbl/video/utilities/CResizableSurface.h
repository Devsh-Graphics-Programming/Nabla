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
				// remember to call in the derived class on all of these
				virtual inline void invalidate_impl() override
				{
					std::fill(cmdbufs.begin(),cmdbufs.end(),nullptr);
				}
				virtual inline bool onCreateSwapchain_impl(const uint8_t qFam) override
				{
					auto device = const_cast<ILogicalDevice*>(swapchain->getOriginDevice());

					// transient commandbuffer and pool to perform the blits or copies to SC images
					auto pool = device->createCommandPool(qFam,IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
					if (!pool || !pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,{cmdbufs.data(),swapchain->getImageCount()}))
						return false;
				
					return true;
				}

				// The `cmdbuf` is already begun when given to the callback, and will be ended inside
				// Returns what stage the submit signal semaphore should signal from for the presentation to wait on 
				virtual asset::PIPELINE_STAGE_FLAGS tripleBufferPresent(IGPUCommandBuffer* cmdbuf, IGPUImage* source) = 0;

			private:
				friend class IResizableSurface;

				std::array<core::smart_refctd_ptr<IGPUCommandBuffer>,ISwapchain::MaxImages> cmdbufs = {};
		};

		struct SPresentInfo
		{
			inline operator bool() const {return source;}

			IGPUImage* source = nullptr;
			// TODO: add sourceRegion
			IQueue::SSubmitInfo::SSemaphoreInfo wait;
			core::IReferenceCounted* frameResources;
		};		
		// TODO: explanations
		inline bool present(const SPresentInfo& presentInfo)
		{
			std::unique_lock guard(m_swapchainResourcesMutex);
			// The only thing we want to do under the mutex, is just enqueue a blit and a present, its not a lot
			return present_impl(presentInfo);
		}

		// Call this when you want to recreate the swapchain with new extents
		inline bool explicitRecreateSwapchain(const uint32_t w, const uint32_t h, CThreadSafeQueueAdapter* blitAndPresentQueue=nullptr)
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

			// The blit enqueue operations are fast enough to be done under a mutex, this is safer on some platforms
			// You need to "race to present" to avoid a flicker
			return present({.source=m_lastPresentSource,.wait=m_lastPresentWait,.frameResources=nullptr});
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

		//
		inline CThreadSafeQueueAdapter* pickQueue(ILogicalDevice* device) const override final
		{
			const auto fam = pickQueueFamily(device);
			return device->getThreadSafeQueue(fam,device->getQueueCount(fam)-1);
		}

		//
		inline bool init_impl(CThreadSafeQueueAdapter* queue, const ISwapchain::SSharedCreationParams& sharedParams) override final
		{
			auto device = const_cast<ILogicalDevice*>(queue->getOriginDevice());

			// want to keep using the same semaphore throughout the lifetime to not run into sync issues
			if (!m_blitSemaphore)
			{
				m_blitSemaphore = device->createSemaphore(0u);
				if (!m_blitSemaphore)
					return false;
			}

			m_sharedParams = sharedParams;
			if (!m_sharedParams.deduce(device->getPhysicalDevice(),getSurface()))
				return false;

			return createSwapchainResources();
		}

		//
		inline bool recreateSwapchain(const uint32_t w=0, const uint32_t h=0)
		{
			auto* surface = getSurface();
			auto device = const_cast<ILogicalDevice*>(getAssignedQueue()->getOriginDevice());

			auto swapchainResources = getSwapchainResources();
			// dont assign straight to `m_swapchainResources` because of complex refcounting and cycles
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
						if (params.deduceFormat(device->getPhysicalDevice()))
							newSwapchain = CVulkanSwapchain::create(core::smart_refctd_ptr<const ILogicalDevice>(device),std::move(params));
					}
				}
				else // parameter deduction failed
					return false;
			}

			if (newSwapchain)
			{
				swapchainResources->invalidate();
				return createSwapchainResources()->onCreateSwapchain(getAssignedQueue()->getFamilyIndex(),std::move(newSwapchain));
			}
			else
			{
				becomeIrrecoverable();
				return false;
			}
		}

		// handlers for acquisition exceptions (will get called under mutexes)
		inline uint8_t handleOutOfDate() override final
		{
			// try again, will re-create swapchain
			return ISimpleManagedSurface::acquireNextImage();
		}

		//
		inline bool present_impl(const SPresentInfo& presentInfo)
		{
			if (!presentInfo)
				return false;

			// delayed init of our swapchain
			if (!getSwapchainResources() && !recreateSwapchain())
				return false;
			
			const uint8_t imageIx = acquireNextImage();
			if (imageIx==ISwapchain::MaxImages)
				return false;

			// now that an image is acquired, we HAVE TO present
			bool willBlit = true;
			const auto acquireCount = getAcquireCount();
			const IQueue::SSubmitInfo::SSemaphoreInfo waitSemaphores[2] = {
				presentInfo.wait,
				{
					.semaphore = getAcquireSemaphore(),
					.value = acquireCount,
					.stageMask = asset::PIPELINE_STAGE_FLAGS::NONE // presentation engine usage isn't a stage
				}
			};
			m_lastPresentSourceImage = core::smart_refctd_ptr<IGPUImage>(presentInfo.source);
			m_lastPresentSource = presentInfo.source;
			m_lastPresentWait = presentInfo.wait;

			// now pointer won't change until we get out from under the lock
			auto swapchainResources = static_cast<ISwapchainResources*>(getSwapchainResources());
			assert(swapchainResources);
			
			//
			auto queue = getAssignedQueue();
			auto device = const_cast<ILogicalDevice*>(queue->getOriginDevice());
			
			// need to wait before resetting a commandbuffer
			const auto scImageCount = swapchainResources->getSwapchain()->getImageCount();
			if (acquireCount>scImageCount)
			{
				const ISemaphore::SWaitInfo cmdbufDonePending[1] = {
					{ 
						.semaphore = m_blitSemaphore.get(),
						.value = acquireCount-scImageCount
					}
				};
				device->blockForSemaphores(cmdbufDonePending);
			}

			const auto cmdBufIx = acquireCount%swapchainResources->getSwapchain()->getImageCount();
			auto cmdbuf = swapchainResources->cmdbufs[cmdBufIx].get();

			willBlit &= cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			// now enqueue the mini-blit
			const IQueue::SSubmitInfo::SSemaphoreInfo blitted[1] = {
				{
					.semaphore = m_blitSemaphore.get(),
					.value = acquireCount,
					.stageMask = swapchainResources->tripleBufferPresent(cmdbuf,presentInfo.source)
				}
			};
			willBlit &= bool(blitted[1].stageMask.value);
			willBlit &= cmdbuf->end();
			
			const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[1] = {{.cmdbuf=cmdbuf}};
			const IQueue::SSubmitInfo submitInfos[1] = {{
				.waitSemaphores = waitSemaphores,
				.commandBuffers = commandBuffers,
				.signalSemaphores = blitted
			}};
			willBlit &= queue->submit(submitInfos)==IQueue::RESULT::SUCCESS;
#if 0
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
#endif
			return ISimpleManagedSurface::present(imageIx,blitted,presentInfo.frameResources);
		}

		// Assume it will execute under a mutex
		virtual ISwapchainResources* createSwapchainResources() = 0;

		// Because the surface can start minimized (extent={0,0}) we might not be able to create the swapchain right away, so store creation parameters until we can create it.
		ISwapchain::SSharedCreationParams m_sharedParams = {};

	private:
		core::smart_refctd_ptr<ISemaphore> m_blitSemaphore;
		// used to protect access to swapchain resources during acquire, present and immediateBlit
		std::mutex m_swapchainResourcesMutex;
		// Why do we delay the swapchain recreate present until the rendering of the most recent present source is done? Couldn't we present whatever latest Triple Buffer is done?
		// No because there can be presents enqueued whose wait semaphores have not signalled yet, meaning there could be images presented in the future.
		// Unless you like your frames to go backwards in time in a special "rewind glitch" you need to blit the frame that has not been presented yet or is the same as most recently enqueued.
		decltype(SPresentInfo::source) m_lastPresentSource = {};
		core::smart_refctd_ptr<IGPUImage> m_lastPresentSourceImage = {};
		IQueue::SSubmitInfo::SSemaphoreInfo m_lastPresentWait = {};
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
		using IResizableSurface::IResizableSurface;

		inline bool checkQueueFamilyProps(const IPhysicalDevice::SQueueFamilyProperties& props) const override {return props.queueFlags.hasFlags(SwapchainResources::RequiredQueueFlags);}

		// All of the below are called under a mutex
		inline ISwapchainResources* createSwapchainResources() override
		{
			m_swapchainResources = core::make_smart_refctd_ptr<SwapchainResources>();
			return m_swapchainResources.get();
		}
		inline ISimpleManagedSurface::ISwapchainResources* getSwapchainResources() override {return m_swapchainResources.get();}
		inline void becomeIrrecoverable() override {m_swapchainResources = nullptr;}

	private:
		// As per the above, the swapchain might not be possible to create or recreate right away, so this might be
		// either nullptr before the first successful acquire or the old to-be-retired swapchain.
		core::smart_refctd_ptr<SwapchainResources> m_swapchainResources = {};
};

}
#endif