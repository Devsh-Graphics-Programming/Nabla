#ifndef _NBL_VIDEO_I_SIMPLE_MANAGED_SURFACE_H_INCLUDED_
#define _NBL_VIDEO_I_SIMPLE_MANAGED_SURFACE_H_INCLUDED_


#include "nbl/video/ISwapchain.h"
#include "nbl/video/ILogicalDevice.h"


namespace nbl::video
{

// The use of this class is supposed to be externally synchronized
// For this whole thing to work, you CAN ONLY ACQUIRE ONE IMAGE AT A TIME BEFORE CALLING PRESENT!
// Why? look at the `becomeIrrecoverable` implementation
class NBL_API2 ISimpleManagedSurface : public core::IReferenceCounted
{
	public:
		// Simple callback to facilitate detection of window being closed
		class ICallback : public ui::IWindow::IEventCallback
		{
			public:
				inline ICallback() : m_windowNotClosed(true) {}

				// unless you create a separate callback per window, both will "trip" this condition
				inline bool isWindowOpen() const {return m_windowNotClosed;}

			private:
				inline virtual bool onWindowClosed_impl() override
				{
					m_windowNotClosed = false;
					return true;
				}

				bool m_windowNotClosed;
		};
		
		//
		inline ISurface* getSurface() {return m_surface.get();}
		inline const ISurface* getSurface() const {return m_surface.get();}

		//
		inline uint8_t getMaxFramesInFlight(IPhysicalDevice* physDev) const
		{
			ISurface::SCapabilities caps = {};
			m_surface->getSurfaceCapabilitiesForPhysicalDevice(physDev,caps);
			// vkAcquireNextImageKHR should not be called if the number of images that the application has currently acquired is greater than SwapchainImages-MinimumImages
			return core::min(caps.maxImageCount+1-caps.minImageCount,ISwapchain::MaxImages);
		}
		
		// A small utility for the boilerplate
		inline uint8_t pickQueueFamily(ILogicalDevice* device) const
		{
			auto physDev = device->getPhysicalDevice();
			uint8_t qFam = 0u;
			for (; qFam<ILogicalDevice::MaxQueueFamilies; qFam++)
			if (device->getQueueCount(qFam) && m_surface->isSupportedForPhysicalDevice(physDev,qFam) && checkQueueFamilyProps(physDev->getQueueFamilyProperties()[qFam]))
				break;
			return qFam;
		}

		// Just pick the first queue within the first compatible family
		virtual inline CThreadSafeQueueAdapter* pickQueue(ILogicalDevice* device) const
		{
			return device->getThreadSafeQueue(pickQueueFamily(device),0);
		}

		// A class to hold resources that can die/invalidate spontaneously when a swapchain gets re-created.
		// Due to the inheritance and down-casting, you should make your own const `getSwapchainResources()` in the derived class thats not an override
		class NBL_API2 ISwapchainResources : core::InterfaceUnmovable
		{
				friend class ISimpleManagedSurface;

			public:
				// If `init` not called yet this might return nullptr, or the old stale swapchain that should be retired
				inline ISwapchain* getSwapchain() const {return swapchain.get();}

				//
				inline IGPUImage* getImage(const uint8_t index) const
				{
					if (index<images.size())
						return images[index].get();
					return nullptr;
				}

				// just drop the per-swapchain resources, e.g. Framebuffers with each of the swapchain images, or even pre-recorded commandbuffers
				inline void invalidate()
				{
					if (!images.front())
						return;
					invalidate_impl();
					std::fill(images.begin(),images.end(),nullptr);
				}
				
				// This will notify you of the swapchain being created and when you can start creating per-swapchain and per-image resources
				// NOTE: Current class doesn't trigger it because it knows nothing about how and when to create or recreate a swapchain.
				inline bool onCreateSwapchain(const uint8_t qFam, core::smart_refctd_ptr<ISwapchain>&& _swapchain)
				{
					swapchain = std::move(_swapchain);
					auto device = const_cast<ILogicalDevice*>(swapchain->getOriginDevice());
					// create images
					for (auto i=0u; i<swapchain->getImageCount(); i++)
					{
						images[i] = swapchain->createImage(i);
						if (!images[i])
						{
							std::fill_n(images.begin(),i,nullptr);
							return false;
						}
					}

					if (!onCreateSwapchain_impl(qFam))
					{
						invalidate();
						return false;
					}
					return true;
				}

			protected:
				virtual ~ISwapchainResources()
				{
					// just to get rid of circular refs
					becomeIrrecoverable();
				}

				//
				inline void becomeIrrecoverable()
				{
					// Want to nullify things in an order that leads to fastest drops (if possible) and shallowest callstacks when refcounting
					invalidate();

					// We need to call this method manually to make sure resources latched on swapchain images are dropped and cycles broken, otherwise its
					// EXTERMELY LIKELY (if you don't reset CommandBuffers) that you'll end up with a circular reference, for example:
					// `CommandBuffer -> SC Image[i] -> Swapchain -> FrameResource[i] -> CommandBuffer`
					// and a memory leak of: Swapchain and its Images, CommandBuffer and its pool CommandPool, and any resource used by the CommandBuffer.
					if (swapchain)
					while (swapchain->acquiredImagesAwaitingPresent()) {}
					swapchain = nullptr;
				}
				
				// here you drop your own resources of the base class
				virtual void invalidate_impl() = 0;

				// extra things you might need
				virtual bool onCreateSwapchain_impl(const uint8_t qFam) = 0;

				// We start with a `nullptr` swapchain because some implementations might defer its creation
				core::smart_refctd_ptr<ISwapchain> swapchain = {};
				// Useful for everyone
				std::array<core::smart_refctd_ptr<IGPUImage>,ISwapchain::MaxImages> images = {};
		};
		
		// We need to defer the swapchain creation till the Physical Device is chosen and Queues are created together with the Logical Device
		inline bool init(CThreadSafeQueueAdapter* queue, const ISwapchain::SSharedCreationParams& sharedParams={})
		{
			if (!queue)
				return false;

			auto device = const_cast<ILogicalDevice*>(queue->getOriginDevice());
			// want to keep using the same semaphore throughout the lifetime to not run into sync issues
			if (!m_acquireSemaphore)
			{
				m_acquireSemaphore = device->createSemaphore(0u);
				if (!m_acquireSemaphore)
					return false;
			}

			if (!init_impl(queue,sharedParams))
				return false;

			m_queue = queue;
			return true;
		}

		//
		inline bool irrecoverable() const {return !const_cast<ISimpleManagedSurface*>(this)->getSwapchainResources();}

		//
		inline CThreadSafeQueueAdapter* getAssignedQueue() const {return m_queue;}

		// An interesting solution to the "Frames In Flight", our tiny wrapper class will have its own Timeline Semaphore incrementing with each acquire, and thats it.
		inline uint64_t getAcquireCount() {return m_acquireCount;}
		inline ISemaphore* getAcquireSemaphore() {return m_acquireSemaphore.get();}

	protected: // some of the methods need to stay protected in this base class because they need to be performed under a Mutex for smooth resize variants
		inline ISimpleManagedSurface(core::smart_refctd_ptr<ISurface>&& _surface, ICallback* _cb) : m_surface(std::move(_surface)), m_cb(_cb) {}
		virtual inline ~ISimpleManagedSurface() = default;
		
		virtual inline bool checkQueueFamilyProps(const IPhysicalDevice::SQueueFamilyProperties& props) const {return true;}

		// RETURNS: `ISwapchain::MaxImages` on failure, otherwise its the acquired image's index.
		inline uint8_t acquireNextImage()
		{
			auto swapchainResources = getSwapchainResources();
			if (!swapchainResources)
				return ISwapchain::MaxImages;

			// Only check upon an acquire, previously acquired images MUST be presented
			// Window/Surface got closed, but won't actually disappear UNTIL the swapchain gets dropped,
			// which is outside of our control here as there is a nice chain of lifetimes of:
			// `ExternalCmdBuf -via usage of-> Swapchain Image -memory provider-> Swapchain -created from-> Window/Surface`
			// Only when the last user of the swapchain image drops it, will the window die.
			if (m_cb->isWindowOpen() && swapchainResources->swapchain)
			{
				const IQueue::SSubmitInfo::SSemaphoreInfo signalInfos[1] = {
					{
						.semaphore=m_acquireSemaphore.get(),
						.value=m_acquireCount+1
					}
				};

				uint32_t imageIndex;
				// We don't support resizing (swapchain recreation) in this example, so a failure to acquire is a failure to keep running
				switch (swapchainResources->swapchain->acquireNextImage({.queue=m_queue,.signalSemaphores=signalInfos},&imageIndex))
				{
					case ISwapchain::ACQUIRE_IMAGE_RESULT::SUBOPTIMAL: [[fallthrough]];
					case ISwapchain::ACQUIRE_IMAGE_RESULT::SUCCESS:
						// the semaphore will only get signalled upon a successful acquire
						m_acquireCount++;
						return static_cast<uint8_t>(imageIndex);
					case ISwapchain::ACQUIRE_IMAGE_RESULT::TIMEOUT: [[fallthrough]];
					case ISwapchain::ACQUIRE_IMAGE_RESULT::NOT_READY: // don't throw our swapchain away just because of a timeout XD
						assert(false); // shouldn't happen though cause we use uint64_t::max() as the timeout
						break;
					case ISwapchain::ACQUIRE_IMAGE_RESULT::OUT_OF_DATE:
						swapchainResources->invalidate();
						// try again, will re-create swapchain
						{
							const auto retval = handleOutOfDate();
							if (retval!=ISwapchain::MaxImages)
								return retval;
						}
					default:
						break;
				}
			}
			becomeIrrecoverable();
			return ISwapchain::MaxImages;
		}

		// nice little callback
		virtual uint8_t handleOutOfDate() = 0;
		
		// Frame Resources are not optional, shouldn't be null!
		inline bool present(const uint8_t imageIndex, const std::span<const IQueue::SSubmitInfo::SSemaphoreInfo> waitSemaphores, core::IReferenceCounted* frameResources)
		{
			auto swapchainResources = getSwapchainResources();
			if (!swapchainResources || !swapchainResources->swapchain || !frameResources)
				return false;

			const ISwapchain::SPresentInfo info = {
				.queue = m_queue,
				.imgIndex = imageIndex,
				.waitSemaphores = waitSemaphores
			};
			switch (swapchainResources->swapchain->present(info,core::smart_refctd_ptr<core::IReferenceCounted>(frameResources)))
			{
				case ISwapchain::PRESENT_RESULT::SUBOPTIMAL: [[fallthrough]];
				case ISwapchain::PRESENT_RESULT::SUCCESS:
					return true;
				case ISwapchain::PRESENT_RESULT::OUT_OF_DATE:
					swapchainResources->invalidate();
					break;
				default:
					// because we won't hold onto the `frameResources` we need to block for `waitSemaphores`
					{
						core::vector<ISemaphore::SWaitInfo> waitInfos(waitSemaphores.size());
						auto outWait = waitInfos.data();
						for (auto wait : waitSemaphores)
							*(outWait++) = {.semaphore=wait.semaphore,.value=wait.value};
						const_cast<ILogicalDevice*>(m_queue->getOriginDevice())->blockForSemaphores(waitInfos);
					}
					becomeIrrecoverable();
					break;
			}
			return false;
		}

		//
		virtual ISwapchainResources* getSwapchainResources() = 0;
		virtual void becomeIrrecoverable() = 0;

		// Generally used to check that per-swapchain resources can be created (including the swapchain itself)
		virtual bool init_impl(CThreadSafeQueueAdapter* queue, const ISwapchain::SSharedCreationParams& sharedParams) = 0;

		//
		ICallback* const m_cb = nullptr;

	private:
		// persistent and constant for whole lifetime of the object
		const core::smart_refctd_ptr<ISurface> m_surface;
		// Use a Threadsafe queue to make sure we can do smooth resize in derived class, might get re-assigned
		CThreadSafeQueueAdapter* m_queue = nullptr;
		// created and persistent after first initialization
		core::smart_refctd_ptr<ISemaphore> m_acquireSemaphore;
		// You don't want to use `m_swapchainResources.swapchain->getAcquireCount()` because it resets when swapchain gets recreated
		uint64_t m_acquireCount = 0;
};

}
#endif