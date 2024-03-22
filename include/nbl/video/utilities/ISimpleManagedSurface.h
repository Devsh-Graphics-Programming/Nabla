#ifndef _NBL_VIDEO_I_SIMPLE_MANAGED_SURFACE_H_INCLUDED_
#define _NBL_VIDEO_I_SIMPLE_MANAGED_SURFACE_H_INCLUDED_


#include "nbl/video/ISwapchain.h"
#include "nbl/video/ILogicalDevice.h"

#include "nbl/video/CVulkanSwapchain.h"


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
		inline uint8_t getMaxFramesInFlight() const {return m_maxFramesInFlight;}
		
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
				//
				virtual inline std::span<const asset::E_FORMAT> getPreferredFormats() const {return ISwapchain::SCreationParams::DefaultPreferredFormats;}
				virtual inline std::span<const asset::ELECTRO_OPTICAL_TRANSFER_FUNCTION> getPreferredEOTFs() const {return ISwapchain::SCreationParams::DefaultEOTFs;}
				virtual inline std::span<const asset::E_COLOR_PRIMARIES> getPreferredColorPrimaries() const {return ISwapchain::SCreationParams::DefaultColorPrimaries;}

				// If `init` not called yet this might return nullptr, or the old stale swapchain that should be retired
				inline ISwapchain* getSwapchain() const {return swapchain.get();}

				//
				inline IGPUImage* getImage(const uint8_t index) const
				{
					if (index<images.size())
						return images[index].get();
					return nullptr;
				}

				enum class STATUS : int8_t
				{
					IRRECOVERABLE = -1,
					USABLE = 0,
					NOT_READY = 1,
				};
				//
				inline STATUS getStatus() const
				{
					if (getSwapchain())
						return getImage(0) ? STATUS::USABLE:STATUS::NOT_READY;
					return STATUS::IRRECOVERABLE;
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
					auto device = const_cast<ILogicalDevice*>(_swapchain->getOriginDevice());
					// create images
					for (auto i=0u; i<_swapchain->getImageCount(); i++)
					{
						images[i] = _swapchain->createImage(i);
						if (!images[i])
						{
							std::fill_n(images.begin(),i,nullptr);
							return false;
						}
					}

					swapchain = std::move(_swapchain);
					if (!onCreateSwapchain_impl(qFam))
					{
						invalidate();
						return false;
					}
					return true;
				}

			protected:
				virtual ~ISwapchainResources() = default;

				//
				inline void becomeIrrecoverable()
				{
					// Want to nullify things in an order that leads to fastest drops (if possible) and shallowest callstacks when refcounting
					invalidate();

					swapchain = nullptr;
				}
				
				// Here you drop your own resources of the base class
				virtual void invalidate_impl() {}

				// For creating extra per-image or swapchain resources you might need
				virtual bool onCreateSwapchain_impl(const uint8_t qFam) {return true;}

				// We start with a `nullptr` swapchain because some implementations might defer its creation
				core::smart_refctd_ptr<ISwapchain> swapchain = {};
				// Useful for everyone
				std::array<core::smart_refctd_ptr<IGPUImage>,ISwapchain::MaxImages> images = {};
		};

		//
		inline void deinit()
		{
			deinit_impl();

			// I'm going to be lazy and do this instead of blocking on each  of `m_acquireSemaphores`
			if (m_queue)
				m_queue->waitIdle();

			m_queue = nullptr;
			m_maxFramesInFlight = 0;
			m_acquireCount = 0;
			std::fill(m_acquireSemaphores.begin(),m_acquireSemaphores.end(),nullptr);
		}

		//
		inline bool irrecoverable() const {return !const_cast<ISimpleManagedSurface*>(this)->getSwapchainResources();}

		//
		inline CThreadSafeQueueAdapter* getAssignedQueue() const {return m_queue;}

		// An interesting solution to the "Frames In Flight", our tiny wrapper class will have its own Timeline Semaphore incrementing with each acquire, and thats it.
		inline uint64_t getAcquireCount() {return m_acquireCount;}
		inline ISemaphore* getAcquireSemaphore(const uint8_t ix) {return ix<m_maxFramesInFlight ? m_acquireSemaphores[ix].get():nullptr;}

		// What `acquireNextImage` returns
		struct SAcquireResult
		{
			explicit inline operator bool() const {return semaphore && imageIndex!=ISwapchain::MaxImages;}

			ISemaphore* semaphore = nullptr;
			uint64_t acquireCount = 0;
			uint8_t imageIndex = ISwapchain::MaxImages;
		};

	protected: // some of the methods need to stay protected in this base class because they need to be performed under a Mutex for smooth resize variants
		inline ISimpleManagedSurface(core::smart_refctd_ptr<ISurface>&& _surface, ICallback* _cb) : m_surface(std::move(_surface)), m_cb(_cb) {}
		virtual inline ~ISimpleManagedSurface() = default;
		
		virtual inline bool checkQueueFamilyProps(const IPhysicalDevice::SQueueFamilyProperties& props) const {return true;}
		
		// We need to defer the swapchain creation till the Physical Device is chosen and Queues are created together with the Logical Device
		// Generally you should have a regular `init` in the final derived class to call this
		inline bool base_init(CThreadSafeQueueAdapter* queue)
		{
			deinit();
			if (queue)
			{
				m_queue = queue;

				auto device = const_cast<ILogicalDevice*>(m_queue->getOriginDevice());
				auto physDev = device->getPhysicalDevice();
				if (m_surface->isSupportedForPhysicalDevice(physDev,queue->getFamilyIndex()))
				{
					{
						ISurface::SCapabilities caps = {};
						m_surface->getSurfaceCapabilitiesForPhysicalDevice(physDev,caps);
						// vkAcquireNextImageKHR should not be called if the number of images that the application has currently acquired is greater than SwapchainImages-MinimumImages
						m_maxFramesInFlight = core::min(caps.maxImageCount+1-caps.minImageCount,ISwapchain::MaxImages);
					}

					for (uint8_t i=0u; i<m_maxFramesInFlight; i++)
					{
						m_acquireSemaphores[i] = device->createSemaphore(0u);
						if (!m_acquireSemaphores[i])
							return init_fail();
					}
				}
			}
			return true;
		}

		// just a simple convenience wrapper
		inline bool init_fail()
		{
			deinit();
			return false;
		}

		// RETURNS: `ISwapchain::MaxImages` on failure, otherwise its the acquired image's index.
		inline SAcquireResult acquireNextImage()
		{
			// Only check upon an acquire, previously acquired images MUST be presented
			// Window/Surface got closed, but won't actually disappear UNTIL the swapchain gets dropped,
			// which is outside of our control here as there is a nice chain of lifetimes of:
			// `ExternalCmdBuf -via usage of-> Swapchain Image -memory provider-> Swapchain -created from-> Window/Surface`
			// Only when the last user of the swapchain image drops it, will the window die.
			if (isWindowOpen())
			{
				auto swapchainResources = getSwapchainResources();
				if (!swapchainResources || swapchainResources->getStatus()!=ISwapchainResources::STATUS::USABLE)
					return {};
				
				const auto nextAcquireSignal = m_acquireCount+1;
				SAcquireResult retval = {
					.semaphore = m_acquireSemaphores[nextAcquireSignal%m_maxFramesInFlight].get(),
					.acquireCount = nextAcquireSignal
				};
				const IQueue::SSubmitInfo::SSemaphoreInfo signalInfos[1] = {
					{
						.semaphore = retval.semaphore,
						.value = nextAcquireSignal
					}
				};

				uint32_t imageIndex;
				// We don't support resizing (swapchain recreation) in this example, so a failure to acquire is a failure to keep running
				switch (swapchainResources->swapchain->acquireNextImage({.queue=m_queue,.signalSemaphores=signalInfos},&imageIndex))
				{
					case ISwapchain::ACQUIRE_IMAGE_RESULT::SUBOPTIMAL: [[fallthrough]];
					case ISwapchain::ACQUIRE_IMAGE_RESULT::SUCCESS:
						// the semaphore will only get signalled upon a successful acquire
						m_acquireCount = nextAcquireSignal;
						retval.imageIndex = imageIndex;
						return retval;
					case ISwapchain::ACQUIRE_IMAGE_RESULT::TIMEOUT: [[fallthrough]];
					case ISwapchain::ACQUIRE_IMAGE_RESULT::NOT_READY: // don't throw our swapchain away just because of a timeout XD
						assert(false); // shouldn't happen though cause we use uint64_t::max() as the timeout
						break;
					case ISwapchain::ACQUIRE_IMAGE_RESULT::OUT_OF_DATE:
						swapchainResources->invalidate();
						// try again, will re-create swapchain
						{
							retval = handleOutOfDate();
							if (bool(retval))
								return retval;
						}
					default:
						break;
				}
			}
			becomeIrrecoverable();
			return {};
		}

		// nice little callback
		virtual SAcquireResult handleOutOfDate() = 0;
		
		//
		inline bool present(const uint8_t imageIndex, const std::span<const IQueue::SSubmitInfo::SSemaphoreInfo> waitSemaphores)
		{
			auto swapchainResources = getSwapchainResources();
			if (!swapchainResources || swapchainResources->getStatus()!=ISwapchainResources::STATUS::USABLE)
				return false;

			const ISwapchain::SPresentInfo info = {
				.queue = m_queue,
				.imgIndex = imageIndex,
				.waitSemaphores = waitSemaphores
			};
			switch (swapchainResources->swapchain->present(info))
			{
				case ISwapchain::PRESENT_RESULT::SUBOPTIMAL: [[fallthrough]];
				case ISwapchain::PRESENT_RESULT::SUCCESS:
					return true;
				case ISwapchain::PRESENT_RESULT::OUT_OF_DATE:
					swapchainResources->invalidate();
					break;
				default:
					becomeIrrecoverable();
					break;
			}
			return false;
		}

		//
		virtual ISwapchainResources* getSwapchainResources() = 0;
		virtual void becomeIrrecoverable() = 0;

		//
		virtual void deinit_impl() = 0;

		// to trigger `becomeIrrecoverable` if window got closwd
		inline bool isWindowOpen()
		{
			if (!m_cb) return true; // native hwnd has no callbacks set -> user's responsibility to not acquire on window close corresponding to the Surface HWND
			return m_cb->isWindowOpen();
		}

		//
		ICallback* const m_cb = nullptr;

	private:
		// persistent and constant for whole lifetime of the object
		const core::smart_refctd_ptr<ISurface> m_surface;

		// Use a Threadsafe queue to make sure we can do smooth resize in derived class, might get re-assigned
		CThreadSafeQueueAdapter* m_queue = nullptr;
		//
		uint8_t m_maxFramesInFlight = 0;
		// Created and persistent after first initialization, Note that we need one swapchain per Frame In Fligth because Acquires can't wait or synchronize with anything
		// The only rule is that you can only have `m_maxFramesInFlight` pending acquires to wait with an infinte timeout, so thats about as far as they synchronize.
		std::array<core::smart_refctd_ptr<ISemaphore>,ISwapchain::MaxImages> m_acquireSemaphores;
		// You don't want to use `m_swapchainResources.swapchain->getAcquireCount()` because it resets when swapchain gets recreated
		uint64_t m_acquireCount = 0;
};

}
#endif