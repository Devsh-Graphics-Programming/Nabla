#define _NBL_STATIC_LIB_
#include <nabla.h>
#include "nbl/system/IApplicationFramework.h"
#include "nbl/ui/CGraphicalApplicationAndroid.h"
#include "nbl/ui/CWindowManagerAndroid.h"
#include "nbl/ui/IGraphicalApplicationFramework.h"
#if defined(_NBL_PLATFORM_WINDOWS_)
#include <nbl/system/CColoredStdoutLoggerWin32.h>
#elif defined(_NBL_PLATFORM_ANDROID_)
#include <nbl/system/CStdoutLoggerAndroid.h>
#endif
#include "nbl/system/CSystemAndroid.h"
#include "nbl/system/CSystemLinux.h"
#include "nbl/system/CSystemWin32.h"
// TODO: make these include themselves via `nabla.h`

class GraphicalApplication : public nbl::system::IApplicationFramework, public nbl::ui::IGraphicalApplicationFramework
{
	protected:
		~GraphicalApplication() {}
	public:
		GraphicalApplication(
			const std::filesystem::path& _localInputCWD,
			const std::filesystem::path& _localOutputCWD,
			const std::filesystem::path& _sharedInputCWD,
			const std::filesystem::path& _sharedOutputCWD
		) : nbl::system::IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}
};
//***** Application framework macros ******
#ifdef _NBL_PLATFORM_ANDROID_
using ApplicationBase = nbl::ui::CGraphicalApplicationAndroid;
#define APP_CONSTRUCTOR(type) type(android_app* app, const nbl::system::path& _localInputCWD,\
const nbl::system::path& _localOutputCWD,\
const nbl::system::path& _sharedInputCWD,\
const nbl::system::path& _sharedOutputCWD) : nbl::ui::CGraphicalApplicationAndroid(app, _localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}
#define NBL_COMMON_API_MAIN(android_app_class) NBL_ANDROID_MAIN_FUNC(android_app_class, CommonAPI::CommonAPIEventCallback)
#else
using ApplicationBase = GraphicalApplication;
#define APP_CONSTRUCTOR(type) type(const nbl::system::path& _localInputCWD,\
const nbl::system::path& _localOutputCWD,\
const nbl::system::path& _sharedInputCWD,\
const nbl::system::path& _sharedOutputCWD) : GraphicalApplication(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}
#define NBL_COMMON_API_MAIN(android_app_class) int main(int argc, char** argv){\
CommonAPI::main<android_app_class>(argc, argv);\
}
#endif




//***** Application framework macros ******


class CommonAPI
{
	CommonAPI() = delete;
public:

	class CommonAPIEventCallback;

	class InputSystem : public nbl::core::IReferenceCounted
	{
		public:
			template <class ChannelType>
			struct Channels
			{
				nbl::core::mutex lock;
				std::condition_variable added;
				nbl::core::vector<nbl::core::smart_refctd_ptr<ChannelType>> channels;
				nbl::core::vector<std::chrono::microseconds> timeStamps;
				uint32_t defaultChannelIndex = 0;
			};
			// TODO: move to "nbl/ui/InputEventChannel.h" once the interface of this utility struct matures, also maybe rename to `Consumer` ?
			template <class ChannelType>
			struct ChannelReader
			{
				template<typename F>
				inline void consumeEvents(F&& processFunc, nbl::system::logger_opt_ptr logger=nullptr)
				{
					auto events = channel->getEvents();
					const auto frontBufferCapacity = channel->getFrontBufferCapacity();
					if (events.size()>consumedCounter+frontBufferCapacity)
					{
						logger.log(
							"Detected overflow, %d unconsumed events in channel of size %d!",
							nbl::system::ILogger::ELL_ERROR,events.size()-consumedCounter,frontBufferCapacity
						);
						consumedCounter = events.size()-frontBufferCapacity;
					}
					typename ChannelType::range_t rng(events.begin() + consumedCounter, events.end());
					processFunc(rng);
					consumedCounter = events.size();
				}

				nbl::core::smart_refctd_ptr<ChannelType> channel = nullptr;
				uint64_t consumedCounter = 0ull;
			};
		
			InputSystem(nbl::system::logger_opt_smart_ptr&& logger) : m_logger(std::move(logger)) {}

			void getDefaultMouse(ChannelReader<nbl::ui::IMouseEventChannel>* reader)
			{
				getDefault(m_mouse,reader);
			}
			void getDefaultKeyboard(ChannelReader<nbl::ui::IKeyboardEventChannel>* reader)
			{
				getDefault(m_keyboard,reader);
			}

		private:
			friend class CommonAPIEventCallback;
			template<class ChannelType>
			void add(Channels<ChannelType>& channels, nbl::core::smart_refctd_ptr<ChannelType>&& channel)
			{
				std::unique_lock lock(channels.lock);
				channels.channels.push_back(std::move(channel));
				
				using namespace std::chrono;
				auto timeStamp = duration_cast<microseconds>(steady_clock::now().time_since_epoch());
				channels.timeStamps.push_back(timeStamp);

				channels.added.notify_all();
			}
			template<class ChannelType>
			void remove(Channels<ChannelType>& channels, const ChannelType* channel)
			{
				std::unique_lock lock(channels.lock);

				auto to_remove_itr = std::find_if(
						channels.channels.begin(),channels.channels.end(),[channel](const auto& chan)->bool{return chan.get()==channel;}
				);

				auto index = std::distance(channels.channels.begin(), to_remove_itr);

				channels.timeStamps.erase(channels.timeStamps.begin() + index);
				channels.channels.erase(to_remove_itr);
			}
			template<class ChannelType>
			void getDefault(Channels<ChannelType>& channels, ChannelReader<ChannelType>* reader)
			{
				/*
				* TODO: Improve default device switching.
				* For nice results, we should actually make a multi-channel reader,
				* and then keep a consumed counter together with a last consumed event from each channel.
				* If there is considerable pause in events received by our current chosen channel or
				* we can detect some other channel of the same "compatible class" is producing more events,
				* Switch the channel choice, but prune away all events younger than the old default's consumption timestamp.
				* (Basically switch keyboards but dont try to process events older than the events you've processed from the old keyboard)
				*/
				
				std::unique_lock lock(channels.lock);
				while (channels.channels.empty())
				{
					m_logger.log("Waiting For Input Device to be connected...",nbl::system::ILogger::ELL_INFO);
					channels.added.wait(lock);
				}
				
				uint64_t consumedCounter = 0ull;

				using namespace std::chrono;
				constexpr long long DefaultChannelTimeoutInMicroSeconds = 100*1e3; // 100 mili-seconds
				auto nowTimeStamp = duration_cast<microseconds>(steady_clock::now().time_since_epoch());

				// Update Timestamp of all channels
				for(uint32_t ch = 0u; ch < channels.channels.size(); ++ch) {
					auto & channel = channels.channels[ch];
					auto & timeStamp = channels.timeStamps[ch];
					auto events = channel->getEvents();
					if(events.size() > 0) {
						auto lastEventTimeStamp = (*(events.end() - 1)).timeStamp; // last event timestamp
						timeStamp = lastEventTimeStamp;
					}
				}

				auto defaultIdx = channels.defaultChannelIndex;
				if(defaultIdx >= channels.channels.size()) {
					defaultIdx = 0;
				}
				auto defaultChannel = channels.channels[defaultIdx];
				auto defaultChannelEvents = defaultChannel->getEvents();
				auto timeDiff = (nowTimeStamp - channels.timeStamps[defaultIdx]).count();
				
				constexpr size_t RewindBackEvents = 50u;

				// If the current one hasn't been active for a while
				if(defaultChannel->empty()) {
					if(timeDiff > DefaultChannelTimeoutInMicroSeconds) {
						// Look for the most active channel (the channel which has got the most events recently)
						auto newDefaultIdx = defaultIdx;
						microseconds maxEventTimeStamp = microseconds(0);

						for(uint32_t chIdx = 0; chIdx < channels.channels.size(); ++chIdx) {
							if(defaultIdx != chIdx) 
							{
								auto channelTimeDiff = (nowTimeStamp - channels.timeStamps[chIdx]).count();
								// Check if was more recently active than the current most active
								if(channelTimeDiff < DefaultChannelTimeoutInMicroSeconds)
								{
									auto & channel = channels.channels[chIdx];
									auto channelEvents = channel->getEvents();
									auto channelEventSize = channelEvents.size();
									const auto frontBufferCapacity = channel->getFrontBufferCapacity();

									size_t rewindBack = std::min(RewindBackEvents, frontBufferCapacity);
									rewindBack = std::min(rewindBack, channelEventSize);

									auto oldEvent = *(channelEvents.end() - rewindBack);

									// Which oldEvent of channels are most recent.
									if(oldEvent.timeStamp > maxEventTimeStamp) {
										maxEventTimeStamp = oldEvent.timeStamp;
										newDefaultIdx = chIdx;
									}
								}
							}
						}

						if(defaultIdx != newDefaultIdx) {
							m_logger.log("Default InputChannel for ChannelType changed from %u to %u",nbl::system::ILogger::ELL_INFO, defaultIdx, newDefaultIdx);

							defaultIdx = newDefaultIdx;
							channels.defaultChannelIndex = newDefaultIdx;
							defaultChannel = channels.channels[newDefaultIdx];
							
							consumedCounter = defaultChannel->getEvents().size() - defaultChannel->getFrontBufferCapacity(); // to not get overflow in reader when consuming.
						}
					}
				}

				if (reader->channel==defaultChannel)
					return;

				reader->channel = defaultChannel;
				reader->consumedCounter = consumedCounter;
			}

			nbl::system::logger_opt_smart_ptr m_logger;
			Channels<nbl::ui::IMouseEventChannel> m_mouse;
			Channels<nbl::ui::IKeyboardEventChannel> m_keyboard;
	};

	// TODO: can you guys just use one callback!?
	class ICommonAPIEventCallback : public nbl::ui::IWindow::IEventCallback
	{
	public:
		virtual void setLogger(nbl::system::logger_opt_smart_ptr& logger) = 0;
	};
	class CTemporaryEventCallback : public ICommonAPIEventCallback
	{
		nbl::system::logger_opt_smart_ptr m_logger = nullptr;
	public:
		void setLogger(nbl::system::logger_opt_smart_ptr& logger) override
		{
			m_logger = logger;
		}
	private:
		bool onWindowShown_impl() override
		{
			m_logger.log("Window Shown");
			return true;
		}
		bool onWindowHidden_impl() override
		{
			m_logger.log("Window hidden");
			return true;
		}
		bool onWindowMoved_impl(int32_t x, int32_t y) override
		{
			m_logger.log("Window window moved to { %d, %d }", nbl::system::ILogger::ELL_WARNING, x, y);
			return true;
		}
		bool onWindowResized_impl(uint32_t w, uint32_t h) override
		{
			m_logger.log("Window resized to { %u, %u }", nbl::system::ILogger::ELL_DEBUG, w, h);
			return true;
		}
		bool onWindowMinimized_impl() override
		{
			m_logger.log("Window minimized", nbl::system::ILogger::ELL_ERROR);
			return true;
		}
		bool onWindowMaximized_impl() override
		{
			m_logger.log("Window maximized", nbl::system::ILogger::ELL_PERFORMANCE);
			return true;
		}
		void onGainedMouseFocus_impl() override
		{
			m_logger.log("Window gained mouse focus", nbl::system::ILogger::ELL_INFO);
		}
		void onLostMouseFocus_impl() override
		{
			m_logger.log("Window lost mouse focus", nbl::system::ILogger::ELL_INFO);
		}
		void onGainedKeyboardFocus_impl() override
		{
			m_logger.log("Window gained keyboard focus", nbl::system::ILogger::ELL_INFO);
		}
		void onLostKeyboardFocus_impl() override
		{
			m_logger.log("Window lost keyboard focus", nbl::system::ILogger::ELL_INFO);
		}
		void onMouseConnected_impl(nbl::core::smart_refctd_ptr<nbl::ui::IMouseEventChannel>&& mch) override
		{
			m_logger.log("A mouse has been connected", nbl::system::ILogger::ELL_INFO);
		}
		void onMouseDisconnected_impl(nbl::ui::IMouseEventChannel* mch) override
		{
			m_logger.log("A mouse has been disconnected", nbl::system::ILogger::ELL_INFO);
		}
		void onKeyboardConnected_impl(nbl::core::smart_refctd_ptr<nbl::ui::IKeyboardEventChannel>&& kbch) override
		{
			m_logger.log("A keyboard has been connected", nbl::system::ILogger::ELL_INFO);
		}
		void onKeyboardDisconnected_impl(nbl::ui::IKeyboardEventChannel* kbch) override
		{
			m_logger.log("A keyboard has been disconnected", nbl::system::ILogger::ELL_INFO);
		}

		bool onWindowClosed_impl() override
		{
			m_logger.log("Window closed");
			return true;
		}
	};
	class CommonAPIEventCallback : public ICommonAPIEventCallback
	{
	public:
		CommonAPIEventCallback(nbl::core::smart_refctd_ptr<InputSystem>&& inputSystem, nbl::system::logger_opt_smart_ptr&& logger) : m_inputSystem(std::move(inputSystem)), m_logger(std::move(logger)), m_gotWindowClosedMsg(false){}
		CommonAPIEventCallback() {}
		bool isWindowOpen() const {return !m_gotWindowClosedMsg;}
		void setLogger(nbl::system::logger_opt_smart_ptr& logger) override
		{
			m_logger = logger;
		}
		void setInputSystem(nbl::core::smart_refctd_ptr<InputSystem>&& inputSystem)
		{
			m_inputSystem = std::move(inputSystem);
		}
	private:
		bool onWindowShown_impl() override
		{
			m_logger.log("Window Shown");
			return true;
		}
		bool onWindowHidden_impl() override
		{
			m_logger.log("Window hidden");
			return true;
		}
		bool onWindowMoved_impl(int32_t x, int32_t y) override
		{
			m_logger.log("Window window moved to { %d, %d }", nbl::system::ILogger::ELL_WARNING, x, y);
			return true;
		}
		bool onWindowResized_impl(uint32_t w, uint32_t h) override
		{
			m_logger.log("Window resized to { %u, %u }", nbl::system::ILogger::ELL_DEBUG, w, h);
			return true;
		}
		bool onWindowMinimized_impl() override
		{
			m_logger.log("Window minimized", nbl::system::ILogger::ELL_ERROR);
			return true;
		}
		bool onWindowMaximized_impl() override
		{
			m_logger.log("Window maximized", nbl::system::ILogger::ELL_PERFORMANCE);
			return true;
		}
		void onGainedMouseFocus_impl() override
		{
			m_logger.log("Window gained mouse focus", nbl::system::ILogger::ELL_INFO);
		}
		void onLostMouseFocus_impl() override
		{
			m_logger.log("Window lost mouse focus", nbl::system::ILogger::ELL_INFO);
		}
		void onGainedKeyboardFocus_impl() override
		{
			m_logger.log("Window gained keyboard focus", nbl::system::ILogger::ELL_INFO);
		}
		void onLostKeyboardFocus_impl() override
		{
			m_logger.log("Window lost keyboard focus", nbl::system::ILogger::ELL_INFO);
		}
		
		bool onWindowClosed_impl() override
		{
			m_logger.log("Window closed");
			m_gotWindowClosedMsg = true;
			return true;
		}

		void onMouseConnected_impl(nbl::core::smart_refctd_ptr<nbl::ui::IMouseEventChannel>&& mch) override
		{
			m_logger.log("A mouse %p has been connected", nbl::system::ILogger::ELL_INFO, mch);
			m_inputSystem.get()->add(m_inputSystem.get()->m_mouse,std::move(mch));
		}
		void onMouseDisconnected_impl(nbl::ui::IMouseEventChannel* mch) override
		{
			m_logger.log("A mouse %p has been disconnected", nbl::system::ILogger::ELL_INFO, mch);
			m_inputSystem.get()->remove(m_inputSystem.get()->m_mouse,mch);
		}
		void onKeyboardConnected_impl(nbl::core::smart_refctd_ptr<nbl::ui::IKeyboardEventChannel>&& kbch) override
		{
			m_logger.log("A keyboard %p has been connected", nbl::system::ILogger::ELL_INFO, kbch);
			m_inputSystem.get()->add(m_inputSystem.get()->m_keyboard,std::move(kbch));
		}
		void onKeyboardDisconnected_impl(nbl::ui::IKeyboardEventChannel* kbch) override
		{
			m_logger.log("A keyboard %p has been disconnected", nbl::system::ILogger::ELL_INFO, kbch);
			m_inputSystem.get()->remove(m_inputSystem.get()->m_keyboard,kbch);
		}

	private:
		nbl::core::smart_refctd_ptr<InputSystem> m_inputSystem = nullptr;
		nbl::system::logger_opt_smart_ptr m_logger = nullptr;
		bool m_gotWindowClosedMsg;
	};

	static nbl::core::smart_refctd_ptr<nbl::system::ISystem> createSystem()
	{
		using namespace nbl;
		using namespace system;
		nbl::core::smart_refctd_ptr<nbl::system::ISystemCaller> caller = nullptr;
#ifdef _NBL_PLATFORM_WINDOWS_
		caller = nbl::core::make_smart_refctd_ptr<nbl::system::CSystemCallerWin32>();
#endif
		return nbl::core::make_smart_refctd_ptr<nbl::system::ISystem>(std::move(caller));
	}
	
	struct QueueFamilyProps
	{
		static constexpr uint32_t InvalidIndex = ~0u;
		uint32_t index                  = InvalidIndex;
		bool supportsGraphics           : 1;
		bool supportsCompute            : 1;
		bool supportsTransfer           : 1;
		bool supportsSparseBinding      : 1;
		bool supportsPresent            : 1;
		bool supportsProtected          : 1;
	};

	struct GPUInfo
	{
		std::vector<nbl::video::ISurface::SFormat> availableSurfaceFormats;
		nbl::video::ISurface::E_PRESENT_MODE availablePresentModes;
		nbl::video::ISurface::SCapabilities surfaceCapabilities;

		struct
		{
			QueueFamilyProps graphics;
			QueueFamilyProps compute;
			QueueFamilyProps transfer;
			QueueFamilyProps present;
		} queueFamilyProps;

		bool hasSurfaceCapabilities = false;
		bool isSwapChainSupported = false;
	};

	static std::vector<GPUInfo> extractGPUInfos(nbl::core::SRange<nbl::video::IPhysicalDevice* const> gpus, nbl::core::smart_refctd_ptr<nbl::video::ISurface> surface)
	{
		using namespace nbl;
		using namespace nbl::video;

		std::vector<GPUInfo> extractedInfos = std::vector<GPUInfo>(gpus.size());

		for (size_t i = 0ull; i < gpus.size(); ++i)
		{
			auto & extractedInfo = extractedInfos[i];
			auto gpu = gpus.begin()[i];

			// Find required queue family indices
			{
				const auto& queueFamilyProperties = gpu->getQueueFamilyProperties();

				for (uint32_t familyIndex = 0u; familyIndex < queueFamilyProperties.size(); ++familyIndex)
				{
					const auto& familyProperty = queueFamilyProperties.begin()[familyIndex];
					auto& outFamilyProp = extractedInfo.queueFamilyProps;

					if(familyProperty.queueCount <= 0)
						continue;

					bool supportsPresent = surface && surface->isSupportedForPhysicalDevice(gpu, familyIndex);
					bool hasGraphicsFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_GRAPHICS_BIT).value != 0;
					bool hasComputeFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_COMPUTE_BIT).value != 0;
					bool hasTransferFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_TRANSFER_BIT).value != 0;
					bool hasSparseBindingFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_SPARSE_BINDING_BIT).value != 0;
					bool hasProtectedFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_PROTECTED_BIT).value != 0;

					// Select Unique queues indices for each queue type (Try to get different "queue index"s for each queue type)
					// Later we can decide if we want them seperate or together. (EXCLUSIVE/CONCURRENT)
					
					// Graphics
					if(hasGraphicsFlag && (outFamilyProp.graphics.index == QueueFamilyProps::InvalidIndex || outFamilyProp.graphics.supportsPresent == false))
					{
						outFamilyProp.graphics.index = familyIndex;
						outFamilyProp.graphics.supportsGraphics = hasGraphicsFlag;
						outFamilyProp.graphics.supportsCompute = hasComputeFlag;
						outFamilyProp.graphics.supportsTransfer = hasTransferFlag;
						outFamilyProp.graphics.supportsSparseBinding = hasSparseBindingFlag;
						outFamilyProp.graphics.supportsPresent = supportsPresent;
						outFamilyProp.graphics.supportsProtected = hasProtectedFlag;
					}
					
					// Present
					if(supportsPresent && (outFamilyProp.present.index == QueueFamilyProps::InvalidIndex || outFamilyProp.present.supportsGraphics == false))
					{
						outFamilyProp.present.index = familyIndex;
						outFamilyProp.present.supportsGraphics = hasGraphicsFlag;
						outFamilyProp.present.supportsCompute = hasComputeFlag;
						outFamilyProp.present.supportsTransfer = hasTransferFlag;
						outFamilyProp.present.supportsSparseBinding = hasSparseBindingFlag;
						outFamilyProp.present.supportsPresent = supportsPresent;
						outFamilyProp.present.supportsProtected = hasProtectedFlag;
					}

					// Compute
					if(hasComputeFlag && !hasGraphicsFlag && outFamilyProp.compute.index == QueueFamilyProps::InvalidIndex)
					{
						outFamilyProp.compute.index = familyIndex;
						outFamilyProp.compute.supportsGraphics = hasGraphicsFlag;
						outFamilyProp.compute.supportsCompute = hasComputeFlag;
						outFamilyProp.compute.supportsTransfer = hasTransferFlag;
						outFamilyProp.compute.supportsSparseBinding = hasSparseBindingFlag;
						outFamilyProp.compute.supportsPresent = supportsPresent;
						outFamilyProp.compute.supportsProtected = hasProtectedFlag;
					}

					// Transfer
					if(hasTransferFlag && !hasGraphicsFlag && !hasComputeFlag && outFamilyProp.transfer.index == QueueFamilyProps::InvalidIndex)
					{
						outFamilyProp.transfer.index = familyIndex;
						outFamilyProp.transfer.supportsGraphics = hasGraphicsFlag;
						outFamilyProp.transfer.supportsCompute = hasComputeFlag;
						outFamilyProp.transfer.supportsTransfer = hasTransferFlag;
						outFamilyProp.transfer.supportsSparseBinding = hasSparseBindingFlag;
						outFamilyProp.transfer.supportsPresent = supportsPresent;
						outFamilyProp.transfer.supportsProtected = hasProtectedFlag;
					}
				}

				// If Graphics supports Present, then use Graphics for Present
				if(extractedInfo.queueFamilyProps.present.index != extractedInfo.queueFamilyProps.graphics.index && extractedInfo.queueFamilyProps.graphics.supportsPresent) 
				{
					extractedInfo.queueFamilyProps.present = extractedInfo.queueFamilyProps.graphics;
				}

				// If a unique Compute is not found but Graphics supports Compute, then use Graphics for Compute
				if(extractedInfo.queueFamilyProps.compute.index == QueueFamilyProps::InvalidIndex && extractedInfo.queueFamilyProps.graphics.supportsCompute)
				{
					extractedInfo.queueFamilyProps.compute = extractedInfo.queueFamilyProps.graphics;
				}
				
				// If a unique Transfer is not found then use either Graphics or Compute for Transfer (prefer compute)
				if(extractedInfo.queueFamilyProps.transfer.index == QueueFamilyProps::InvalidIndex)
				{
					if(extractedInfo.queueFamilyProps.compute.supportsTransfer)
						extractedInfo.queueFamilyProps.transfer = extractedInfo.queueFamilyProps.compute;
					else if(extractedInfo.queueFamilyProps.graphics.supportsTransfer)
						extractedInfo.queueFamilyProps.transfer = extractedInfo.queueFamilyProps.graphics;
				}
			}

			// Since our workload is not headless compute, a swapchain is mandatory
			extractedInfo.isSwapChainSupported = gpu->isSwapchainSupported();

			// Check if the surface is adequate
			if(surface)
			{
				uint32_t surfaceFormatCount;
				surface->getAvailableFormatsForPhysicalDevice(gpu, surfaceFormatCount, nullptr);
				extractedInfo.availableSurfaceFormats = std::vector<nbl::video::ISurface::SFormat>(surfaceFormatCount);
				surface->getAvailableFormatsForPhysicalDevice(gpu, surfaceFormatCount, extractedInfo.availableSurfaceFormats.data());

				extractedInfo.availablePresentModes = surface->getAvailablePresentModesForPhysicalDevice(gpu);

				// TODO: @achal OpenGL shouldn't fail this
				extractedInfo.surfaceCapabilities = {};
				if (surface->getSurfaceCapabilitiesForPhysicalDevice(gpu, extractedInfo.surfaceCapabilities))
					extractedInfo.hasSurfaceCapabilities = true;
			}
		}

		return extractedInfos;
	}
	
	// TODO: also implement a function:findBestGPU
	// Returns an index into gpus info vector
	static uint32_t findSuitableGPU(const std::vector<GPUInfo>& extractedInfos, const bool graphicsQueueEnable)
	{
		uint32_t ret = ~0u;
		for(uint32_t i = 0; i < extractedInfos.size(); ++i)
		{
			bool isGPUSuitable = false;
			const auto& extractedInfo = extractedInfos[i];

			if(graphicsQueueEnable)
			{
				if ((extractedInfo.queueFamilyProps.graphics.index != QueueFamilyProps::InvalidIndex) &&
					(extractedInfo.queueFamilyProps.compute.index != QueueFamilyProps::InvalidIndex) &&
					(extractedInfo.queueFamilyProps.transfer.index != QueueFamilyProps::InvalidIndex) &&
					(extractedInfo.queueFamilyProps.present.index != QueueFamilyProps::InvalidIndex))
					isGPUSuitable = true;
			}
			else
			{
				if ((extractedInfo.queueFamilyProps.compute.index != QueueFamilyProps::InvalidIndex) &&
					(extractedInfo.queueFamilyProps.transfer.index != QueueFamilyProps::InvalidIndex))
					isGPUSuitable = true;
			}

			if(extractedInfo.isSwapChainSupported == false)
				isGPUSuitable = false;

			if(extractedInfo.hasSurfaceCapabilities == false)
				isGPUSuitable = false;

			if(isGPUSuitable)
			{
				// find the first suitable GPU
				ret = i;
				break;
			}
		}

		if(ret == ~0u)
		{
			//_NBL_DEBUG_BREAK_IF(true);
			ret = 0;
		}

		return ret;
	}

	template<uint32_t sc_image_count> //! input with window creation
	struct InitOutput
	{
		enum E_QUEUE_TYPE
		{
			EQT_GRAPHICS,
			EQT_COMPUTE,
			EQT_TRANSFER_UP,
			EQT_TRANSFER_DOWN,
			EQT_COUNT
		};

		nbl::core::smart_refctd_ptr<nbl::ui::IWindow> window;
		nbl::core::smart_refctd_ptr<CommonAPIEventCallback> windowCb;
		nbl::core::smart_refctd_ptr<nbl::video::IAPIConnection> apiConnection;
		nbl::core::smart_refctd_ptr<nbl::video::ISurface> surface;
		nbl::core::smart_refctd_ptr<nbl::video::IUtilities> utilities;
		nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice;
		nbl::video::IPhysicalDevice* physicalDevice;
		nbl::video::IGPUQueue* mainQueue = nullptr; // it's better to let the user know of the main queue that was used to create result.commandPool and result.cpu2gpu
		std::array<nbl::video::IGPUQueue*, EQT_COUNT> queues = { nullptr, nullptr, nullptr, nullptr };
		nbl::core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain;
		nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass;
		std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>, sc_image_count> fbo;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool> commandPool; // TODO: Multibuffer and reset the commandpools
		nbl::core::smart_refctd_ptr<nbl::system::ISystem> system;
		nbl::core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
		nbl::video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
		nbl::core::smart_refctd_ptr<nbl::system::ILogger> logger;
		nbl::core::smart_refctd_ptr<InputSystem> inputSystem;
	};

	template<> //! input without window creation
	struct InitOutput<0>
	{
		enum E_QUEUE_TYPE
		{
			EQT_COMPUTE,
			EQT_TRANSFER_UP,
			EQT_TRANSFER_DOWN,
			EQT_COUNT
		};
		
		nbl::core::smart_refctd_ptr<nbl::video::IAPIConnection> apiConnection;
		nbl::core::smart_refctd_ptr<nbl::video::IUtilities> utilities;
		nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice;
		nbl::video::IPhysicalDevice* physicalDevice;
		nbl::video::IGPUQueue* mainQueue = nullptr; // it's better to let the user know of the main queue that was used to create result.commandPool and result.cpu2gpu
		std::array<nbl::video::IGPUQueue*, EQT_COUNT> queues = {nullptr, nullptr, nullptr};
		nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool> commandPool; // TODO: Multibuffer and reset the commandpools
		nbl::core::smart_refctd_ptr<nbl::system::ISystem> system;
		nbl::core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
		nbl::video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
		nbl::core::smart_refctd_ptr<nbl::system::ILogger> logger;
		nbl::core::smart_refctd_ptr<InputSystem> inputSystem;
	};

	template<typename AppClassName>
	static void main(int argc, char** argv)
	{
#ifndef _NBL_PLATFORM_ANDROID_
		nbl::system::path CWD = nbl::system::path(argv[0]).parent_path().generic_string() + "/";
		nbl::system::path sharedInputCWD = CWD / "../../media/";
		nbl::system::path sharedOutputCWD = CWD / "../../tmp/";;
		nbl::system::path localInputCWD = CWD / "../assets";
		nbl::system::path localOutputCWD = CWD;
		auto app = nbl::core::make_smart_refctd_ptr<AppClassName>(localInputCWD, localOutputCWD, sharedInputCWD, sharedOutputCWD);

		for (size_t i = 0; i < argc; ++i)
			app->argv.push_back(std::string(argv[i]));

		app->onAppInitialized();
		while (app->keepRunning())
		{
			app->workLoopBody();
		}
		app->onAppTerminated();
#endif
	}

	static void Init(InitOutput<0>& result, nbl::video::E_API_TYPE api_type, const std::string_view app_name)
	{
		using namespace nbl;
		using namespace nbl::video;

#ifdef _NBL_PLATFORM_WINDOWS_
		result.system = createSystem();
		result.logger = nbl::core::make_smart_refctd_ptr<system::CColoredStdoutLoggerWin32>(); // we should let user choose it?
#elif defined(_NBL_PLATFORM_ANDROID_)
		result.logger = nbl::core::make_smart_refctd_ptr<system::CStdoutLoggerAndroid>(); // we should let user choose it?
#endif
		result.inputSystem = nbl::core::make_smart_refctd_ptr<InputSystem>(system::logger_opt_smart_ptr(nbl::core::smart_refctd_ptr(result.logger)));

		if(api_type == EAT_VULKAN) 
		{
			result.apiConnection = nbl::video::CVulkanConnection::create(nbl::core::smart_refctd_ptr(result.system), 0, app_name.data(), true);
		}
		else if(api_type == EAT_OPENGL)
		{
			result.apiConnection = nbl::video::COpenGLConnection::create(nbl::core::smart_refctd_ptr(result.system), 0, app_name.data(), nbl::video::COpenGLDebugCallback(nbl::core::smart_refctd_ptr(result.logger)));
		}
		else if(api_type == EAT_OPENGL_ES)
		{
			result.apiConnection = nbl::video::COpenGLESConnection::create(nbl::core::smart_refctd_ptr(result.system), 0, app_name.data(), nbl::video::COpenGLDebugCallback(nbl::core::smart_refctd_ptr(result.logger)));
		}
		else
		{
			_NBL_TODO();
		}

		auto gpus = result.apiConnection->getPhysicalDevices();
		assert(!gpus.empty());
		auto extractedInfos = extractGPUInfos(gpus, nullptr);
		auto suitableGPUIndex = findSuitableGPU(extractedInfos, false);
		auto gpu = gpus.begin()[suitableGPUIndex];

		const auto& gpuInfo = extractedInfos[suitableGPUIndex];

		float queuePriority = 1.f;
		constexpr uint32_t MaxQueueCount = 4;
		nbl::video::ILogicalDevice::SQueueCreationParams qcp[MaxQueueCount] = {}; 
		
		uint32_t actualQueueCount = 1;
		uint32_t mainQueueFamilyIndex = QueueFamilyProps::InvalidIndex;
		mainQueueFamilyIndex = gpuInfo.queueFamilyProps.compute.index;

		qcp[0].familyIndex = mainQueueFamilyIndex;
		qcp[0].count = 1u;
		qcp[0].flags = static_cast<nbl::video::IGPUQueue::E_CREATE_FLAGS>(0);
		qcp[0].priorities = &queuePriority;

		if(qcp[0].familyIndex != gpuInfo.queueFamilyProps.compute.index)
		{
			qcp[actualQueueCount].flags = static_cast<nbl::video::IGPUQueue::E_CREATE_FLAGS>(0);
			qcp[actualQueueCount].familyIndex = gpuInfo.queueFamilyProps.compute.index;
			qcp[actualQueueCount].count = 1u;
			qcp[actualQueueCount].priorities = &queuePriority;
			actualQueueCount++;
		}
		if(gpuInfo.queueFamilyProps.transfer.index != gpuInfo.queueFamilyProps.compute.index && gpuInfo.queueFamilyProps.transfer.index != gpuInfo.queueFamilyProps.graphics.index)
		{
			qcp[actualQueueCount].flags = static_cast<nbl::video::IGPUQueue::E_CREATE_FLAGS>(0);
			qcp[actualQueueCount].familyIndex = gpuInfo.queueFamilyProps.transfer.index;
			qcp[actualQueueCount].count = 1u;
			qcp[actualQueueCount].priorities = &queuePriority;
			actualQueueCount++;
		}

		nbl::video::ILogicalDevice::SCreationParams dev_params;
		dev_params.queueParamsCount = actualQueueCount;
		dev_params.queueParams = qcp;
		result.logicalDevice = gpu->createLogicalDevice(dev_params);

		result.utilities = nbl::core::make_smart_refctd_ptr<nbl::video::IUtilities>(nbl::core::smart_refctd_ptr(result.logicalDevice));
		
		result.mainQueue = result.logicalDevice->getQueue(mainQueueFamilyIndex, 0);
		result.queues[InitOutput<0>::EQT_COMPUTE] = result.logicalDevice->getQueue(gpuInfo.queueFamilyProps.compute.index, 0);
		result.queues[InitOutput<0>::EQT_TRANSFER_UP] = result.logicalDevice->getQueue(gpuInfo.queueFamilyProps.transfer.index, 0);
		result.queues[InitOutput<0>::EQT_TRANSFER_DOWN] = result.logicalDevice->getQueue(gpuInfo.queueFamilyProps.transfer.index, 0);

		result.renderpass = createRenderpass(result.logicalDevice, asset::EF_B8G8R8A8_UNORM, asset::EF_UNKNOWN);

		result.commandPool = result.logicalDevice->createCommandPool(mainQueueFamilyIndex, IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT);
		assert(result.commandPool);
		result.physicalDevice = gpu;

		result.assetManager = nbl::core::make_smart_refctd_ptr<nbl::asset::IAssetManager>(nbl::core::smart_refctd_ptr(result.system)); // we should let user choose it?

		result.cpu2gpuParams.assetManager = result.assetManager.get();
		result.cpu2gpuParams.device = result.logicalDevice.get();
		result.cpu2gpuParams.finalQueueFamIx = mainQueueFamilyIndex;
		result.cpu2gpuParams.limits = result.physicalDevice->getLimits();
		result.cpu2gpuParams.pipelineCache = nullptr;
		result.cpu2gpuParams.sharingMode = nbl::asset::ESM_EXCLUSIVE;
		result.cpu2gpuParams.utilities = result.utilities.get();

		result.cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_TRANSFER].queue = result.queues[InitOutput<0>::EQT_TRANSFER_UP];
		result.cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_COMPUTE].queue = result.queues[InitOutput<0>::EQT_COMPUTE];
	}

	template<uint32_t window_width, uint32_t window_height, uint32_t sc_image_count, class EventCallback = CommonAPIEventCallback>
	static void Init(InitOutput<sc_image_count>& result, nbl::video::E_API_TYPE api_type, const std::string_view app_name, nbl::asset::E_FORMAT depthFormat = nbl::asset::EF_UNKNOWN, const bool graphicsQueueEnable = true)
	{
		using namespace nbl;
		using namespace nbl::video;

		// TODO: Windows/Linux logger define switch
#ifndef _NBL_PLATFORM_ANDROID_
		auto windowManager = nbl::core::make_smart_refctd_ptr<nbl::ui::CWindowManagerWin32>(); // should we store it in result?
#endif
#ifdef _NBL_PLATFORM_WINDOWS_
		result.system = createSystem();
		result.logger = nbl::core::make_smart_refctd_ptr<system::CColoredStdoutLoggerWin32>(); // we should let user choose it?
#elif defined(_NBL_PLATFORM_ANDROID_)
		result.logger = nbl::core::make_smart_refctd_ptr<system::CStdoutLoggerAndroid>(); // we should let user choose it?
#endif
		result.inputSystem = nbl::core::make_smart_refctd_ptr<InputSystem>(system::logger_opt_smart_ptr(nbl::core::smart_refctd_ptr(result.logger)));

#ifndef _NBL_PLATFORM_ANDROID_
		result.windowCb = nbl::core::make_smart_refctd_ptr<EventCallback>(nbl::core::smart_refctd_ptr(result.inputSystem), system::logger_opt_smart_ptr(nbl::core::smart_refctd_ptr(result.logger)));
		nbl::ui::IWindow::SCreationParams windowsCreationParams;
		windowsCreationParams.width = window_width;
		windowsCreationParams.height = window_height;
		windowsCreationParams.x = 64u;
		windowsCreationParams.y = 64u;
		windowsCreationParams.system = nbl::core::smart_refctd_ptr(result.system);
		windowsCreationParams.flags = nbl::ui::IWindow::ECF_NONE;
		windowsCreationParams.windowCaption = app_name.data();
		windowsCreationParams.callback = result.windowCb;
		
		result.window = windowManager->createWindow(std::move(windowsCreationParams));
		result.windowCb->setInputSystem(nbl::core::smart_refctd_ptr(result.inputSystem));
#else
		result.windowCb = nbl::core::smart_refctd_ptr<EventCallback>((CommonAPIEventCallback*)result.window->getEventCallback());
		result.windowCb->setInputSystem(core::smart_refctd_ptr(result.inputSystem));
		//result.window->setEventCallback(core::smart_refctd_ptr(result.windowCb));
#endif
		if(api_type == EAT_VULKAN) 
		{
			auto _apiConnection = nbl::video::CVulkanConnection::create(nbl::core::smart_refctd_ptr(result.system), 0, app_name.data(), true);
#ifdef _NBL_PLATFORM_WINDOWS_
			result.surface = nbl::video::CSurfaceVulkanWin32::create(nbl::core::smart_refctd_ptr(_apiConnection), nbl::core::smart_refctd_ptr<ui::IWindowWin32>(static_cast<ui::IWindowWin32*>(result.window.get())));
#elif defined(_NBL_PLATFORM_ANDROID_)
			////result.surface = nbl::video::CSurfaceVulkanAndroid::create(nbl::core::smart_refctd_ptr(_apiConnection), nbl::core::smart_refctd_ptr<ui::IWindowAndroid>(static_cast<ui::IWindowAndroid*>(result.window.get())));
#endif
			result.apiConnection = _apiConnection;
		}
		else if(api_type == EAT_OPENGL)
		{
			auto _apiConnection = nbl::video::COpenGLConnection::create(nbl::core::smart_refctd_ptr(result.system), 0, app_name.data(), nbl::video::COpenGLDebugCallback(nbl::core::smart_refctd_ptr(result.logger)));
#ifdef _NBL_PLATFORM_WINDOWS_
			result.surface = nbl::video::CSurfaceGLWin32::create(nbl::core::smart_refctd_ptr(_apiConnection), nbl::core::smart_refctd_ptr<ui::IWindowWin32>(static_cast<ui::IWindowWin32*>(result.window.get())));
#elif defined(_NBL_PLATFORM_ANDROID_)
			result.surface = nbl::video::CSurfaceGLAndroid::create(nbl::core::smart_refctd_ptr(_apiConnection), nbl::core::smart_refctd_ptr<ui::IWindowAndroid>(static_cast<ui::IWindowAndroid*>(result.window.get())));
#endif
			result.apiConnection = _apiConnection;
		}
		else if(api_type == EAT_OPENGL_ES)
		{
			auto _apiConnection = nbl::video::COpenGLESConnection::create(nbl::core::smart_refctd_ptr(result.system), 0, app_name.data(), nbl::video::COpenGLDebugCallback(nbl::core::smart_refctd_ptr(result.logger)));
#ifdef _NBL_PLATFORM_WINDOWS_
			result.surface = nbl::video::CSurfaceGLWin32::create(nbl::core::smart_refctd_ptr(_apiConnection), nbl::core::smart_refctd_ptr<ui::IWindowWin32>(static_cast<ui::IWindowWin32*>(result.window.get())));
#elif defined(_NBL_PLATFORM_ANDROID_)
			result.surface = nbl::video::CSurfaceGLAndroid::create(nbl::core::smart_refctd_ptr(_apiConnection), nbl::core::smart_refctd_ptr<ui::IWindowAndroid>(static_cast<ui::IWindowAndroid*>(result.window.get())));
#endif
			result.apiConnection = _apiConnection;
		}
		else
		{
			_NBL_TODO();
		}

		auto gpus = result.apiConnection->getPhysicalDevices();
		assert(!gpus.empty());
		auto extractedInfos = extractGPUInfos(gpus, result.surface);
		auto suitableGPUIndex = findSuitableGPU(extractedInfos, graphicsQueueEnable);
		auto gpu = gpus.begin()[suitableGPUIndex];
		const auto& gpuInfo = extractedInfos[suitableGPUIndex];

		float queuePriority = 1.f;
		constexpr uint32_t MaxQueueCount = 4;
		nbl::video::ILogicalDevice::SQueueCreationParams qcp[MaxQueueCount] = {}; 
		
		uint32_t actualQueueCount = 1;
		uint32_t mainQueueFamilyIndex = QueueFamilyProps::InvalidIndex;
		if(graphicsQueueEnable)
			mainQueueFamilyIndex = gpuInfo.queueFamilyProps.graphics.index;
		else
			mainQueueFamilyIndex = gpuInfo.queueFamilyProps.compute.index;

		qcp[0].familyIndex = mainQueueFamilyIndex;
		qcp[0].count = 1u;
		qcp[0].flags = static_cast<nbl::video::IGPUQueue::E_CREATE_FLAGS>(0);
		qcp[0].priorities = &queuePriority;

		if(qcp[0].familyIndex != gpuInfo.queueFamilyProps.compute.index)
		{
			qcp[actualQueueCount].flags = static_cast<nbl::video::IGPUQueue::E_CREATE_FLAGS>(0);
			qcp[actualQueueCount].familyIndex = gpuInfo.queueFamilyProps.compute.index;
			qcp[actualQueueCount].count = 1u;
			qcp[actualQueueCount].priorities = &queuePriority;
			actualQueueCount++;
		}
		if(gpuInfo.queueFamilyProps.transfer.index != gpuInfo.queueFamilyProps.compute.index && gpuInfo.queueFamilyProps.transfer.index != gpuInfo.queueFamilyProps.graphics.index)
		{
			qcp[actualQueueCount].flags = static_cast<nbl::video::IGPUQueue::E_CREATE_FLAGS>(0);
			qcp[actualQueueCount].familyIndex = gpuInfo.queueFamilyProps.transfer.index;
			qcp[actualQueueCount].count = 1u;
			qcp[actualQueueCount].priorities = &queuePriority;
			actualQueueCount++;
		}
		if(gpuInfo.queueFamilyProps.present.index != gpuInfo.queueFamilyProps.compute.index &&
			gpuInfo.queueFamilyProps.present.index != gpuInfo.queueFamilyProps.graphics.index &&
			gpuInfo.queueFamilyProps.present.index != gpuInfo.queueFamilyProps.transfer.index )
		{
			qcp[actualQueueCount].flags = static_cast<nbl::video::IGPUQueue::E_CREATE_FLAGS>(0);
			qcp[actualQueueCount].familyIndex = gpuInfo.queueFamilyProps.present.index;
			qcp[actualQueueCount].count = 1u;
			qcp[actualQueueCount].priorities = &queuePriority;
			actualQueueCount++;
		}

		nbl::video::ILogicalDevice::SCreationParams dev_params;
		dev_params.queueParamsCount = actualQueueCount;
		dev_params.queueParams = qcp;
		result.logicalDevice = gpu->createLogicalDevice(dev_params);

		result.utilities = nbl::core::make_smart_refctd_ptr<nbl::video::IUtilities>(nbl::core::smart_refctd_ptr(result.logicalDevice));

		result.mainQueue = result.logicalDevice->getQueue(mainQueueFamilyIndex, 0);
		if(graphicsQueueEnable)
			result.queues[InitOutput<sc_image_count>::EQT_GRAPHICS] = result.logicalDevice->getQueue(gpuInfo.queueFamilyProps.graphics.index, 0);
		result.queues[InitOutput<sc_image_count>::EQT_COMPUTE] = result.logicalDevice->getQueue(gpuInfo.queueFamilyProps.compute.index, 0);
		result.queues[InitOutput<sc_image_count>::EQT_TRANSFER_UP] = result.logicalDevice->getQueue(gpuInfo.queueFamilyProps.transfer.index, 0);
		result.queues[InitOutput<sc_image_count>::EQT_TRANSFER_DOWN] = result.logicalDevice->getQueue(gpuInfo.queueFamilyProps.transfer.index, 0);

		nbl::video::ISurface::SFormat requestedFormat;
		if(api_type == EAT_VULKAN)
		{
			requestedFormat.format = asset::EF_B8G8R8A8_UNORM;
			requestedFormat.colorSpace.eotf = asset::EOTF_sRGB;
			requestedFormat.colorSpace.primary = asset::ECP_SRGB;
		}
		else
		{
			// Temporary to make previous examples work
			requestedFormat.format = asset::EF_R8G8B8A8_SRGB;
			requestedFormat.colorSpace.eotf = asset::EOTF_sRGB;
			requestedFormat.colorSpace.primary = asset::ECP_SRGB;
		}

		result.swapchain = createSwapchain(api_type, window_width, window_height, sc_image_count, result.logicalDevice, result.surface, nbl::video::ISurface::EPM_FIFO_RELAXED, requestedFormat, gpuInfo);
		assert(result.swapchain);
		
		asset::E_FORMAT swapChainFormat = result.swapchain->getCreationParameters().surfaceFormat.format;
		result.renderpass = createRenderpass(result.logicalDevice, swapChainFormat, depthFormat);

		result.fbo = createFBOWithSwapchainImages<sc_image_count, window_width, window_height>(result.logicalDevice, result.swapchain, result.renderpass, depthFormat);

		result.commandPool = result.logicalDevice->createCommandPool(mainQueueFamilyIndex, IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT);
		assert(result.commandPool);
		result.physicalDevice = gpu;

		result.assetManager = nbl::core::make_smart_refctd_ptr<nbl::asset::IAssetManager>(nbl::core::smart_refctd_ptr(result.system)); // we should let user choose it?
		
		result.cpu2gpuParams.assetManager = result.assetManager.get();
		result.cpu2gpuParams.device = result.logicalDevice.get();
		result.cpu2gpuParams.finalQueueFamIx = mainQueueFamilyIndex;
		result.cpu2gpuParams.limits = result.physicalDevice->getLimits();
		result.cpu2gpuParams.pipelineCache = nullptr;
		result.cpu2gpuParams.sharingMode = nbl::asset::ESM_EXCLUSIVE;
		result.cpu2gpuParams.utilities = result.utilities.get();


		// TODO: Temprory Fix Because cpu2gpuParams needs GraphicsPipeline (but doesn't take input for usages like blitImage)
		if(graphicsQueueEnable && gpuInfo.queueFamilyProps.graphics.supportsTransfer)
			result.cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_TRANSFER].queue = result.queues[InitOutput<sc_image_count>::EQT_GRAPHICS];
		else
			result.cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_TRANSFER].queue = result.queues[InitOutput<sc_image_count>::EQT_TRANSFER_UP];

		if(graphicsQueueEnable && gpuInfo.queueFamilyProps.graphics.supportsCompute)
			result.cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_COMPUTE].queue = result.queues[InitOutput<sc_image_count>::EQT_GRAPHICS];
		else
			result.cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_COMPUTE].queue = result.queues[InitOutput<sc_image_count>::EQT_COMPUTE];
	}
	static nbl::core::smart_refctd_ptr<nbl::video::ISwapchain> createSwapchain(
		nbl::video::E_API_TYPE api_type,
		uint32_t width,
		uint32_t height,
		uint32_t imageCount,
		const nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice>& device,
		const nbl::core::smart_refctd_ptr<nbl::video::ISurface>& surface,
		nbl::video::ISurface::E_PRESENT_MODE requestedPresentMode,
		nbl::video::ISurface::SFormat requestedSurfaceFormat,
		const GPUInfo& gpuInfo)
	{
		using namespace nbl;

		nbl::asset::E_SHARING_MODE imageSharingMode;
		if (gpuInfo.queueFamilyProps.graphics.index == gpuInfo.queueFamilyProps.present.index)
			imageSharingMode = asset::ESM_EXCLUSIVE;
		else
			imageSharingMode = asset::ESM_CONCURRENT;
		
		nbl::video::ISurface::SFormat surfaceFormat;

		if(api_type == nbl::video::EAT_VULKAN)
		{
			uint32_t found_format_and_colorspace = ~0u;
			uint32_t found_format = ~0u;
			for(uint32_t i = 0; i < gpuInfo.availableSurfaceFormats.size(); ++i)
			{
				const auto & supportedFormat = gpuInfo.availableSurfaceFormats[i];
				if(requestedSurfaceFormat.format == supportedFormat.format)
				{
					if(found_format == ~0u)
						found_format = i;
					if(requestedSurfaceFormat.colorSpace.eotf == supportedFormat.colorSpace.eotf && requestedSurfaceFormat.colorSpace.primary == supportedFormat.colorSpace.primary)
					{
						found_format_and_colorspace = i;
						break;
					}
				}
			}
		
			if(found_format_and_colorspace != ~0u)
			{
				surfaceFormat = gpuInfo.availableSurfaceFormats[found_format_and_colorspace];
			}
			else if(found_format != ~0u) // fallback
			{
				assert(false && "Fallback: requested 'colorspace' is not supported.");
				surfaceFormat = gpuInfo.availableSurfaceFormats[found_format];
			}
			else
			{
				assert(false && "Fallback: requested 'format' and 'colorspace' is not supported.");
				surfaceFormat = gpuInfo.availableSurfaceFormats[0];
			}

			bool presentModeSupported = (gpuInfo.availablePresentModes & requestedPresentMode) != 0;
			assert(presentModeSupported);
			if(!presentModeSupported) // fallback 
			{
				requestedPresentMode = nbl::video::ISurface::E_PRESENT_MODE::EPM_FIFO;
				assert(false && "Fallback: requested 'present mode' is not supported");
			}
		}
		else
		{
			surfaceFormat = requestedSurfaceFormat;
		}

		nbl::video::ISwapchain::SCreationParams sc_params = {};
		sc_params.width = width;
		sc_params.height = height;
		sc_params.arrayLayers = 1u;
		sc_params.minImageCount = imageCount;
		sc_params.presentMode = requestedPresentMode;
		sc_params.imageUsage = static_cast<asset::IImage::E_USAGE_FLAGS>(asset::IImage::EUF_COLOR_ATTACHMENT_BIT | asset::IImage::EUF_STORAGE_BIT | asset::IImage::EUF_TRANSFER_DST_BIT);;
		sc_params.surface = surface;
		sc_params.imageSharingMode = imageSharingMode;
		sc_params.surfaceFormat = surfaceFormat;

		return device->createSwapchain(std::move(sc_params));
	}
	static nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> createRenderpass(const nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice>& device, nbl::asset::E_FORMAT colorAttachmentFormat = nbl::asset::EF_UNKNOWN, nbl::asset::E_FORMAT depthFormat = nbl::asset::EF_UNKNOWN)
	{
		using namespace nbl;

		bool useDepth = asset::isDepthOrStencilFormat(depthFormat);

		nbl::video::IGPURenderpass::SCreationParams::SAttachmentDescription attachments[2];
		attachments[0].initialLayout = asset::EIL_UNDEFINED;
		attachments[0].finalLayout = asset::EIL_PRESENT_SRC_KHR;
		attachments[0].format = colorAttachmentFormat;
		attachments[0].samples = asset::IImage::ESCF_1_BIT;
		attachments[0].loadOp = nbl::video::IGPURenderpass::ELO_CLEAR;
		attachments[0].storeOp = nbl::video::IGPURenderpass::ESO_STORE;

		attachments[1].initialLayout = asset::EIL_UNDEFINED;
		attachments[1].finalLayout = asset::EIL_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		attachments[1].format = depthFormat;
		attachments[1].samples = asset::IImage::ESCF_1_BIT;
		attachments[1].loadOp = nbl::video::IGPURenderpass::ELO_CLEAR;
		attachments[1].storeOp = nbl::video::IGPURenderpass::ESO_STORE;

		nbl::video::IGPURenderpass::SCreationParams::SSubpassDescription::SAttachmentRef colorAttRef;
		colorAttRef.attachment = 0u;
		colorAttRef.layout = asset::EIL_COLOR_ATTACHMENT_OPTIMAL;

		nbl::video::IGPURenderpass::SCreationParams::SSubpassDescription::SAttachmentRef depthStencilAttRef;
		depthStencilAttRef.attachment = 1u;
		depthStencilAttRef.layout = asset::EIL_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		nbl::video::IGPURenderpass::SCreationParams::SSubpassDescription sp;
		sp.pipelineBindPoint = asset::EPBP_GRAPHICS;
		sp.colorAttachmentCount = 1u;
		sp.colorAttachments = &colorAttRef;
		if(useDepth) {
			sp.depthStencilAttachment = &depthStencilAttRef;
		} else {
			sp.depthStencilAttachment = nullptr;
		}
		sp.flags = nbl::video::IGPURenderpass::ESDF_NONE;
		sp.inputAttachmentCount = 0u;
		sp.inputAttachments = nullptr;
		sp.preserveAttachmentCount = 0u;
		sp.preserveAttachments = nullptr;
		sp.resolveAttachments = nullptr;

		nbl::video::IGPURenderpass::SCreationParams rp_params;
		rp_params.attachmentCount = (useDepth) ? 2u : 1u;
		rp_params.attachments = attachments;
		rp_params.dependencies = nullptr;
		rp_params.dependencyCount = 0u;
		rp_params.subpasses = &sp;
		rp_params.subpassCount = 1u;

		return device->createGPURenderpass(rp_params);
	}

	template<size_t imageCount, size_t width, size_t height>
	static auto createFBOWithSwapchainImages(const nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice>& device,
		nbl::core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain,
		nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass, 
		nbl::asset::E_FORMAT depthFormat = nbl::asset::EF_UNKNOWN)->std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>, imageCount>
	{
		using namespace nbl;
		bool useDepth = asset::isDepthOrStencilFormat(depthFormat);
		std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>, imageCount> fbo;
		auto sc_images = swapchain->getImages();
		assert(sc_images.size() == imageCount);
		for (uint32_t i = 0u; i < imageCount; ++i)
		{
			nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> view[2] = {};
			
			auto img = sc_images.begin()[i];
			{
				nbl::video::IGPUImageView::SCreationParams view_params;
				view_params.format = img->getCreationParameters().format;
				view_params.viewType = asset::IImageView<nbl::video::IGPUImage>::ET_2D;
				view_params.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
				view_params.subresourceRange.baseMipLevel = 0u;
				view_params.subresourceRange.levelCount = 1u;
				view_params.subresourceRange.baseArrayLayer = 0u;
				view_params.subresourceRange.layerCount = 1u;
				view_params.image = std::move(img);

				view[0] = device->createGPUImageView(std::move(view_params));
				assert(view[0]);
			}
			
			if(useDepth) {
				nbl::video::IGPUImage::SCreationParams imgParams;
				imgParams.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);
				imgParams.type = asset::IImage::ET_2D;
				imgParams.format = depthFormat;
				imgParams.extent = {width, height, 1};
				imgParams.usage = asset::IImage::E_USAGE_FLAGS::EUF_DEPTH_STENCIL_ATTACHMENT_BIT;
				imgParams.mipLevels = 1u;
				imgParams.arrayLayers = 1u;
				imgParams.samples = asset::IImage::ESCF_1_BIT;
				nbl::core::smart_refctd_ptr<nbl::video::IGPUImage> depthImg = device->createDeviceLocalGPUImageOnDedMem(std::move(imgParams));

				nbl::video::IGPUImageView::SCreationParams view_params;
				view_params.format = depthFormat;
				view_params.viewType = asset::IImageView<nbl::video::IGPUImage>::ET_2D;
				view_params.subresourceRange.aspectMask = asset::IImage::EAF_DEPTH_BIT;
				view_params.subresourceRange.baseMipLevel = 0u;
				view_params.subresourceRange.levelCount = 1u;
				view_params.subresourceRange.baseArrayLayer = 0u;
				view_params.subresourceRange.layerCount = 1u;
				view_params.image = std::move(depthImg);

				view[1] = device->createGPUImageView(std::move(view_params));
				assert(view[1]);
			}

			nbl::video::IGPUFramebuffer::SCreationParams fb_params;
			fb_params.width = width;
			fb_params.height = height;
			fb_params.layers = 1u;
			fb_params.renderpass = renderpass;
			fb_params.flags = static_cast<nbl::video::IGPUFramebuffer::E_CREATE_FLAGS>(0);
			fb_params.attachmentCount = (useDepth) ? 2u : 1u;
			fb_params.attachments = view;

			fbo[i] = device->createGPUFramebuffer(std::move(fb_params));
			assert(fbo[i]);
		}
		return fbo;
	}

	static constexpr inline nbl::asset::E_PIPELINE_STAGE_FLAGS DefaultSubmitWaitStage = nbl::asset::EPSF_COLOR_ATTACHMENT_OUTPUT_BIT;
	static void Submit(nbl::video::ILogicalDevice* device,
		nbl::video::ISwapchain* sc,
		nbl::video::IGPUCommandBuffer* cmdbuf,
		nbl::video::IGPUQueue* queue,
		nbl::video::IGPUSemaphore* const waitSemaphore, // usually the image acquire semaphore
		nbl::video::IGPUSemaphore* const renderFinishedSemaphore,
		nbl::video::IGPUFence* fence=nullptr,
		const nbl::core::bitflag<nbl::asset::E_PIPELINE_STAGE_FLAGS> waitDstStageMask=DefaultSubmitWaitStage // only matters if `waitSemaphore` not null
	)
	{
		using namespace nbl;
		nbl::video::IGPUQueue::SSubmitInfo submit;
		{
			submit.commandBufferCount = 1u;
			submit.commandBuffers = &cmdbuf;
			submit.waitSemaphoreCount = waitSemaphore ? 1u:0u;
			submit.pWaitSemaphores = &waitSemaphore;
			submit.pWaitDstStageMask = &waitDstStageMask.value;
			submit.signalSemaphoreCount = renderFinishedSemaphore ? 1u:0u;
			submit.pSignalSemaphores = &renderFinishedSemaphore;

			queue->submit(1u,&submit,fence);
		}
	}

	static void Present(nbl::video::ILogicalDevice* device,
		nbl::video::ISwapchain* sc,
		nbl::video::IGPUQueue* queue,
		nbl::video::IGPUSemaphore* waitSemaphore, // usually the render finished semaphore
		uint32_t imageNum)
	{
		using namespace nbl;
		nbl::video::IGPUQueue::SPresentInfo present;
		{
			present.swapchainCount = 1u;
			present.imgIndices = &imageNum;
			nbl::video::ISwapchain* swapchain = sc;
			present.swapchains = &swapchain;
			present.waitSemaphoreCount = waitSemaphore ? 1u:0u;
			present.waitSemaphores = &waitSemaphore;

			queue->present(present);
		}
	}
	static std::pair<nbl::core::smart_refctd_ptr<nbl::video::IGPUImage>, nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView>> createEmpty2DTexture(
		const nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice>& device,
		uint32_t width,
		uint32_t height,
		nbl::asset::E_FORMAT format = nbl::asset::E_FORMAT::EF_R8G8B8A8_UNORM)
	{
		nbl::video::IGPUImage::SCreationParams gpu_image_params;
		gpu_image_params.mipLevels = 1;
		gpu_image_params.extent = { width, height, 1 };
		gpu_image_params.format = format;
		gpu_image_params.arrayLayers = 1u;
		gpu_image_params.type = nbl::asset::IImage::ET_2D;
		gpu_image_params.samples = nbl::asset::IImage::ESCF_1_BIT;
		gpu_image_params.flags = static_cast<nbl::asset::IImage::E_CREATE_FLAGS>(0u);
		nbl::core::smart_refctd_ptr image = device->createGPUImageOnDedMem(std::move(gpu_image_params), device->getDeviceLocalGPUMemoryReqs());

		nbl::video::IGPUImageView::SCreationParams creation_params;
		creation_params.format = image->getCreationParameters().format;
		creation_params.image = image;
		creation_params.viewType = nbl::video::IGPUImageView::ET_2D;
		creation_params.subresourceRange = { static_cast<nbl::asset::IImage::E_ASPECT_FLAGS>(0u), 0, 1, 0, 1 };
		creation_params.flags = static_cast<nbl::video::IGPUImageView::E_CREATE_FLAGS>(0u);
		nbl::core::smart_refctd_ptr image_view = device->createGPUImageView(std::move(creation_params));
		return std::pair(image, image_view);
	}

	static int getQueueFamilyIndex(const nbl::video::IPhysicalDevice* gpu, uint32_t requiredQueueFlags)
	{
		auto props = gpu->getQueueFamilyProperties();
		int currentIndex = 0;
		for (const auto& property : props)
		{
			if ((property.queueFlags.value & requiredQueueFlags) == requiredQueueFlags)
			{
				return currentIndex;
			}
			++currentIndex;
		}
		return -1;
	}
};