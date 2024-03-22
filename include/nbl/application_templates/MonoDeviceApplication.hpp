// Copyright (C) 2023-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXAMPLES_APPLICATION_TEMPLATES_MONO_DEVICE_APPLICATION_HPP_INCLUDED_
#define _NBL_EXAMPLES_APPLICATION_TEMPLATES_MONO_DEVICE_APPLICATION_HPP_INCLUDED_

// Build on top of the previous one
#include "MonoSystemMonoLoggerApplication.hpp"

namespace nbl::application_templates
{

// Virtual Inheritance because apps might end up doing diamond inheritance
class MonoDeviceApplication : public virtual MonoSystemMonoLoggerApplication
{
		using base_t = MonoSystemMonoLoggerApplication;

	public:
		using base_t::base_t;

		// Just to run destructors in a nice order
		virtual bool onAppTerminated() override
		{
			// break the circular references from queues tracking submit resources
			m_device->waitIdle();
			m_device = nullptr;
			m_api = nullptr;
			return base_t::onAppTerminated();
		}

	protected:
		// need this one for skipping passing all args into ApplicationFramework
		MonoDeviceApplication() = default;

		// This time we build upon the Mono-System and Mono-Logger application and add the choice of a single physical device
		virtual bool onAppInitialized(core::smart_refctd_ptr<system::ISystem>&& system) override
		{
			if (!base_t::onAppInitialized(std::move(system)))
				return false;

			using namespace nbl::core;
			using namespace nbl::video;
			// TODO: specify version of the app
			m_api = CVulkanConnection::create(smart_refctd_ptr(m_system),0,_NBL_APP_NAME_,smart_refctd_ptr(base_t::m_logger),getAPIFeaturesToEnable());
			if (!m_api)
				return logFail("Failed to crate an IAPIConnection!");

			// declaring as auto so we can migrate to span easily later
			auto gpus = m_api->getPhysicalDevices();
			if (gpus.empty())
				return logFail("Failed to find any Nabla Core Profile Vulkan devices!");

			core::set<video::IPhysicalDevice*> suitablePhysicalDevices(gpus.begin(),gpus.end());
			filterDevices(suitablePhysicalDevices);
			if (suitablePhysicalDevices.empty())
				return logFail("No PhysicalDevice met the feature requirements of the application!");

			// we're very constrained by the physical device selection so there's nothing to override here
			{
				m_physicalDevice = selectPhysicalDevice(suitablePhysicalDevices);

				ILogicalDevice::SCreationParams params = {};
				params.queueParams = getQueueCreationParameters(m_physicalDevice->getQueueFamilyProperties());

				bool noQueues = true;
				for (const auto& famParams : params.queueParams)
				if (famParams.count)
					noQueues = false;
				if (noQueues)
					return logFail("Failed to compute queue creation parameters for a Logical Device!");
				
				const auto supportedPreferredFormats = getPreferredDeviceFeatures().intersectWith(m_physicalDevice->getFeatures());
				params.featuresToEnable = getRequiredDeviceFeatures().unionWith(supportedPreferredFormats);

				m_device = m_physicalDevice->createLogicalDevice(std::move(params));
				if (!m_device)
					return logFail("Failed to create a Logical Device!");
			}

			return true;
		}

		// virtual function so you can override as needed for some example father down the line
		virtual video::IAPIConnection::SFeatures getAPIFeaturesToEnable()
		{
			video::IAPIConnection::SFeatures retval = {};
			retval.validations = true;
			// re-enable when https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/7600 gets fixed
			retval.synchronizationValidation = false;
			retval.debugUtils = true;
			return retval;
		}

		// a device filter helps you create a set of physical devices that satisfy your requirements in terms of features, limits etc.
		virtual void filterDevices(core::set<video::IPhysicalDevice*>& physicalDevices) const
		{
			video::SPhysicalDeviceFilter deviceFilter = {};

			deviceFilter.minApiVersion = { 1,3,0 };
			deviceFilter.minConformanceVersion = {1,3,0,0};

			deviceFilter.minimumLimits = getRequiredDeviceLimits();
			deviceFilter.requiredFeatures = getRequiredDeviceFeatures();

			deviceFilter.requiredImageFormatUsagesOptimalTiling = getRequiredOptimalTilingImageUsages();

			const auto memoryReqs = getMemoryRequirements();
			deviceFilter.memoryRequirements = memoryReqs;

			const auto queueReqs = getQueueRequirements();
			deviceFilter.queueRequirements = queueReqs;
			
			deviceFilter(physicalDevices);
		}

		// virtual function so you can override as needed for some example father down the line
		virtual video::SPhysicalDeviceLimits getRequiredDeviceLimits() const
		{
			video::SPhysicalDeviceLimits retval = {};

			retval.subgroupOpsShaderStages = asset::IShader::ESS_COMPUTE;

			return retval;
		}

		// virtual function so you can override as needed for some example father down the line
		virtual video::SPhysicalDeviceFeatures getRequiredDeviceFeatures() const
		{
			video::SPhysicalDeviceFeatures retval = {};

			return retval;
		}

		// Lets declare a few common usages of images
		struct CommonFormatImageUsages
		{
			using usages_t = video::IPhysicalDevice::SFormatImageUsages;
			using format_usage_t = usages_t::SUsage;
			using image_t = nbl::asset::IImage;

			const static inline format_usage_t sampling = format_usage_t(image_t::EUF_SAMPLED_BIT);
			const static inline format_usage_t transferUpAndDown = format_usage_t(image_t::EUF_TRANSFER_DST_BIT|image_t::EUF_TRANSFER_SRC_BIT);
			const static inline format_usage_t shaderStorage = format_usage_t(image_t::EUF_STORAGE_BIT);
			const static inline format_usage_t shaderStorageAtomic = shaderStorage|[]()->auto {format_usage_t tmp; tmp.storageImageAtomic = true; return tmp;}();
			const static inline format_usage_t attachment = []()->auto {format_usage_t tmp; tmp.attachment = true; return tmp; }();
			const static inline format_usage_t attachmentBlend = []()->auto {format_usage_t tmp; tmp.attachmentBlend = true; return tmp; }();
			const static inline format_usage_t blitSrc = []()->auto {format_usage_t tmp; tmp.blitSrc = true; return tmp; }();
			const static inline format_usage_t blitDst = []()->auto {format_usage_t tmp; tmp.blitDst = true; return tmp; }();
			// TODO: redo when we incorporate blits into the asset converter (just sampling then)
			const static inline format_usage_t mipmapGeneration = sampling|blitSrc|blitDst;
			const static inline format_usage_t opaqueRendering = sampling|transferUpAndDown|attachment|mipmapGeneration;
			const static inline format_usage_t genericRendering = opaqueRendering|attachmentBlend|mipmapGeneration;
			const static inline format_usage_t renderingAndStorage = genericRendering|shaderStorage;
		};

		// virtual function so you can override as needed for some example father down the line
		virtual video::IPhysicalDevice::SFormatImageUsages getRequiredOptimalTilingImageUsages() const
		{
			video::IPhysicalDevice::SFormatImageUsages retval = {};
			
			using namespace nbl::asset;
			// we care that certain "basic" formats are usable in some "basic" ways
			retval[EF_R32_UINT] = CommonFormatImageUsages::shaderStorageAtomic;
			retval[EF_R8_UNORM] = CommonFormatImageUsages::genericRendering;
			retval[EF_R8G8_UNORM] = CommonFormatImageUsages::genericRendering;
			retval[EF_R8G8B8A8_UNORM] = CommonFormatImageUsages::genericRendering;
			retval[EF_R8G8B8A8_SRGB] = CommonFormatImageUsages::genericRendering;
			retval[EF_R16_SFLOAT] = CommonFormatImageUsages::renderingAndStorage;
			retval[EF_R16G16_SFLOAT] = CommonFormatImageUsages::renderingAndStorage;
			retval[EF_R16G16B16A16_SFLOAT] = CommonFormatImageUsages::renderingAndStorage;
			retval[EF_R32_SFLOAT] = CommonFormatImageUsages::renderingAndStorage;
			retval[EF_R32G32_SFLOAT] = CommonFormatImageUsages::renderingAndStorage;
			retval[EF_R32G32B32A32_SFLOAT] = CommonFormatImageUsages::renderingAndStorage;

			return retval;
		}

		// virtual function so you can override as needed for some example father down the line
		virtual core::vector<video::SPhysicalDeviceFilter::MemoryRequirement> getMemoryRequirements() const
		{
			using namespace core;
			using namespace video;
			
			vector<SPhysicalDeviceFilter::MemoryRequirement> retval;
			using memory_flags_t = IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS;
			// at least 512 MB of Device Local Memory
			retval.push_back({.size=512<<20,.memoryFlags=memory_flags_t::EMPF_DEVICE_LOCAL_BIT});

			return retval;
		}

		// virtual function so you can override as needed for some example father down the line
		using queue_req_t = video::SPhysicalDeviceFilter::QueueRequirement;
		virtual core::vector<queue_req_t> getQueueRequirements() const
		{
			core::vector<queue_req_t> retval;
			
			using flags_t = video::IQueue::FAMILY_FLAGS;
			// The Graphics Queue should be able to do Compute and image transfers of any granularity (transfer only queue families can have problems with that)
			retval.push_back({.requiredFlags=flags_t::COMPUTE_BIT,.disallowedFlags=flags_t::NONE,.queueCount=1,.maxImageTransferGranularity={1,1,1}});

			return retval;
		}

		// These features are features you'll enable if present but won't interfere with your choice of device
		// There's no intersection operator (yet) on the features, so its not used yet!
		// virtual function so you can override as needed for some example father down the line
		virtual video::SPhysicalDeviceFeatures getPreferredDeviceFeatures() const
		{
			video::SPhysicalDeviceFeatures retval = {};

			/*retval.shaderFloat64 = true;
			retval.shaderDrawParameters = true;
			retval.drawIndirectCount = true;*/

			return retval;
		}

		// This will get called after all physical devices go through filtering via `InitParams::physicalDeviceFilter`
		virtual video::IPhysicalDevice* selectPhysicalDevice(const core::set<video::IPhysicalDevice*>& suitablePhysicalDevices)
		{
			using namespace nbl::video;

			using driver_id_enum = IPhysicalDevice::E_DRIVER_ID;
			// from least to most buggy
			const core::vector<driver_id_enum> preference = {
				driver_id_enum::EDI_NVIDIA_PROPRIETARY,
				driver_id_enum::EDI_INTEL_OPEN_SOURCE_MESA,
				driver_id_enum::EDI_MESA_RADV,
				driver_id_enum::EDI_AMD_OPEN_SOURCE,
				driver_id_enum::EDI_MOLTENVK,
				driver_id_enum::EDI_MESA_LLVMPIPE,
				driver_id_enum::EDI_INTEL_PROPRIETARY_WINDOWS,
				driver_id_enum::EDI_AMD_PROPRIETARY,
				driver_id_enum::EDI_GOOGLE_SWIFTSHADER
			};
			// @Hazardu you'll probably want to add an override from cmdline for GPU choice here
			for (auto driver_id : preference)
			for (auto device : suitablePhysicalDevices)
			if (device->getProperties().driverID==driver_id)
				return device;

			return nullptr;
		}

		// This will most certainly be overriden
		using queue_family_range_t = std::span<const video::IPhysicalDevice::SQueueFamilyProperties>;
		virtual std::array<video::ILogicalDevice::SQueueCreationParams,video::ILogicalDevice::MaxQueueFamilies> getQueueCreationParameters(const queue_family_range_t& familyProperties)
		{
			using namespace video;
			std::array<ILogicalDevice::SQueueCreationParams,ILogicalDevice::MaxQueueFamilies> retval = {};

			// since we requested a device that has such a capable queue family (unless `getQueueRequirements` got overriden) we're sure we'll get at least one family
			for (auto i=0u; i<familyProperties.size(); i++)
			if (familyProperties[i].queueFlags.hasFlags(getQueueRequirements().front().requiredFlags))
				retval[i].count = 1;

			return retval;
		}

		virtual video::IQueue* getQueue(video::IQueue::FAMILY_FLAGS flags) const
		{
			// In the default implementation of everything I asked only for one queue from first compute family
			const auto familyProperties = m_device->getPhysicalDevice()->getQueueFamilyProperties();
			for (auto i=0u; i<familyProperties.size(); i++)
			if (familyProperties[i].queueFlags.hasFlags(flags))
				return m_device->getQueue(i,0);

			return nullptr;
		}

		// virtual to allow aliasing and total flexibility
		virtual video::IQueue* getComputeQueue() const
		{
			return getQueue(video::IQueue::FAMILY_FLAGS::COMPUTE_BIT);
		}

		core::smart_refctd_ptr<video::CVulkanConnection> m_api;
		core::smart_refctd_ptr<video::ILogicalDevice> m_device;
		video::IPhysicalDevice* m_physicalDevice;
};

}

#endif // _CAMERA_IMPL_