#include "nbl/video/CVulkanPhysicalDevice.h"
#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{
	
std::unique_ptr<CVulkanPhysicalDevice> CVulkanPhysicalDevice::create(core::smart_refctd_ptr<system::ISystem>&& sys, IAPIConnection* const api, renderdoc_api_t* const rdoc, const VkPhysicalDevice vk_physicalDevice)
{
    system::logger_opt_ptr logger = api->getDebugCallback()->getLogger();

    IPhysicalDevice::SInitData initData = {std::move(sys),api};

    auto& properties = initData.properties;
    auto& features = initData.features;
    // First call just with Vulkan 1.0 API because:
    // "The value of apiVersion may be different than the version returned by vkEnumerateInstanceVersion; either higher or lower.
    //  In such cases, the application must not use functionality that exceeds the version of Vulkan associated with a given object.
    //  The pApiVersion parameter returned by vkEnumerateInstanceVersion is the version associated with a VkInstance and its children,
    //  except for a VkPhysicalDevice and its children. VkPhysicalDeviceProperties::apiVersion is the version associated with a VkPhysicalDevice and its children."
    VkPhysicalDeviceProperties vk_deviceProperties;
    {
        vkGetPhysicalDeviceProperties(vk_physicalDevice, &vk_deviceProperties);
        if (vk_deviceProperties.apiVersion < MinimumVulkanApiVersion)
        {
            logger.log(
                "Not enumerating VkPhysicalDevice %p because it does not support minimum required version %d, supports %d instead!",
                system::ILogger::ELL_INFO, vk_physicalDevice, MinimumVulkanApiVersion, vk_deviceProperties.apiVersion
            );
            return nullptr;
        }

        properties.apiVersion.major = VK_API_VERSION_MAJOR(vk_deviceProperties.apiVersion);
        properties.apiVersion.minor = VK_API_VERSION_MINOR(vk_deviceProperties.apiVersion);
        properties.apiVersion.subminor = 0;
        properties.apiVersion.patch = VK_API_VERSION_PATCH(vk_deviceProperties.apiVersion);

        properties.driverVersion = vk_deviceProperties.driverVersion;
        properties.vendorID = vk_deviceProperties.vendorID;
        properties.deviceID = vk_deviceProperties.deviceID;
        switch (vk_deviceProperties.deviceType)
        {
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
                properties.deviceType = E_TYPE::ET_INTEGRATED_GPU;
                break;
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                properties.deviceType = E_TYPE::ET_DISCRETE_GPU;
                break;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
                properties.deviceType = E_TYPE::ET_VIRTUAL_GPU;
                break;
            case VK_PHYSICAL_DEVICE_TYPE_CPU:
                properties.deviceType = E_TYPE::ET_CPU;
                break;
            default:
                properties.deviceType = E_TYPE::ET_UNKNOWN;
        }
        memcpy(properties.deviceName, vk_deviceProperties.deviceName, VK_MAX_PHYSICAL_DEVICE_NAME_SIZE);
        memcpy(properties.pipelineCacheUUID, vk_deviceProperties.pipelineCacheUUID, VK_UUID_SIZE);


        // grab all limits we expose
        properties.limits.maxImageDimension1D = vk_deviceProperties.limits.maxImageDimension1D;
        properties.limits.maxImageDimension2D = vk_deviceProperties.limits.maxImageDimension2D;
        properties.limits.maxImageDimension3D = vk_deviceProperties.limits.maxImageDimension3D;
        properties.limits.maxImageDimensionCube = vk_deviceProperties.limits.maxImageDimensionCube;
        properties.limits.maxImageArrayLayers = vk_deviceProperties.limits.maxImageArrayLayers;
        properties.limits.maxBufferViewTexels = vk_deviceProperties.limits.maxTexelBufferElements;
        properties.limits.maxUBOSize = vk_deviceProperties.limits.maxUniformBufferRange;
        properties.limits.maxSSBOSize = vk_deviceProperties.limits.maxStorageBufferRange;
        if (vk_deviceProperties.limits.maxPushConstantsSize > SLimits::MaxMaxPushConstantsSize)
        {
            logger.log(
                "Encountered VkPhysicalDevice %p which has higher Push Constant Size (%d) limit than we anticipated (%d), clamping the reported value!",
                system::ILogger::ELL_WARNING, vk_physicalDevice, vk_deviceProperties.limits.maxPushConstantsSize, SLimits::MaxMaxPushConstantsSize
            );
            properties.limits.maxPushConstantsSize = SLimits::MaxMaxPushConstantsSize;
        }
        else
            properties.limits.maxPushConstantsSize = vk_deviceProperties.limits.maxPushConstantsSize;
        properties.limits.maxMemoryAllocationCount = vk_deviceProperties.limits.maxMemoryAllocationCount;
        properties.limits.maxSamplerAllocationCount = vk_deviceProperties.limits.maxSamplerAllocationCount;
        properties.limits.bufferImageGranularity = vk_deviceProperties.limits.bufferImageGranularity;
        //vk_deviceProperties.limits.sparseAddressSpaceSize;
        // we hardcoded this in the engine to 4
        if (vk_deviceProperties.limits.maxBoundDescriptorSets<4u)
        {
            logger.log("Not enumerating VkPhysicalDevice %p because it reports limits below Vulkan specification requirements!", system::ILogger::ELL_INFO, vk_physicalDevice);
            return nullptr;
        }

        properties.limits.maxPerStageDescriptorSamplers = vk_deviceProperties.limits.maxPerStageDescriptorSamplers;
        properties.limits.maxPerStageDescriptorUBOs = vk_deviceProperties.limits.maxPerStageDescriptorUniformBuffers;
        properties.limits.maxPerStageDescriptorSSBOs = vk_deviceProperties.limits.maxPerStageDescriptorStorageBuffers;
        properties.limits.maxPerStageDescriptorImages = vk_deviceProperties.limits.maxPerStageDescriptorSampledImages;
        properties.limits.maxPerStageDescriptorStorageImages = vk_deviceProperties.limits.maxPerStageDescriptorStorageImages;
        properties.limits.maxPerStageDescriptorInputAttachments = vk_deviceProperties.limits.maxPerStageDescriptorInputAttachments;
        properties.limits.maxPerStageResources = vk_deviceProperties.limits.maxPerStageResources;
            
        // Max Descriptors (too lazy to check GPU info what the values for these are, they depend on the number of supported stages)
        // TODO: derive and implement checks according to https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#_required_limits
        properties.limits.maxDescriptorSetSamplers = vk_deviceProperties.limits.maxDescriptorSetSamplers;
        properties.limits.maxDescriptorSetUBOs = vk_deviceProperties.limits.maxDescriptorSetUniformBuffers;
        properties.limits.maxDescriptorSetDynamicOffsetUBOs = vk_deviceProperties.limits.maxDescriptorSetUniformBuffersDynamic;
        properties.limits.maxDescriptorSetSSBOs = vk_deviceProperties.limits.maxDescriptorSetStorageBuffers;
        properties.limits.maxDescriptorSetDynamicOffsetSSBOs = vk_deviceProperties.limits.maxDescriptorSetStorageBuffersDynamic;
        properties.limits.maxDescriptorSetImages = vk_deviceProperties.limits.maxDescriptorSetSampledImages;
        properties.limits.maxDescriptorSetStorageImages = vk_deviceProperties.limits.maxDescriptorSetStorageImages;
        properties.limits.maxDescriptorSetInputAttachments = vk_deviceProperties.limits.maxDescriptorSetInputAttachments;

        // we don't even store this as everyone does these limits and we've hardcoded that
        if (vk_deviceProperties.limits.maxVertexInputAttributes<16u || vk_deviceProperties.limits.maxVertexInputAttributeOffset<2047u)
        {
            logger.log("Not enumerating VkPhysicalDevice %p because it reports limits below Vulkan specification requirements!", system::ILogger::ELL_INFO, vk_physicalDevice);
            return nullptr;
        }
        if (vk_deviceProperties.limits.maxVertexInputBindings<16u || vk_deviceProperties.limits.maxVertexInputBindingStride<2048u)
        {
            logger.log("Not enumerating VkPhysicalDevice %p because it reports limits below Vulkan specification requirements!", system::ILogger::ELL_INFO, vk_physicalDevice);
            return nullptr;
        }
        properties.limits.maxVertexOutputComponents = vk_deviceProperties.limits.maxVertexOutputComponents;

        properties.limits.maxTessellationGenerationLevel = vk_deviceProperties.limits.maxTessellationGenerationLevel;
        properties.limits.maxTessellationPatchSize = vk_deviceProperties.limits.maxTessellationPatchSize;
        properties.limits.maxTessellationControlPerVertexInputComponents = vk_deviceProperties.limits.maxTessellationControlPerVertexInputComponents;
        properties.limits.maxTessellationControlPerVertexOutputComponents = vk_deviceProperties.limits.maxTessellationControlPerVertexOutputComponents;
        properties.limits.maxTessellationControlPerPatchOutputComponents = vk_deviceProperties.limits.maxTessellationControlPerPatchOutputComponents;
        properties.limits.maxTessellationControlTotalOutputComponents = vk_deviceProperties.limits.maxTessellationControlTotalOutputComponents;
        properties.limits.maxTessellationEvaluationInputComponents = vk_deviceProperties.limits.maxTessellationEvaluationInputComponents;
        properties.limits.maxTessellationEvaluationOutputComponents = vk_deviceProperties.limits.maxTessellationEvaluationOutputComponents;

        properties.limits.maxGeometryShaderInvocations = vk_deviceProperties.limits.maxGeometryShaderInvocations;
        properties.limits.maxGeometryInputComponents = vk_deviceProperties.limits.maxGeometryInputComponents;
        properties.limits.maxGeometryOutputComponents = vk_deviceProperties.limits.maxGeometryOutputComponents;
        properties.limits.maxGeometryOutputVertices = vk_deviceProperties.limits.maxGeometryOutputVertices;
        properties.limits.maxGeometryTotalOutputComponents = vk_deviceProperties.limits.maxGeometryTotalOutputComponents;
        
        properties.limits.maxFragmentInputComponents = vk_deviceProperties.limits.maxFragmentInputComponents;
        properties.limits.maxFragmentOutputAttachments = vk_deviceProperties.limits.maxFragmentOutputAttachments;
        properties.limits.maxFragmentDualSrcAttachments = vk_deviceProperties.limits.maxFragmentDualSrcAttachments;
        properties.limits.maxFragmentCombinedOutputResources = vk_deviceProperties.limits.maxFragmentCombinedOutputResources;

        properties.limits.maxComputeSharedMemorySize = vk_deviceProperties.limits.maxComputeSharedMemorySize;
        properties.limits.maxComputeWorkGroupCount[0] = vk_deviceProperties.limits.maxComputeWorkGroupCount[0];
        properties.limits.maxComputeWorkGroupCount[1] = vk_deviceProperties.limits.maxComputeWorkGroupCount[1];
        properties.limits.maxComputeWorkGroupCount[2] = vk_deviceProperties.limits.maxComputeWorkGroupCount[2];
        properties.limits.maxComputeWorkGroupInvocations = vk_deviceProperties.limits.maxComputeWorkGroupInvocations;
        properties.limits.maxWorkgroupSize[0] = vk_deviceProperties.limits.maxComputeWorkGroupSize[0];
        properties.limits.maxWorkgroupSize[1] = vk_deviceProperties.limits.maxComputeWorkGroupSize[1];
        properties.limits.maxWorkgroupSize[2] = vk_deviceProperties.limits.maxComputeWorkGroupSize[2];

        properties.limits.subPixelPrecisionBits = vk_deviceProperties.limits.subPixelPrecisionBits;
        properties.limits.subTexelPrecisionBits = vk_deviceProperties.limits.subTexelPrecisionBits;
        properties.limits.mipmapPrecisionBits = vk_deviceProperties.limits.mipmapPrecisionBits;

        if (vk_deviceProperties.limits.maxDrawIndexedIndexValue!=0xffFFffFFu)
        {
            logger.log("Not enumerating VkPhysicalDevice %p because it reports a limit below ROADMAP2022 spec which is ubiquitously supported already!", system::ILogger::ELL_INFO, vk_physicalDevice);
            return nullptr;
        }
        properties.limits.maxDrawIndirectCount = vk_deviceProperties.limits.maxDrawIndirectCount;
        
        properties.limits.maxSamplerLodBias = vk_deviceProperties.limits.maxSamplerLodBias;
        properties.limits.maxSamplerAnisotropyLog2 = static_cast<uint8_t>(std::log2(vk_deviceProperties.limits.maxSamplerAnisotropy));

        properties.limits.maxViewports = vk_deviceProperties.limits.maxViewports;
        properties.limits.maxViewportDims[0] = vk_deviceProperties.limits.maxViewportDimensions[0];
        properties.limits.maxViewportDims[1] = vk_deviceProperties.limits.maxViewportDimensions[1];
        properties.limits.viewportBoundsRange[0] = vk_deviceProperties.limits.viewportBoundsRange[0];
        properties.limits.viewportBoundsRange[1] = vk_deviceProperties.limits.viewportBoundsRange[1];
        properties.limits.viewportSubPixelBits = vk_deviceProperties.limits.viewportSubPixelBits;

        properties.limits.minMemoryMapAlignment = vk_deviceProperties.limits.minMemoryMapAlignment;
        properties.limits.bufferViewAlignment = vk_deviceProperties.limits.minTexelBufferOffsetAlignment;
        properties.limits.minUBOAlignment = vk_deviceProperties.limits.minUniformBufferOffsetAlignment;
        properties.limits.minSSBOAlignment = vk_deviceProperties.limits.minStorageBufferOffsetAlignment;

        properties.limits.minTexelOffset = vk_deviceProperties.limits.minTexelOffset;
        properties.limits.maxTexelOffset = vk_deviceProperties.limits.maxTexelOffset;
        properties.limits.minTexelGatherOffset = vk_deviceProperties.limits.minTexelGatherOffset;
        properties.limits.maxTexelGatherOffset = vk_deviceProperties.limits.maxTexelGatherOffset;
        properties.limits.minInterpolationOffset = vk_deviceProperties.limits.minInterpolationOffset;
        properties.limits.maxInterpolationOffset = vk_deviceProperties.limits.maxInterpolationOffset;
        properties.limits.subPixelInterpolationOffsetBits = vk_deviceProperties.limits.subPixelInterpolationOffsetBits;
        
        properties.limits.maxFramebufferWidth = vk_deviceProperties.limits.maxFramebufferWidth;
        properties.limits.maxFramebufferHeight = vk_deviceProperties.limits.maxFramebufferHeight;
        properties.limits.maxFramebufferLayers = vk_deviceProperties.limits.maxFramebufferLayers;
        
        // not checking framebuffer sample count limits, will check in format reporting instead (TODO)
        properties.limits.maxColorAttachments = vk_deviceProperties.limits.maxColorAttachments;

        // not checking sampled image sample count limits, will check in format reporting instead (TODO)
        properties.limits.maxSampleMaskWords = vk_deviceProperties.limits.maxSampleMaskWords;
        
        if (!vk_deviceProperties.limits.timestampComputeAndGraphics)
        {
            logger.log("Not enumerating VkPhysicalDevice %p because it reports a limit below ROADMAP2022 spec which is ubiquitously supported already!", system::ILogger::ELL_INFO, vk_physicalDevice);
            return nullptr;
        }
        properties.limits.timestampPeriodInNanoSeconds = vk_deviceProperties.limits.timestampPeriod;

        properties.limits.maxClipDistances = vk_deviceProperties.limits.maxClipDistances;
        properties.limits.maxCullDistances = vk_deviceProperties.limits.maxCullDistances;
        properties.limits.maxCombinedClipAndCullDistances = vk_deviceProperties.limits.maxCombinedClipAndCullDistances;
        
        properties.limits.discreteQueuePriorities = vk_deviceProperties.limits.discreteQueuePriorities;

        properties.limits.pointSizeRange[0] = vk_deviceProperties.limits.pointSizeRange[0];
        properties.limits.pointSizeRange[1] = vk_deviceProperties.limits.pointSizeRange[1];
        properties.limits.lineWidthRange[0] = vk_deviceProperties.limits.lineWidthRange[0];
        properties.limits.lineWidthRange[1] = vk_deviceProperties.limits.lineWidthRange[1];
        properties.limits.pointSizeGranularity = vk_deviceProperties.limits.pointSizeGranularity;
        properties.limits.lineWidthGranularity = vk_deviceProperties.limits.lineWidthGranularity;
        properties.limits.strictLines = vk_deviceProperties.limits.strictLines;

        properties.limits.standardSampleLocations = vk_deviceProperties.limits.standardSampleLocations;

        properties.limits.optimalBufferCopyOffsetAlignment = vk_deviceProperties.limits.optimalBufferCopyOffsetAlignment;
        properties.limits.optimalBufferCopyRowPitchAlignment = vk_deviceProperties.limits.optimalBufferCopyRowPitchAlignment;
        properties.limits.nonCoherentAtomSize = vk_deviceProperties.limits.nonCoherentAtomSize;

        /* TODO Verify
        constexpr uint32_t MaxRoadmap2022PerStageResources = 200u;
        const uint32_t MaxResources = core::min(MaxRoadmap2022PerStageResources,
            properties.limits.maxPerStageDescriptorUBOs+properties.limits.maxPerStageDescriptorSSBOs+
            properties.limits.maxPerStageDescriptorImages+properties.limits.maxPerStageDescriptorStorageImages+
            properties.limits.maxPerStageDescriptorInputAttachments+properties.limits.maxColorAttachments
        );
        if (vk_deviceProperties.limits.maxPerStageResources<MaxResources)
        {
            logger.log("Not enumerating VkPhysicalDevice %p because it reports limits below Vulkan specification requirements!", system::ILogger::ELL_INFO, vk_physicalDevice);
            return nullptr;
        }
        */
    }

    // Get Supported Extensions
    core::unordered_set<std::string> availableFeatureSet;
    {
        uint32_t count;
        VkResult res = vkEnumerateDeviceExtensionProperties(vk_physicalDevice,nullptr,&count,nullptr);
        assert(VK_SUCCESS==res);
        core::vector<VkExtensionProperties> vk_extensions(count);
        res = vkEnumerateDeviceExtensionProperties(vk_physicalDevice,nullptr,&count,vk_extensions.data());
        assert(VK_SUCCESS==res);

        for (const auto& vk_extension : vk_extensions)
            availableFeatureSet.insert(vk_extension.extensionName);
    }
    auto isExtensionSupported = [&availableFeatureSet](const char* name)->bool
    {
        return availableFeatureSet.find(name)!=availableFeatureSet.end();
    };
    //! Required by Nabla Core Profile
    if (!rdoc && !isExtensionSupported(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME))
        return nullptr;
    if (!isExtensionSupported(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME))
        return nullptr;
    if (!isExtensionSupported(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME))
        return nullptr;

    // Basic stuff for constructing pNext chains
    VkBaseInStructure* pNextTail;
    auto setPNextChainTail = [&pNextTail](void* structure) -> void {pNextTail = reinterpret_cast<VkBaseInStructure*>(structure);};
    auto addToPNextChain = [&pNextTail](void* structure) -> void
    {
        auto toAdd = reinterpret_cast<VkBaseInStructure*>(structure);
        pNextTail->pNext = toAdd;
        pNextTail = toAdd;
    };
    auto finalizePNextChain = [&pNextTail]() -> void
    {
        pNextTail->pNext = nullptr;
    };

    // Get physical device's limits/properties
    {
        VkPhysicalDeviceProperties2 deviceProperties2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
        setPNextChainTail(&deviceProperties2);
        // !! Our minimum supported Vulkan version is 1.1, no need to check anything before using `vulkan11Properties`
        VkPhysicalDeviceVulkan11Properties                      vulkan11Properties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES };
        addToPNextChain(&vulkan11Properties);
        //! Declare all the property structs before so they don't go out of scope
        //! Provided by Vk 1.2
        VkPhysicalDeviceVulkan12Properties                      vulkan12Properties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES };
        addToPNextChain(&vulkan12Properties);
        //! Provided by Vk 1.3
        VkPhysicalDeviceVulkan13Properties                      vulkan13Properties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES };
        addToPNextChain(&vulkan13Properties);
        //! Required by Nabla Core Profile
        VkPhysicalDeviceExternalMemoryHostPropertiesEXT         externalMemoryHostProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT };
        VkPhysicalDeviceRobustness2PropertiesEXT                robustness2Properties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_PROPERTIES_EXT };
        addToPNextChain(&robustness2Properties);
        //! Extensions (ordered by spec extension number)
        VkPhysicalDeviceConservativeRasterizationPropertiesEXT  conservativeRasterizationProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONSERVATIVE_RASTERIZATION_PROPERTIES_EXT };
        VkPhysicalDeviceDiscardRectanglePropertiesEXT           discardRectangleProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DISCARD_RECTANGLE_PROPERTIES_EXT };
        VkPhysicalDeviceSampleLocationsPropertiesEXT            sampleLocationsProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLE_LOCATIONS_PROPERTIES_EXT };
        VkPhysicalDeviceAccelerationStructurePropertiesKHR      accelerationStructureProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR };
        VkPhysicalDevicePCIBusInfoPropertiesEXT                 PCIBusInfoProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT };
        VkPhysicalDeviceFragmentDensityMapPropertiesEXT         fragmentDensityMapProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_PROPERTIES_EXT };
        VkPhysicalDeviceLineRasterizationPropertiesEXT          lineRasterizationProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_PROPERTIES_EXT };
        //VkPhysicalDeviceDeviceGeneratedCommandsPropertiesNV     deviceGeneratedCommandsPropertiesNV = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_PROPERTIES_NV };
        //VkPhysicalDeviceGraphicsPipelineLibraryPropertiesEXT    graphicsPipelineLibraryProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_PROPERTIES_EXT };
        VkPhysicalDeviceFragmentDensityMap2PropertiesEXT        fragmentDensityMap2Properties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_2_PROPERTIES_EXT };
        VkPhysicalDeviceRayTracingPipelinePropertiesKHR         rayTracingPipelineProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR };
#if 0 // TODO
        VkPhysicalDeviceCooperativeMatrixPropertiesKHR          cooperativeMatrixProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR };
#endif
        VkPhysicalDeviceShaderSMBuiltinsPropertiesNV            shaderSMBuiltinsPropertiesNV = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_PROPERTIES_NV };
        VkPhysicalDeviceShaderCoreProperties2AMD                shaderCoreProperties2AMD = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_2_AMD };
        //! Because Renderdoc is special and instead of ignoring extensions it whitelists them
        if (isExtensionSupported(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME))
            addToPNextChain(&externalMemoryHostProperties);
        //! This is only written for convenience to avoid getting validation errors otherwise vulkan will just skip any strutctures it doesn't recognize
        if (isExtensionSupported(VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME))
            addToPNextChain(&conservativeRasterizationProperties);
        if (isExtensionSupported(VK_EXT_DISCARD_RECTANGLES_EXTENSION_NAME))
            addToPNextChain(&discardRectangleProperties);
        if (isExtensionSupported(VK_EXT_SAMPLE_LOCATIONS_EXTENSION_NAME))
            addToPNextChain(&sampleLocationsProperties);
        if (isExtensionSupported(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME))
            addToPNextChain(&accelerationStructureProperties);
        if (isExtensionSupported(VK_EXT_PCI_BUS_INFO_EXTENSION_NAME))
            addToPNextChain(&PCIBusInfoProperties);
        if (isExtensionSupported(VK_EXT_FRAGMENT_DENSITY_MAP_EXTENSION_NAME))
            addToPNextChain(&fragmentDensityMapProperties);
        if (isExtensionSupported(VK_EXT_LINE_RASTERIZATION_EXTENSION_NAME))
            addToPNextChain(&lineRasterizationProperties);
        if (isExtensionSupported(VK_EXT_FRAGMENT_DENSITY_MAP_2_EXTENSION_NAME))
            addToPNextChain(&fragmentDensityMap2Properties);
        if (isExtensionSupported(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME))
            addToPNextChain(&rayTracingPipelineProperties);
#if 0 // TODO
        if (isExtensionSupported(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME))
            addToPNextChain(&cooperativeMatrixProperties);
#endif
        if (isExtensionSupported(VK_NV_SHADER_SM_BUILTINS_EXTENSION_NAME))
            addToPNextChain(&shaderSMBuiltinsPropertiesNV);
        if (isExtensionSupported(VK_AMD_SHADER_CORE_PROPERTIES_2_EXTENSION_NAME))
            addToPNextChain(&shaderCoreProperties2AMD);
        // call
        finalizePNextChain();
        vkGetPhysicalDeviceProperties2(vk_physicalDevice,&deviceProperties2);

        /* Vulkan 1.1 Core  */
        memcpy(properties.deviceUUID, vulkan11Properties.deviceUUID, VK_UUID_SIZE);
        memcpy(properties.driverUUID, vulkan11Properties.driverUUID, VK_UUID_SIZE);
        memcpy(properties.deviceLUID, vulkan11Properties.deviceLUID, VK_LUID_SIZE);
        properties.deviceNodeMask = vulkan11Properties.deviceNodeMask;
        properties.deviceLUIDValid = vulkan11Properties.deviceLUIDValid;

        properties.limits.subgroupSize = vulkan11Properties.subgroupSize;
        properties.limits.subgroupOpsShaderStages = static_cast<IGPUShader::E_SHADER_STAGE>(vulkan11Properties.subgroupSupportedStages);
        // ROADMAP 2022 would also like ARITHMETIC and QUAD
        constexpr uint32_t NablaSubgroupOperationMask = VK_SUBGROUP_FEATURE_BASIC_BIT|VK_SUBGROUP_FEATURE_VOTE_BIT|VK_SUBGROUP_FEATURE_BALLOT_BIT|VK_SUBGROUP_FEATURE_SHUFFLE_BIT|VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT;
        if ((vulkan11Properties.subgroupSupportedOperations&NablaSubgroupOperationMask)!=NablaSubgroupOperationMask)
        {
            logger.log(
                "Not enumerating VkPhysicalDevice %p because its supported subgroup op bitmask is %d and we require at least %d!",
                system::ILogger::ELL_INFO, vk_physicalDevice, vulkan11Properties.subgroupSupportedOperations, NablaSubgroupOperationMask
            );
            return nullptr;
        }
        properties.limits.shaderSubgroupArithmetic = vulkan11Properties.subgroupSupportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
        properties.limits.shaderSubgroupClustered = vulkan11Properties.subgroupSupportedOperations & VK_SUBGROUP_FEATURE_CLUSTERED_BIT;
        properties.limits.shaderSubgroupQuad = vulkan11Properties.subgroupSupportedOperations & VK_SUBGROUP_FEATURE_QUAD_BIT;
        properties.limits.shaderSubgroupQuadAllStages = vulkan11Properties.subgroupQuadOperationsInAllStages;

        properties.limits.pointClippingBehavior = static_cast<SLimits::E_POINT_CLIPPING_BEHAVIOR>(vulkan11Properties.pointClippingBehavior);

        properties.limits.maxMultiviewViewCount = vulkan11Properties.maxMultiviewViewCount;
        properties.limits.maxMultiviewInstanceIndex = vulkan11Properties.maxMultiviewInstanceIndex;

        //vulkan11Properties.protectedNoFault;

        properties.limits.maxPerSetDescriptors = vulkan11Properties.maxPerSetDescriptors;
        properties.limits.maxMemoryAllocationSize = vulkan11Properties.maxMemoryAllocationSize;


        /* Vulkan 1.2 Core  */
        properties.driverID = getDriverIdFromVkDriverId(vulkan12Properties.driverID);
        memcpy(properties.driverName, vulkan12Properties.driverName, VK_MAX_DRIVER_NAME_SIZE);
        memcpy(properties.driverInfo, vulkan12Properties.driverInfo, VK_MAX_DRIVER_INFO_SIZE);
        properties.conformanceVersion.major = vulkan12Properties.conformanceVersion.major;
        properties.conformanceVersion.minor = vulkan12Properties.conformanceVersion.minor;
        properties.conformanceVersion.subminor = vulkan12Properties.conformanceVersion.subminor;
        properties.conformanceVersion.patch = vulkan12Properties.conformanceVersion.patch;
            
        // Helper bools :D
        const bool isIntelGPU = (properties.driverID == E_DRIVER_ID::EDI_INTEL_OPEN_SOURCE_MESA || properties.driverID == E_DRIVER_ID::EDI_INTEL_PROPRIETARY_WINDOWS);
        const bool isAMDGPU = (properties.driverID == E_DRIVER_ID::EDI_AMD_OPEN_SOURCE || properties.driverID == E_DRIVER_ID::EDI_AMD_PROPRIETARY);
        const bool isNVIDIAGPU = (properties.driverID == E_DRIVER_ID::EDI_NVIDIA_PROPRIETARY);

        //vulkan12Properties.denormBehaviorIndependence;
        //vulkan12Properties.denormBehaviorIndependence;
        if (!vulkan12Properties.shaderSignedZeroInfNanPreserveFloat16)
            return nullptr;
        if (!vulkan12Properties.shaderSignedZeroInfNanPreserveFloat32)
            return nullptr;
        properties.limits.shaderSignedZeroInfNanPreserveFloat64 = vulkan12Properties.shaderSignedZeroInfNanPreserveFloat64;
        properties.limits.shaderDenormPreserveFloat16 = vulkan12Properties.shaderDenormPreserveFloat16;
        properties.limits.shaderDenormPreserveFloat32 = vulkan12Properties.shaderDenormPreserveFloat32;
        properties.limits.shaderDenormPreserveFloat64 = vulkan12Properties.shaderDenormPreserveFloat64;
        properties.limits.shaderDenormFlushToZeroFloat16 = vulkan12Properties.shaderDenormFlushToZeroFloat16;
        properties.limits.shaderDenormFlushToZeroFloat32 = vulkan12Properties.shaderDenormFlushToZeroFloat32;
        properties.limits.shaderDenormFlushToZeroFloat64 = vulkan12Properties.shaderDenormFlushToZeroFloat64;
        properties.limits.shaderRoundingModeRTEFloat16 = vulkan12Properties.shaderRoundingModeRTEFloat16;
        properties.limits.shaderRoundingModeRTEFloat32 = vulkan12Properties.shaderRoundingModeRTEFloat32;
        properties.limits.shaderRoundingModeRTEFloat64 = vulkan12Properties.shaderRoundingModeRTEFloat64;
        properties.limits.shaderRoundingModeRTZFloat16 = vulkan12Properties.shaderRoundingModeRTZFloat16;
        properties.limits.shaderRoundingModeRTZFloat32 = vulkan12Properties.shaderRoundingModeRTZFloat32;
        properties.limits.shaderRoundingModeRTZFloat64 = vulkan12Properties.shaderRoundingModeRTZFloat64;
            
        // descriptor indexing
        properties.limits.maxUpdateAfterBindDescriptorsInAllPools                 = vulkan12Properties.maxUpdateAfterBindDescriptorsInAllPools;
        properties.limits.shaderUniformBufferArrayNonUniformIndexingNative        = vulkan12Properties.shaderUniformBufferArrayNonUniformIndexingNative;
        properties.limits.shaderSampledImageArrayNonUniformIndexingNative         = vulkan12Properties.shaderSampledImageArrayNonUniformIndexingNative;
        properties.limits.shaderStorageBufferArrayNonUniformIndexingNative        = vulkan12Properties.shaderStorageBufferArrayNonUniformIndexingNative;
        properties.limits.shaderStorageImageArrayNonUniformIndexingNative         = vulkan12Properties.shaderStorageImageArrayNonUniformIndexingNative;
        properties.limits.shaderInputAttachmentArrayNonUniformIndexingNative      = vulkan12Properties.shaderInputAttachmentArrayNonUniformIndexingNative;
        properties.limits.robustBufferAccessUpdateAfterBind                       = vulkan12Properties.robustBufferAccessUpdateAfterBind;
        properties.limits.quadDivergentImplicitLod                                = vulkan12Properties.quadDivergentImplicitLod;
        properties.limits.maxPerStageDescriptorUpdateAfterBindSamplers            = vulkan12Properties.maxPerStageDescriptorUpdateAfterBindSamplers;
        properties.limits.maxPerStageDescriptorUpdateAfterBindUBOs                = vulkan12Properties.maxPerStageDescriptorUpdateAfterBindUniformBuffers;
        properties.limits.maxPerStageDescriptorUpdateAfterBindSSBOs               = vulkan12Properties.maxPerStageDescriptorUpdateAfterBindStorageBuffers;
        properties.limits.maxPerStageDescriptorUpdateAfterBindImages              = vulkan12Properties.maxPerStageDescriptorUpdateAfterBindSampledImages;
        properties.limits.maxPerStageDescriptorUpdateAfterBindStorageImages       = vulkan12Properties.maxPerStageDescriptorUpdateAfterBindStorageImages;
        properties.limits.maxPerStageDescriptorUpdateAfterBindInputAttachments    = vulkan12Properties.maxPerStageDescriptorUpdateAfterBindInputAttachments;
        properties.limits.maxPerStageUpdateAfterBindResources                     = vulkan12Properties.maxPerStageUpdateAfterBindResources;
        properties.limits.maxDescriptorSetUpdateAfterBindSamplers                 = vulkan12Properties.maxDescriptorSetUpdateAfterBindSamplers;
        properties.limits.maxDescriptorSetUpdateAfterBindUBOs                     = vulkan12Properties.maxDescriptorSetUpdateAfterBindUniformBuffers;
        properties.limits.maxDescriptorSetUpdateAfterBindDynamicOffsetUBOs        = vulkan12Properties.maxDescriptorSetUpdateAfterBindUniformBuffersDynamic;
        properties.limits.maxDescriptorSetUpdateAfterBindSSBOs                    = vulkan12Properties.maxDescriptorSetUpdateAfterBindStorageBuffers;
        properties.limits.maxDescriptorSetUpdateAfterBindDynamicOffsetSSBOs       = vulkan12Properties.maxDescriptorSetUpdateAfterBindStorageBuffersDynamic;
        properties.limits.maxDescriptorSetUpdateAfterBindImages                   = vulkan12Properties.maxDescriptorSetUpdateAfterBindSampledImages;
        properties.limits.maxDescriptorSetUpdateAfterBindStorageImages            = vulkan12Properties.maxDescriptorSetUpdateAfterBindStorageImages;
        properties.limits.maxDescriptorSetUpdateAfterBindInputAttachments         = vulkan12Properties.maxDescriptorSetUpdateAfterBindInputAttachments;

        properties.limits.supportedDepthResolveModes = static_cast<SPhysicalDeviceLimits::RESOLVE_MODE_FLAGS>(vulkan12Properties.supportedDepthResolveModes);
        properties.limits.supportedStencilResolveModes = static_cast<SPhysicalDeviceLimits::RESOLVE_MODE_FLAGS>(vulkan12Properties.supportedStencilResolveModes);
        properties.limits.independentResolveNone = vulkan12Properties.independentResolveNone;
        properties.limits.independentResolve = vulkan12Properties.independentResolve;

        // not dealing with vulkan12Properties.filterMinmaxSingleComponentFormats, TODO report in usage
        properties.limits.filterMinmaxImageComponentMapping = vulkan12Properties.filterMinmaxImageComponentMapping;

        constexpr uint64_t ROADMAP2022TimelineSemahoreValueDifference = (0x1ull<<31u)-1ull;
        if (vulkan12Properties.maxTimelineSemaphoreValueDifference<ROADMAP2022TimelineSemahoreValueDifference)
            return nullptr;

        // don't deal with vulkan12PRoperties.framebufferIntegerColorSampleCounts, TODO report in usage


        /* Vulkan 1.3 Core  */
        properties.limits.minSubgroupSize = vulkan13Properties.minSubgroupSize;
        properties.limits.maxSubgroupSize = vulkan13Properties.maxSubgroupSize;
        properties.limits.maxComputeWorkgroupSubgroups = vulkan13Properties.maxComputeWorkgroupSubgroups;
        properties.limits.requiredSubgroupSizeStages = static_cast<asset::IShader::E_SHADER_STAGE>(vulkan13Properties.requiredSubgroupSizeStages&VK_SHADER_STAGE_ALL);

        // don't real with inline uniform blocks yet

        properties.limits.integerDotProduct8BitUnsignedAccelerated = vulkan13Properties.integerDotProduct8BitUnsignedAccelerated;
        properties.limits.integerDotProduct8BitSignedAccelerated = vulkan13Properties.integerDotProduct8BitSignedAccelerated;
        properties.limits.integerDotProduct8BitMixedSignednessAccelerated = vulkan13Properties.integerDotProduct8BitMixedSignednessAccelerated;
        properties.limits.integerDotProduct4x8BitPackedUnsignedAccelerated = vulkan13Properties.integerDotProduct4x8BitPackedUnsignedAccelerated;
        properties.limits.integerDotProduct4x8BitPackedSignedAccelerated = vulkan13Properties.integerDotProduct4x8BitPackedSignedAccelerated;
        properties.limits.integerDotProduct4x8BitPackedMixedSignednessAccelerated = vulkan13Properties.integerDotProduct4x8BitPackedMixedSignednessAccelerated;
        properties.limits.integerDotProduct16BitUnsignedAccelerated = vulkan13Properties.integerDotProduct16BitUnsignedAccelerated;
        properties.limits.integerDotProduct16BitSignedAccelerated = vulkan13Properties.integerDotProduct16BitSignedAccelerated;
        properties.limits.integerDotProduct16BitMixedSignednessAccelerated = vulkan13Properties.integerDotProduct16BitMixedSignednessAccelerated;
        properties.limits.integerDotProduct32BitUnsignedAccelerated = vulkan13Properties.integerDotProduct32BitUnsignedAccelerated;
        properties.limits.integerDotProduct32BitSignedAccelerated = vulkan13Properties.integerDotProduct32BitSignedAccelerated;
        properties.limits.integerDotProduct32BitMixedSignednessAccelerated = vulkan13Properties.integerDotProduct32BitMixedSignednessAccelerated;
        properties.limits.integerDotProduct64BitUnsignedAccelerated = vulkan13Properties.integerDotProduct64BitUnsignedAccelerated;
        properties.limits.integerDotProduct64BitSignedAccelerated = vulkan13Properties.integerDotProduct64BitSignedAccelerated;
        properties.limits.integerDotProduct64BitMixedSignednessAccelerated = vulkan13Properties.integerDotProduct64BitMixedSignednessAccelerated;
        properties.limits.integerDotProductAccumulatingSaturating8BitUnsignedAccelerated = vulkan13Properties.integerDotProductAccumulatingSaturating8BitUnsignedAccelerated;
        properties.limits.integerDotProductAccumulatingSaturating8BitSignedAccelerated = vulkan13Properties.integerDotProductAccumulatingSaturating8BitSignedAccelerated;
        properties.limits.integerDotProductAccumulatingSaturating8BitMixedSignednessAccelerated = vulkan13Properties.integerDotProductAccumulatingSaturating8BitMixedSignednessAccelerated;
        properties.limits.integerDotProductAccumulatingSaturating4x8BitPackedUnsignedAccelerated = vulkan13Properties.integerDotProductAccumulatingSaturating4x8BitPackedUnsignedAccelerated;
        properties.limits.integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated = vulkan13Properties.integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated;
        properties.limits.integerDotProductAccumulatingSaturating4x8BitPackedMixedSignednessAccelerated = vulkan13Properties.integerDotProductAccumulatingSaturating4x8BitPackedMixedSignednessAccelerated;
        properties.limits.integerDotProductAccumulatingSaturating16BitUnsignedAccelerated = vulkan13Properties.integerDotProductAccumulatingSaturating16BitUnsignedAccelerated;
        properties.limits.integerDotProductAccumulatingSaturating16BitSignedAccelerated = vulkan13Properties.integerDotProductAccumulatingSaturating16BitSignedAccelerated;
        properties.limits.integerDotProductAccumulatingSaturating16BitMixedSignednessAccelerated = vulkan13Properties.integerDotProductAccumulatingSaturating16BitMixedSignednessAccelerated;
        properties.limits.integerDotProductAccumulatingSaturating32BitUnsignedAccelerated = vulkan13Properties.integerDotProductAccumulatingSaturating32BitUnsignedAccelerated;
        properties.limits.integerDotProductAccumulatingSaturating32BitSignedAccelerated = vulkan13Properties.integerDotProductAccumulatingSaturating32BitSignedAccelerated;
        properties.limits.integerDotProductAccumulatingSaturating32BitMixedSignednessAccelerated = vulkan13Properties.integerDotProductAccumulatingSaturating32BitMixedSignednessAccelerated;
        properties.limits.integerDotProductAccumulatingSaturating64BitUnsignedAccelerated = vulkan13Properties.integerDotProductAccumulatingSaturating64BitUnsignedAccelerated;
        properties.limits.integerDotProductAccumulatingSaturating64BitSignedAccelerated = vulkan13Properties.integerDotProductAccumulatingSaturating64BitSignedAccelerated;
        properties.limits.integerDotProductAccumulatingSaturating64BitMixedSignednessAccelerated = vulkan13Properties.integerDotProductAccumulatingSaturating64BitMixedSignednessAccelerated;

        properties.limits.storageTexelBufferOffsetAlignmentBytes = vulkan13Properties.storageTexelBufferOffsetAlignmentBytes;
        // vulkan13Properties.storageTexelBufferOffsetSingleTexelAlignment;
        properties.limits.uniformTexelBufferOffsetAlignmentBytes = vulkan13Properties.uniformTexelBufferOffsetAlignmentBytes;
        // vulkan13Properties.uniformTexelBufferOffsetSingleTexelAlignment;

        properties.limits.maxBufferSize = vulkan13Properties.maxBufferSize;

            
        //! Nabla Core Extensions
        if (isExtensionSupported(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME)) // renderdoc special
            properties.limits.minImportedHostPointerAlignment = externalMemoryHostProperties.minImportedHostPointerAlignment;

        // there's no ShaderAtomicFloatPropertiesEXT 

        properties.limits.robustStorageBufferAccessSizeAlignment = robustness2Properties.robustStorageBufferAccessSizeAlignment;
        properties.limits.robustUniformBufferAccessSizeAlignment = robustness2Properties.robustUniformBufferAccessSizeAlignment;


        //! Extensions
        properties.limits.shaderTrinaryMinmax = isExtensionSupported(VK_AMD_SHADER_TRINARY_MINMAX_EXTENSION_NAME);

        properties.limits.shaderExplicitVertexParameter = isExtensionSupported(VK_AMD_SHADER_EXPLICIT_VERTEX_PARAMETER_EXTENSION_NAME);

        properties.limits.gpuShaderHalfFloatAMD = isExtensionSupported(VK_AMD_GPU_SHADER_HALF_FLOAT_EXTENSION_NAME);

        properties.limits.shaderImageLoadStoreLod = isExtensionSupported(VK_AMD_SHADER_IMAGE_LOAD_STORE_LOD_EXTENSION_NAME);

        properties.limits.displayTiming = isExtensionSupported(VK_GOOGLE_DISPLAY_TIMING_EXTENSION_NAME);

        if (isExtensionSupported(VK_EXT_DISCARD_RECTANGLES_EXTENSION_NAME))
            properties.limits.maxDiscardRectangles = discardRectangleProperties.maxDiscardRectangles;

        if (isExtensionSupported(VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME))
        {
            properties.limits.primitiveOverestimationSize = conservativeRasterizationProperties.primitiveOverestimationSize;
            properties.limits.maxExtraPrimitiveOverestimationSize = conservativeRasterizationProperties.maxExtraPrimitiveOverestimationSize;
            properties.limits.extraPrimitiveOverestimationSizeGranularity = conservativeRasterizationProperties.extraPrimitiveOverestimationSizeGranularity;
            properties.limits.primitiveUnderestimation = conservativeRasterizationProperties.primitiveUnderestimation;
            properties.limits.conservativePointAndLineRasterization = conservativeRasterizationProperties.conservativePointAndLineRasterization;
            properties.limits.degenerateTrianglesRasterized = conservativeRasterizationProperties.degenerateTrianglesRasterized;
            properties.limits.degenerateLinesRasterized = conservativeRasterizationProperties.degenerateLinesRasterized;
            properties.limits.fullyCoveredFragmentShaderInputVariable = conservativeRasterizationProperties.fullyCoveredFragmentShaderInputVariable;
            properties.limits.conservativeRasterizationPostDepthCoverage = conservativeRasterizationProperties.conservativeRasterizationPostDepthCoverage;
        }

        properties.limits.queueFamilyForeign = isExtensionSupported(VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME);

        properties.limits.shaderStencilExport = isExtensionSupported(VK_EXT_SHADER_STENCIL_EXPORT_EXTENSION_NAME);

        if (isExtensionSupported(VK_EXT_SAMPLE_LOCATIONS_EXTENSION_NAME))
        {
            properties.limits.variableSampleLocations = sampleLocationsProperties.variableSampleLocations;
            properties.limits.sampleLocationSubPixelBits = sampleLocationsProperties.sampleLocationSubPixelBits;
            properties.limits.sampleLocationSampleCounts = static_cast<asset::IImage::E_SAMPLE_COUNT_FLAGS>(sampleLocationsProperties.sampleLocationSampleCounts);
            properties.limits.maxSampleLocationGridSize = {sampleLocationsProperties.maxSampleLocationGridSize.width,sampleLocationsProperties.maxSampleLocationGridSize.height};
            properties.limits.sampleLocationCoordinateRange[0] = sampleLocationsProperties.sampleLocationCoordinateRange[0];
            properties.limits.sampleLocationCoordinateRange[1] = sampleLocationsProperties.sampleLocationCoordinateRange[1];
        }

        if (isExtensionSupported(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME))
        {
            properties.limits.maxAccelerationStructureGeometryCount = accelerationStructureProperties.maxGeometryCount;
            properties.limits.maxAccelerationStructureInstanceCount = accelerationStructureProperties.maxInstanceCount;
            properties.limits.maxAccelerationStructurePrimitiveCount = accelerationStructureProperties.maxPrimitiveCount;
            properties.limits.maxPerStageDescriptorAccelerationStructures = accelerationStructureProperties.maxPerStageDescriptorAccelerationStructures;
            properties.limits.maxPerStageDescriptorUpdateAfterBindAccelerationStructures = accelerationStructureProperties.maxPerStageDescriptorUpdateAfterBindAccelerationStructures;
            properties.limits.maxDescriptorSetAccelerationStructures = accelerationStructureProperties.maxDescriptorSetAccelerationStructures;
            properties.limits.maxDescriptorSetUpdateAfterBindAccelerationStructures = accelerationStructureProperties.maxDescriptorSetUpdateAfterBindAccelerationStructures;
            properties.limits.minAccelerationStructureScratchOffsetAlignment = accelerationStructureProperties.minAccelerationStructureScratchOffsetAlignment;
        }

        properties.limits.postDepthCoverage = isExtensionSupported(VK_EXT_POST_DEPTH_COVERAGE_EXTENSION_NAME);

        if (isExtensionSupported(VK_EXT_PCI_BUS_INFO_EXTENSION_NAME))
        {
            properties.limits.pciDomain = PCIBusInfoProperties.pciDomain;
            properties.limits.pciBus = PCIBusInfoProperties.pciBus;
            properties.limits.pciDevice = PCIBusInfoProperties.pciDevice;
            properties.limits.pciFunction = PCIBusInfoProperties.pciFunction;
        }

        if (isExtensionSupported(VK_EXT_FRAGMENT_DENSITY_MAP_EXTENSION_NAME))
        {
            properties.limits.minFragmentDensityTexelSize = {fragmentDensityMapProperties.minFragmentDensityTexelSize.width,fragmentDensityMapProperties.minFragmentDensityTexelSize.height};
            properties.limits.maxFragmentDensityTexelSize = {fragmentDensityMapProperties.maxFragmentDensityTexelSize.width,fragmentDensityMapProperties.maxFragmentDensityTexelSize.height};
            properties.limits.fragmentDensityInvocations = fragmentDensityMapProperties.fragmentDensityInvocations;
        }

        properties.limits.decorateString = isExtensionSupported(VK_GOOGLE_DECORATE_STRING_EXTENSION_NAME);

        if (isExtensionSupported(VK_EXT_LINE_RASTERIZATION_EXTENSION_NAME))
            properties.limits.lineSubPixelPrecisionBits = lineRasterizationProperties.lineSubPixelPrecisionBits;

        properties.limits.shaderNonSemanticInfo = isExtensionSupported(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);

        if (isExtensionSupported(VK_EXT_FRAGMENT_DENSITY_MAP_2_EXTENSION_NAME))
        {
            properties.limits.subsampledLoads = fragmentDensityMap2Properties.subsampledLoads;
            properties.limits.subsampledCoarseReconstructionEarlyAccess = fragmentDensityMap2Properties.subsampledCoarseReconstructionEarlyAccess;
            properties.limits.maxSubsampledArrayLayers = fragmentDensityMap2Properties.maxSubsampledArrayLayers;
            properties.limits.maxDescriptorSetSubsampledSamplers = fragmentDensityMap2Properties.maxDescriptorSetSubsampledSamplers;
        }

        if (isExtensionSupported(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME))
        {
            if (rayTracingPipelineProperties.shaderGroupHandleSize!=32)
            {
                logger.log("Not enumerating VkPhysicalDevice %p because it reports limits of exact-type contrary to Vulkan specification!", system::ILogger::ELL_INFO, vk_physicalDevice);
                return nullptr;
            }
            properties.limits.maxRayRecursionDepth = rayTracingPipelineProperties.maxRayRecursionDepth;
            properties.limits.maxShaderGroupStride = rayTracingPipelineProperties.maxShaderGroupStride;
            properties.limits.shaderGroupBaseAlignment = rayTracingPipelineProperties.shaderGroupBaseAlignment;
            properties.limits.maxRayDispatchInvocationCount = rayTracingPipelineProperties.maxRayDispatchInvocationCount;
            properties.limits.shaderGroupHandleAlignment = rayTracingPipelineProperties.shaderGroupHandleAlignment;
            properties.limits.maxRayHitAttributeSize = rayTracingPipelineProperties.maxRayHitAttributeSize;
        }
#if 0 //TODO
        if (isExtensionSupported(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME))
            properties.limits.cooperativeMatrixSupportedStages = static_cast<asset::IShader::E_SHADER_STAGE>(cooperativeMatrixProperties.cooperativeMatrixSupportedStages);
#endif


        //! Nabla
        if (isExtensionSupported(VK_NV_SHADER_SM_BUILTINS_EXTENSION_NAME))
            properties.limits.computeUnits = shaderSMBuiltinsPropertiesNV.shaderSMCount;
        else if(isExtensionSupported(VK_AMD_SHADER_CORE_PROPERTIES_2_EXTENSION_NAME))
            properties.limits.computeUnits = shaderCoreProperties2AMD.activeComputeUnitCount;
        //else if (isExtensionSupported(VK_ARM_..._EXTENSION_NAME)) TODO implement the ARM equivalent
        else 
            properties.limits.computeUnits = getMaxComputeUnitsFromDriverID(properties.driverID);
            
        properties.limits.dispatchBase = true;
        properties.limits.allowCommandBufferQueryCopies = true; // TODO: REDO WE NOW SUPPORT PERF QUERIES always true in vk for all query types instead of PerformanceQuery which we don't support at the moment (have VkPhysicalDevicePerformanceQueryPropertiesKHR::allowCommandBufferQueryCopies in mind)
        properties.limits.maxOptimallyResidentWorkgroupInvocations = core::min(core::roundDownToPoT(properties.limits.maxComputeWorkGroupInvocations),512u);
            
        auto invocationsPerComputeUnit = getMaxInvocationsPerComputeUnitsFromDriverID(properties.driverID);
        if(isExtensionSupported(VK_NV_SHADER_SM_BUILTINS_EXTENSION_NAME))
        {
            constexpr auto invocationsPerWarp = 32u; // unless Nvidia changed something recently
            invocationsPerComputeUnit = shaderSMBuiltinsPropertiesNV.shaderWarpsPerSM*invocationsPerWarp;
        }
        properties.limits.maxResidentInvocations = properties.limits.computeUnits*invocationsPerComputeUnit;

        /*
            [NO NABLA SUPPORT] Vulkan 1.0 implementation must support the 1.0 version of SPIR-V and the 1.0 version of the SPIR-V Extended Instructions for GLSL. If the VK_KHR_spirv_1_4 extension is enabled, the implementation must additionally support the 1.4 version of SPIR-V.
            [NO NABLA SUPPORT] A Vulkan 1.1 implementation must support the 1.0, 1.1, 1.2, and 1.3 versions of SPIR-V and the 1.0 version of the SPIR-V Extended Instructions for GLSL.
            [NO NABLA SUPPORT] A Vulkan 1.2 implementation must support the 1.0, 1.1, 1.2, 1.3, 1.4, and 1.5 versions of SPIR-V and the 1.0 version of the SPIR-V Extended Instructions for GLSL.
            A Vulkan 1.3 implementation must support the 1.0, 1.1, 1.2, 1.3, 1.4, and 1.5 versions of SPIR-V and the 1.0 version of the SPIR-V Extended Instructions for GLSL.
        */
        properties.limits.spirvVersion = asset::IShaderCompiler::E_SPIRV_VERSION::ESV_1_6;
    }

        
    // ! In Vulkan: These will be reported based on availability of an extension and will be enabled by enabling an extension
    // Table 51. Extension Feature Aliases (vkspec 1.3.211)
    // Extension                               Feature(s)
    // VK_KHR_shader_draw_parameters           shaderDrawParameters
    // VK_KHR_draw_indirect_count              drawIndirectCount
    // VK_KHR_sampler_mirror_clamp_to_edge     samplerMirrorClampToEdge
    // VK_EXT_descriptor_indexing              descriptorIndexing
    // VK_EXT_sampler_filter_minmax            samplerFilterMinmax
    // VK_EXT_shader_viewport_index_layer      shaderOutputViewportIndex, shaderOutputLayer
    // but we enable them all from Vulkan1XFeatures anyway!

    // Get Device Features
    {
        VkPhysicalDeviceFeatures2 deviceFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
        setPNextChainTail(&deviceFeatures);
        // !! Our minimum supported Vulkan version is 1.1, no need to check anything before using `vulkan11Features`
        VkPhysicalDeviceVulkan11Features                                vulkan11Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
        addToPNextChain(&vulkan11Features);
        //! Declare all the property structs before so they don't go out of scope
        //! Provided by Vk 1.2
        VkPhysicalDeviceVulkan12Features                                vulkan12Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
        addToPNextChain(&vulkan12Features);
        //! Provided by Vk 1.3
        VkPhysicalDeviceVulkan13Features                                vulkan13Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
        addToPNextChain(&vulkan13Features);
        //! Nabla Core Profile Features
        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT                    shaderAtomicFloatFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT };
        addToPNextChain(&shaderAtomicFloatFeatures);
        VkPhysicalDeviceRobustness2FeaturesEXT                          robustness2Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT };
        addToPNextChain(&robustness2Features);
        //! Extensions (ordered by spec extension number)
        VkPhysicalDeviceConditionalRenderingFeaturesEXT                 conditionalRenderingFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT };
        VkPhysicalDevicePerformanceQueryFeaturesKHR                     performanceQueryFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PERFORMANCE_QUERY_FEATURES_KHR };
        VkPhysicalDeviceAccelerationStructureFeaturesKHR                accelerationStructureFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR };
        VkPhysicalDeviceRayTracingPipelineFeaturesKHR                   rayTracingPipelineFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR };
        VkPhysicalDeviceShaderSMBuiltinsFeaturesNV                      shaderSMBuiltinsFeaturesNV = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_FEATURES_NV };
        VkPhysicalDeviceRepresentativeFragmentTestFeaturesNV            representativeFragmentTestFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_REPRESENTATIVE_FRAGMENT_TEST_FEATURES_NV };
        VkPhysicalDeviceShaderClockFeaturesKHR                          shaderClockFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR };
        VkPhysicalDeviceComputeShaderDerivativesFeaturesNV              computeShaderDerivativesFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_NV };
        VkPhysicalDeviceShaderImageFootprintFeaturesNV                  shaderImageFootprintFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_FOOTPRINT_FEATURES_NV };
        VkPhysicalDeviceShaderIntegerFunctions2FeaturesINTEL            intelShaderIntegerFunctions2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_FUNCTIONS_2_FEATURES_INTEL };
        VkPhysicalDeviceFragmentDensityMapFeaturesEXT                   fragmentDensityMapFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_FEATURES_EXT };
        VkPhysicalDeviceFragmentDensityMap2FeaturesEXT                  fragmentDensityMap2Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_2_FEATURES_EXT };
        VkPhysicalDeviceCoherentMemoryFeaturesAMD                       coherentMemoryFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COHERENT_MEMORY_FEATURES_AMD };
        VkPhysicalDeviceMemoryPriorityFeaturesEXT                       memoryPriorityFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT };
        VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT              fragmentShaderInterlockFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT };
        VkPhysicalDeviceLineRasterizationFeaturesEXT                    lineRasterizationFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_FEATURES_EXT };
        VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT                   shaderAtomicFloat2Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES_EXT };
        VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT               shaderImageAtomicInt64Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES_EXT };
        VkPhysicalDeviceIndexTypeUint8FeaturesEXT                       indexTypeUint8Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_EXT };
        VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR         pipelineExecutablePropertiesFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR };
        VkPhysicalDeviceDeviceGeneratedCommandsFeaturesNV               deviceGeneratedCommandsFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_FEATURES_NV };
        VkPhysicalDeviceDeviceMemoryReportFeaturesEXT                   deviceMemoryReportFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_MEMORY_REPORT_FEATURES_EXT };
        VkPhysicalDeviceShaderEarlyAndLateFragmentTestsFeaturesAMD      shaderEarlyAndLateFragmentTestsFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_EARLY_AND_LATE_FRAGMENT_TESTS_FEATURES_AMD };
        VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR            fragmentShaderBarycentricFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR };
        VkPhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR     subgroupUniformControlFlowFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_FEATURES_KHR };
        VkPhysicalDeviceRayTracingMotionBlurFeaturesNV                  rayTracingMotionBlurFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_MOTION_BLUR_FEATURES_NV };
        VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR        workgroupMemoryExplicitLayout = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR };
        VkPhysicalDeviceRasterizationOrderAttachmentAccessFeaturesARM   rasterizationOrderAttachmentAccessFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_FEATURES_ARM };
        VkPhysicalDeviceColorWriteEnableFeaturesEXT                     colorWriteEnableFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COLOR_WRITE_ENABLE_FEATURES_EXT };
#if 0
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR                     cooperativeMatrixFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR };
#endif
        if (isExtensionSupported(VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME))
            addToPNextChain(&conditionalRenderingFeatures);
        if (isExtensionSupported(VK_KHR_PERFORMANCE_QUERY_EXTENSION_NAME))
            addToPNextChain(&performanceQueryFeatures);
        if (isExtensionSupported(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME))
            addToPNextChain(&accelerationStructureFeatures);
        if (isExtensionSupported(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME))
            addToPNextChain(&rayTracingPipelineFeatures);
        if (isExtensionSupported(VK_NV_SHADER_SM_BUILTINS_EXTENSION_NAME))
            addToPNextChain(&shaderSMBuiltinsFeaturesNV);
        if (isExtensionSupported(VK_NV_REPRESENTATIVE_FRAGMENT_TEST_EXTENSION_NAME))
            addToPNextChain(&representativeFragmentTestFeatures);
        if (isExtensionSupported(VK_KHR_SHADER_CLOCK_EXTENSION_NAME))
            addToPNextChain(&shaderClockFeatures);
        if (isExtensionSupported(VK_NV_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME))
            addToPNextChain(&computeShaderDerivativesFeatures);
        if (isExtensionSupported(VK_NV_SHADER_IMAGE_FOOTPRINT_EXTENSION_NAME))
            addToPNextChain(&shaderImageFootprintFeatures);
        if (isExtensionSupported(VK_INTEL_SHADER_INTEGER_FUNCTIONS_2_EXTENSION_NAME))
            addToPNextChain(&intelShaderIntegerFunctions2);
        if (isExtensionSupported(VK_EXT_FRAGMENT_DENSITY_MAP_EXTENSION_NAME))
            addToPNextChain(&fragmentDensityMapFeatures);
        if (isExtensionSupported(VK_EXT_FRAGMENT_DENSITY_MAP_2_EXTENSION_NAME))
            addToPNextChain(&fragmentDensityMap2Features);
        if (isExtensionSupported(VK_AMD_DEVICE_COHERENT_MEMORY_EXTENSION_NAME))
            addToPNextChain(&coherentMemoryFeatures);
        if (isExtensionSupported(VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME))
            addToPNextChain(&memoryPriorityFeatures);
        if (isExtensionSupported(VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME))
            addToPNextChain(&fragmentShaderInterlockFeatures);
        if (isExtensionSupported(VK_EXT_LINE_RASTERIZATION_EXTENSION_NAME))
            addToPNextChain(&lineRasterizationFeatures);
        if (isExtensionSupported(VK_EXT_SHADER_ATOMIC_FLOAT_2_EXTENSION_NAME))
            addToPNextChain(&shaderAtomicFloat2Features);
        if (isExtensionSupported(VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME))
            addToPNextChain(&shaderImageAtomicInt64Features);
        if (isExtensionSupported(VK_EXT_INDEX_TYPE_UINT8_EXTENSION_NAME))
            addToPNextChain(&indexTypeUint8Features);
        if (isExtensionSupported(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME))
            addToPNextChain(&pipelineExecutablePropertiesFeatures);
        if (isExtensionSupported(VK_NV_DEVICE_GENERATED_COMMANDS_EXTENSION_NAME))
            addToPNextChain(&deviceGeneratedCommandsFeatures);
        if (isExtensionSupported(VK_EXT_DEVICE_MEMORY_REPORT_EXTENSION_NAME))
            addToPNextChain(&deviceMemoryReportFeatures);
        if (isExtensionSupported(VK_AMD_SHADER_EARLY_AND_LATE_FRAGMENT_TESTS_EXTENSION_NAME))
            addToPNextChain(&shaderEarlyAndLateFragmentTestsFeatures);
        if (isExtensionSupported(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME))
            addToPNextChain(&fragmentShaderBarycentricFeatures);
        if (isExtensionSupported(VK_KHR_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_EXTENSION_NAME))
            addToPNextChain(&subgroupUniformControlFlowFeatures);
        if (isExtensionSupported(VK_NV_RAY_TRACING_MOTION_BLUR_EXTENSION_NAME))
            addToPNextChain(&rayTracingMotionBlurFeatures);
        if (isExtensionSupported(VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME))
            addToPNextChain(&workgroupMemoryExplicitLayout);
        if (isExtensionSupported(VK_ARM_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_EXTENSION_NAME))
            addToPNextChain(&rasterizationOrderAttachmentAccessFeatures);
        if (isExtensionSupported(VK_EXT_COLOR_WRITE_ENABLE_EXTENSION_NAME))
            addToPNextChain(&colorWriteEnableFeatures);
        // call
        finalizePNextChain();
        vkGetPhysicalDeviceFeatures2(vk_physicalDevice,&deviceFeatures);


        /* Vulkan 1.0 Core  */
        features.robustBufferAccess = deviceFeatures.features.robustBufferAccess;
            
        if (!deviceFeatures.features.fullDrawIndexUint32)
            return nullptr;
        if (!deviceFeatures.features.imageCubeArray)
            return nullptr;
        if (!deviceFeatures.features.independentBlend)
            return nullptr;
            
        features.geometryShader = deviceFeatures.features.geometryShader;
        features.tessellationShader = deviceFeatures.features.tessellationShader;
            
        if (!deviceFeatures.features.sampleRateShading || !deviceFeatures.features.dualSrcBlend)
            return nullptr;
        properties.limits.logicOp = deviceFeatures.features.logicOp;
            
        if (!deviceFeatures.features.multiDrawIndirect || !deviceFeatures.features.drawIndirectFirstInstance)
            return nullptr;
        if (!deviceFeatures.features.depthClamp || !deviceFeatures.features.depthBiasClamp)
            return nullptr;

        if (!deviceFeatures.features.fillModeNonSolid)
            return nullptr;
            
        features.depthBounds = deviceFeatures.features.depthBounds;

        features.wideLines = deviceFeatures.features.wideLines;
        features.largePoints = deviceFeatures.features.largePoints;

        features.alphaToOne = deviceFeatures.features.alphaToOne;

        if (!deviceFeatures.features.multiViewport)
            return nullptr;

        if (!deviceFeatures.features.samplerAnisotropy)
            return nullptr;

        // no checking of deviceFeatures.features.textureCompression...

        if (!deviceFeatures.features.occlusionQueryPrecise)
            return nullptr;
        features.pipelineStatisticsQuery = deviceFeatures.features.pipelineStatisticsQuery;

        properties.limits.vertexPipelineStoresAndAtomics = deviceFeatures.features.vertexPipelineStoresAndAtomics;
        properties.limits.fragmentStoresAndAtomics = deviceFeatures.features.fragmentStoresAndAtomics;
        properties.limits.shaderTessellationAndGeometryPointSize = deviceFeatures.features.shaderTessellationAndGeometryPointSize;
            
        if (!deviceFeatures.features.shaderImageGatherExtended)
            return nullptr;
        if (!deviceFeatures.features.shaderStorageImageExtendedFormats)
            return nullptr;
        properties.limits.shaderStorageImageMultisample = deviceFeatures.features.shaderStorageImageMultisample;
        properties.limits.shaderStorageImageReadWithoutFormat = deviceFeatures.features.shaderStorageImageReadWithoutFormat;
        if (!deviceFeatures.features.shaderStorageImageWriteWithoutFormat)
            return nullptr;

        if (!deviceFeatures.features.shaderUniformBufferArrayDynamicIndexing || !deviceFeatures.features.shaderSampledImageArrayDynamicIndexing || !deviceFeatures.features.shaderStorageBufferArrayDynamicIndexing)
            return nullptr;
        properties.limits.shaderStorageImageArrayDynamicIndexing = deviceFeatures.features.shaderStorageImageArrayDynamicIndexing;

        if (!deviceFeatures.features.shaderClipDistance)
            return nullptr;
        features.shaderCullDistance = deviceFeatures.features.shaderCullDistance;

        properties.limits.shaderFloat64 = deviceFeatures.features.shaderFloat64;
        if (!deviceFeatures.features.shaderInt16)
            return nullptr;
        if (!deviceFeatures.features.shaderInt64)
            return nullptr;

        features.shaderResourceResidency = deviceFeatures.features.shaderResourceResidency;
        features.shaderResourceMinLod = deviceFeatures.features.shaderResourceMinLod;

        // TODO sparse stuff

        properties.limits.variableMultisampleRate = deviceFeatures.features.variableMultisampleRate;
        if (!deviceFeatures.features.inheritedQueries)
            return nullptr;
            

        /* Vulkan 1.1 Core  */
        if (!vulkan11Features.storageBuffer16BitAccess || !vulkan11Features.uniformAndStorageBuffer16BitAccess)
            return nullptr;
        properties.limits.storagePushConstant16 = vulkan11Features.storagePushConstant16;
        properties.limits.storageInputOutput16 = vulkan11Features.storageInputOutput16;

        if (!vulkan11Features.multiview)
            return nullptr;
        properties.limits.multiviewGeometryShader = vulkan11Features.multiviewGeometryShader;
        properties.limits.multiviewTessellationShader = vulkan11Features.multiviewTessellationShader;

        if (!vulkan11Features.variablePointers || !vulkan11Features.variablePointersStorageBuffer)
            return nullptr;

        // could check protectedMemory and YcbcrConversion but no point in doing so yet
            
        if (!vulkan11Features.shaderDrawParameters)
            return nullptr;

            
        /* Vulkan 1.2 Core  */
        if (!vulkan12Features.samplerMirrorClampToEdge)
            return nullptr;

        properties.limits.drawIndirectCount = vulkan12Features.drawIndirectCount;

        if (!vulkan12Features.storageBuffer8BitAccess || !vulkan12Features.uniformAndStorageBuffer8BitAccess)
            return nullptr;
        properties.limits.storagePushConstant8 = vulkan12Features.storagePushConstant8;

        properties.limits.shaderBufferInt64Atomics = vulkan12Features.shaderBufferInt64Atomics;
        properties.limits.shaderSharedInt64Atomics = vulkan12Features.shaderSharedInt64Atomics;

        properties.limits.shaderFloat16 = vulkan12Features.shaderFloat16;
        if (!vulkan12Features.shaderInt8)
            return nullptr;
            
        if (!vulkan12Features.descriptorIndexing)
            return nullptr;
        // dynamically uniform
        properties.limits.shaderInputAttachmentArrayDynamicIndexing = vulkan12Features.shaderInputAttachmentArrayDynamicIndexing;
        if (!vulkan12Features.shaderUniformTexelBufferArrayDynamicIndexing || !vulkan12Features.shaderStorageTexelBufferArrayDynamicIndexing)
            return nullptr;
        // not uniform at all
        properties.limits.shaderUniformBufferArrayNonUniformIndexing = vulkan12Features.shaderUniformBufferArrayNonUniformIndexing;
        if (!vulkan12Features.shaderSampledImageArrayNonUniformIndexing || !vulkan12Features.shaderStorageBufferArrayNonUniformIndexing || !vulkan12Features.shaderStorageImageArrayNonUniformIndexing)
            return nullptr;
        properties.limits.shaderInputAttachmentArrayNonUniformIndexing = vulkan12Features.shaderInputAttachmentArrayNonUniformIndexing;
        if (!vulkan12Features.shaderUniformTexelBufferArrayNonUniformIndexing || !vulkan12Features.shaderStorageTexelBufferArrayNonUniformIndexing)
            return nullptr;
        // update after bind
        properties.limits.descriptorBindingUniformBufferUpdateAfterBind = vulkan12Features.descriptorBindingUniformBufferUpdateAfterBind;
        if (!vulkan12Features.descriptorBindingSampledImageUpdateAfterBind || !vulkan12Features.descriptorBindingStorageImageUpdateAfterBind ||
            !vulkan12Features.descriptorBindingStorageBufferUpdateAfterBind || !vulkan12Features.descriptorBindingUniformTexelBufferUpdateAfterBind ||
            !vulkan12Features.descriptorBindingStorageTexelBufferUpdateAfterBind || !vulkan12Features.descriptorBindingUpdateUnusedWhilePending)
            return nullptr;
        if (!vulkan12Features.descriptorBindingPartiallyBound || !vulkan12Features.descriptorBindingVariableDescriptorCount)
            return nullptr;
        if (!vulkan12Features.runtimeDescriptorArray)
            return nullptr;

        properties.limits.samplerFilterMinmax = vulkan12Features.samplerFilterMinmax;

        if (!vulkan12Features.scalarBlockLayout)
            return nullptr;
            
        // not checking imageless Framebuffer

        if (!vulkan12Features.uniformBufferStandardLayout)
            return nullptr;

        if (!vulkan12Features.shaderSubgroupExtendedTypes)
            return nullptr;
            
        if (!vulkan12Features.separateDepthStencilLayouts)
            return nullptr;
            
        if (!vulkan12Features.hostQueryReset)
            return nullptr;
            
        if (!vulkan12Features.timelineSemaphore)
            return nullptr;

        if (!vulkan12Features.bufferDeviceAddress)
            return nullptr;
        features.bufferDeviceAddressMultiDevice = vulkan12Features.bufferDeviceAddressMultiDevice;

        if (!vulkan12Features.vulkanMemoryModel || !vulkan12Features.vulkanMemoryModelDeviceScope)
            return nullptr;
        properties.limits.vulkanMemoryModelAvailabilityVisibilityChains = vulkan12Features.vulkanMemoryModelAvailabilityVisibilityChains;

        properties.limits.shaderOutputViewportIndex = vulkan12Features.shaderOutputViewportIndex;
        properties.limits.shaderOutputLayer = vulkan12Features.shaderOutputLayer;

        if (!vulkan12Features.subgroupBroadcastDynamicId)
            return nullptr;


        /* Vulkan 1.3 Core  */
        features.robustImageAccess = vulkan13Features.robustImageAccess;

        // not checking inline uniform blocks yet

        if (!vulkan13Features.pipelineCreationCacheControl)
            return nullptr;

        // not checking privateData

        properties.limits.shaderDemoteToHelperInvocation = vulkan13Features.shaderDemoteToHelperInvocation;
        properties.limits.shaderTerminateInvocation = vulkan13Features.shaderTerminateInvocation;

        if (!vulkan13Features.subgroupSizeControl || !vulkan13Features.computeFullSubgroups)
            return nullptr;

        if (!vulkan13Features.synchronization2)
            return nullptr;

        // not checking textureCompressionASTC_HDR

        properties.limits.shaderZeroInitializeWorkgroupMemory = vulkan13Features.shaderZeroInitializeWorkgroupMemory;

        // not checking dynamicRendering

        if (!vulkan13Features.shaderIntegerDotProduct)
            return nullptr;
        if (!vulkan13Features.maintenance4)
            return nullptr;


        /* Nabla Core Profile as Features */
        if (!shaderAtomicFloatFeatures.shaderBufferFloat32Atomics)
            return nullptr;
        properties.limits.shaderBufferFloat32AtomicAdd = shaderAtomicFloatFeatures.shaderBufferFloat32AtomicAdd;
        properties.limits.shaderBufferFloat64Atomics = shaderAtomicFloatFeatures.shaderBufferFloat64Atomics;
        properties.limits.shaderBufferFloat64AtomicAdd = shaderAtomicFloatFeatures.shaderBufferFloat64AtomicAdd;
        if (!shaderAtomicFloatFeatures.shaderSharedFloat32Atomics)
            return nullptr;
        properties.limits.shaderSharedFloat32AtomicAdd = shaderAtomicFloatFeatures.shaderSharedFloat32AtomicAdd;
        properties.limits.shaderSharedFloat64Atomics = shaderAtomicFloatFeatures.shaderSharedFloat64Atomics;
        properties.limits.shaderSharedFloat64AtomicAdd = shaderAtomicFloatFeatures.shaderSharedFloat64AtomicAdd;
        if (!shaderAtomicFloatFeatures.shaderImageFloat32Atomics)
            return nullptr;
        properties.limits.shaderImageFloat32AtomicAdd = shaderAtomicFloatFeatures.shaderImageFloat32AtomicAdd;
        properties.limits.sparseImageFloat32Atomics = shaderAtomicFloatFeatures.sparseImageFloat32Atomics;
        properties.limits.sparseImageFloat32AtomicAdd = shaderAtomicFloatFeatures.sparseImageFloat32AtomicAdd;

        features.robustBufferAccess2 = robustness2Features.robustBufferAccess2;
        features.robustImageAccess2 = robustness2Features.robustImageAccess2;
        features.nullDescriptor = robustness2Features.nullDescriptor;


        /* Vulkan Extensions as Features */
        if (isExtensionSupported(VK_KHR_SWAPCHAIN_EXTENSION_NAME))
            features.swapchainMode |= E_SWAPCHAIN_MODE::ESM_SURFACE;

        features.shaderInfoAMD = isExtensionSupported(VK_AMD_SHADER_INFO_EXTENSION_NAME);

        if (isExtensionSupported(VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME))
        {
            features.conditionalRendering = conditionalRenderingFeatures.conditionalRendering;
            features.inheritedConditionalRendering = conditionalRenderingFeatures.inheritedConditionalRendering;
        }

        features.geometryShaderPassthrough = isExtensionSupported(VK_NV_GEOMETRY_SHADER_PASSTHROUGH_EXTENSION_NAME);

        features.hdrMetadata = isExtensionSupported(VK_EXT_HDR_METADATA_EXTENSION_NAME);

        if (isExtensionSupported(VK_KHR_PERFORMANCE_QUERY_EXTENSION_NAME))
        {
            features.performanceCounterQueryPools = performanceQueryFeatures.performanceCounterQueryPools;
            features.performanceCounterMultipleQueryPools = performanceQueryFeatures.performanceCounterMultipleQueryPools;
        }

        features.mixedAttachmentSamples = isExtensionSupported(VK_AMD_MIXED_ATTACHMENT_SAMPLES_EXTENSION_NAME) || isExtensionSupported(VK_NV_FRAMEBUFFER_MIXED_SAMPLES_EXTENSION_NAME);

        if (isExtensionSupported(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME))
        {
            features.accelerationStructure = accelerationStructureFeatures.accelerationStructure;
            features.accelerationStructureIndirectBuild = accelerationStructureFeatures.accelerationStructureIndirectBuild;
            features.accelerationStructureHostCommands = accelerationStructureFeatures.accelerationStructureHostCommands;
            if (!accelerationStructureFeatures.descriptorBindingAccelerationStructureUpdateAfterBind)
            {
                logger.log("Not enumerating VkPhysicalDevice %p because it reports features contrary to Vulkan specification!", system::ILogger::ELL_INFO, vk_physicalDevice);
                return nullptr;
            }
        }

        if (isExtensionSupported(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME))
        {
            features.rayTracingPipeline = rayTracingPipelineFeatures.rayTracingPipeline;
            if (!rayTracingPipelineFeatures.rayTracingPipelineTraceRaysIndirect)
            {
                logger.log("Not enumerating VkPhysicalDevice %p because it reports features contrary to Vulkan specification!", system::ILogger::ELL_INFO, vk_physicalDevice);
                return nullptr;
            }
            if (rayTracingPipelineFeatures.rayTraversalPrimitiveCulling && !isExtensionSupported(VK_KHR_RAY_QUERY_EXTENSION_NAME))
            {
                logger.log("Not enumerating VkPhysicalDevice %p because it reports features contrary to Vulkan specification!", system::ILogger::ELL_INFO, vk_physicalDevice);
                return nullptr;
            }
            features.rayTraversalPrimitiveCulling = rayTracingPipelineFeatures.rayTraversalPrimitiveCulling;
        }

        features.rayQuery = isExtensionSupported(VK_KHR_RAY_QUERY_EXTENSION_NAME);

        if (isExtensionSupported(VK_NV_REPRESENTATIVE_FRAGMENT_TEST_EXTENSION_NAME))
            features.representativeFragmentTest = representativeFragmentTestFeatures.representativeFragmentTest;

        features.bufferMarkerAMD = isExtensionSupported(VK_AMD_BUFFER_MARKER_EXTENSION_NAME);

        if (isExtensionSupported(VK_EXT_FRAGMENT_DENSITY_MAP_EXTENSION_NAME))
        {
            features.fragmentDensityMap = fragmentDensityMapFeatures.fragmentDensityMap;
            features.fragmentDensityMapDynamic = fragmentDensityMapFeatures.fragmentDensityMapDynamic;
            features.fragmentDensityMapNonSubsampledImages = fragmentDensityMapFeatures.fragmentDensityMapNonSubsampledImages;
        }

        if (isExtensionSupported(VK_EXT_FRAGMENT_DENSITY_MAP_2_EXTENSION_NAME))
        {
            features.fragmentDensityMapDeferred = fragmentDensityMap2Features.fragmentDensityMapDeferred;
        }

        if (isExtensionSupported(VK_AMD_DEVICE_COHERENT_MEMORY_EXTENSION_NAME))
            features.deviceCoherentMemory = coherentMemoryFeatures.deviceCoherentMemory;

        if (isExtensionSupported(VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME))
            features.memoryPriority = memoryPriorityFeatures.memoryPriority;

        if (isExtensionSupported(VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME))
        {
            features.fragmentShaderPixelInterlock = fragmentShaderInterlockFeatures.fragmentShaderPixelInterlock;
            features.fragmentShaderSampleInterlock = fragmentShaderInterlockFeatures.fragmentShaderSampleInterlock;
            features.fragmentShaderShadingRateInterlock = fragmentShaderInterlockFeatures.fragmentShaderShadingRateInterlock;
        }

        if (isExtensionSupported(VK_EXT_LINE_RASTERIZATION_EXTENSION_NAME))
        {
            features.rectangularLines = lineRasterizationFeatures.rectangularLines;
            features.bresenhamLines = lineRasterizationFeatures.bresenhamLines;
            features.smoothLines = lineRasterizationFeatures.smoothLines;
            features.stippledRectangularLines = lineRasterizationFeatures.stippledRectangularLines;
            features.stippledBresenhamLines = lineRasterizationFeatures.stippledBresenhamLines;
            features.stippledSmoothLines = lineRasterizationFeatures.stippledSmoothLines;
        }

        if (isExtensionSupported(VK_EXT_INDEX_TYPE_UINT8_EXTENSION_NAME))
            features.indexTypeUint8 = indexTypeUint8Features.indexTypeUint8;

        features.deferredHostOperations = isExtensionSupported(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);

        if (isExtensionSupported(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME))
            features.pipelineExecutableInfo = pipelineExecutablePropertiesFeatures.pipelineExecutableInfo;

        if (isExtensionSupported(VK_NV_DEVICE_GENERATED_COMMANDS_EXTENSION_NAME))
            features.deviceGeneratedCommands = deviceGeneratedCommandsFeatures.deviceGeneratedCommands;

        if (isExtensionSupported(VK_NV_RAY_TRACING_MOTION_BLUR_EXTENSION_NAME))
        {
            features.rayTracingMotionBlur = rayTracingMotionBlurFeatures.rayTracingMotionBlur;
            features.rayTracingMotionBlurPipelineTraceRaysIndirect = rayTracingMotionBlurFeatures.rayTracingMotionBlurPipelineTraceRaysIndirect;
        }

        if (isExtensionSupported(VK_ARM_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_EXTENSION_NAME))
        {
            features.rasterizationOrderColorAttachmentAccess = rasterizationOrderAttachmentAccessFeatures.rasterizationOrderColorAttachmentAccess;
            features.rasterizationOrderDepthAttachmentAccess = rasterizationOrderAttachmentAccessFeatures.rasterizationOrderDepthAttachmentAccess;
            features.rasterizationOrderStencilAttachmentAccess = rasterizationOrderAttachmentAccessFeatures.rasterizationOrderStencilAttachmentAccess;
        }
#if 0
        if (isExtensionSupported(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME))
            features.cooperativeMatrixRobustBufferAccess = cooperativeMatrixFeatures.cooperativeMatrixRobustBufferAccess;
#endif


        /* Vulkan Extensions Features as Limits */
        if (isExtensionSupported(VK_NV_SHADER_SM_BUILTINS_EXTENSION_NAME))
            properties.limits.shaderSMBuiltins = shaderSMBuiltinsFeaturesNV.shaderSMBuiltins;

        if (isExtensionSupported(VK_KHR_SHADER_CLOCK_EXTENSION_NAME))
        {
            if (!shaderClockFeatures.shaderSubgroupClock)
                return nullptr;
            properties.limits.shaderDeviceClock = shaderClockFeatures.shaderDeviceClock;
        }

        if (isExtensionSupported(VK_NV_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME))
        {
            properties.limits.computeDerivativeGroupQuads = computeShaderDerivativesFeatures.computeDerivativeGroupQuads;
            properties.limits.computeDerivativeGroupLinear = computeShaderDerivativesFeatures.computeDerivativeGroupLinear;
        }

        if (isExtensionSupported(VK_NV_SHADER_IMAGE_FOOTPRINT_EXTENSION_NAME))
            properties.limits.imageFootprint = shaderImageFootprintFeatures.imageFootprint;

        if (isExtensionSupported(VK_INTEL_SHADER_INTEGER_FUNCTIONS_2_EXTENSION_NAME))
            properties.limits.shaderIntegerFunctions2 = intelShaderIntegerFunctions2.shaderIntegerFunctions2;

        if (isExtensionSupported(VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME))
        {
            properties.limits.shaderImageInt64Atomics = shaderImageAtomicInt64Features.shaderImageInt64Atomics;
            properties.limits.sparseImageInt64Atomics = shaderImageAtomicInt64Features.sparseImageInt64Atomics;
        }

        if (isExtensionSupported(VK_EXT_SHADER_ATOMIC_FLOAT_2_EXTENSION_NAME))
        {
            properties.limits.shaderBufferFloat16Atomics = shaderAtomicFloat2Features.shaderBufferFloat16Atomics;
            properties.limits.shaderBufferFloat16AtomicAdd = shaderAtomicFloat2Features.shaderBufferFloat16AtomicAdd;
            properties.limits.shaderBufferFloat16AtomicMinMax = shaderAtomicFloat2Features.shaderBufferFloat16AtomicMinMax;
            properties.limits.shaderBufferFloat32AtomicMinMax = shaderAtomicFloat2Features.shaderBufferFloat32AtomicMinMax;
            properties.limits.shaderBufferFloat64AtomicMinMax = shaderAtomicFloat2Features.shaderBufferFloat64AtomicMinMax;
            properties.limits.shaderSharedFloat16Atomics = shaderAtomicFloat2Features.shaderSharedFloat16Atomics;
            properties.limits.shaderSharedFloat16AtomicAdd = shaderAtomicFloat2Features.shaderSharedFloat16AtomicAdd;
            properties.limits.shaderSharedFloat16AtomicMinMax = shaderAtomicFloat2Features.shaderSharedFloat16AtomicMinMax;
            properties.limits.shaderSharedFloat32AtomicMinMax = shaderAtomicFloat2Features.shaderSharedFloat32AtomicMinMax;
            properties.limits.shaderSharedFloat64AtomicMinMax = shaderAtomicFloat2Features.shaderSharedFloat64AtomicMinMax;
            properties.limits.shaderImageFloat32AtomicMinMax = shaderAtomicFloat2Features.shaderImageFloat32AtomicMinMax;
            properties.limits.sparseImageFloat32AtomicMinMax = shaderAtomicFloat2Features.sparseImageFloat32AtomicMinMax;
        }

        if (isExtensionSupported(VK_EXT_DEVICE_MEMORY_REPORT_EXTENSION_NAME))
            properties.limits.deviceMemoryReport = deviceMemoryReportFeatures.deviceMemoryReport;

        if (isExtensionSupported(VK_AMD_SHADER_EARLY_AND_LATE_FRAGMENT_TESTS_EXTENSION_NAME))
            properties.limits.shaderEarlyAndLateFragmentTests = shaderEarlyAndLateFragmentTestsFeatures.shaderEarlyAndLateFragmentTests;

        if (isExtensionSupported(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME))
            properties.limits.fragmentShaderBarycentric = fragmentShaderBarycentricFeatures.fragmentShaderBarycentric;

        if (isExtensionSupported(VK_KHR_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_EXTENSION_NAME))
            properties.limits.shaderSubgroupUniformControlFlow = subgroupUniformControlFlowFeatures.shaderSubgroupUniformControlFlow;

        if (isExtensionSupported(VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME))
        {
            properties.limits.workgroupMemoryExplicitLayout = workgroupMemoryExplicitLayout.workgroupMemoryExplicitLayout;
            properties.limits.workgroupMemoryExplicitLayoutScalarBlockLayout = workgroupMemoryExplicitLayout.workgroupMemoryExplicitLayoutScalarBlockLayout;
            properties.limits.workgroupMemoryExplicitLayout8BitAccess = workgroupMemoryExplicitLayout.workgroupMemoryExplicitLayout8BitAccess;
            properties.limits.workgroupMemoryExplicitLayout16BitAccess = workgroupMemoryExplicitLayout.workgroupMemoryExplicitLayout16BitAccess;
        }

        if (isExtensionSupported(VK_EXT_COLOR_WRITE_ENABLE_EXTENSION_NAME))
            properties.limits.colorWriteEnable = colorWriteEnableFeatures.colorWriteEnable;
#if 0 //TODO
        if (isExtensionSupported(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME))
            properties.limits.cooperativeMatrixRobustness = cooperativeMatrixFeatures.robustness;
#endif
    }

    // we compare all limits against the defaults easily!
    const SPhysicalDeviceLimits NablaCoreProfile = {};
    if (!NablaCoreProfile.isSubsetOf(properties.limits))
    {
        logger.log("Not enumerating VkPhysicalDevice %p because its limits do not satisfy Nabla Core Profile!", system::ILogger::ELL_INFO, vk_physicalDevice);
        return nullptr;
    }

    // Get physical device's memory properties
    {
        auto& memoryProperties = initData.memoryProperties;
        VkPhysicalDeviceMemoryProperties2 vk_physicalDeviceMemoryProperties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,nullptr};
        vkGetPhysicalDeviceMemoryProperties2(vk_physicalDevice,&vk_physicalDeviceMemoryProperties);
        memoryProperties.memoryTypeCount = vk_physicalDeviceMemoryProperties.memoryProperties.memoryTypeCount;
        for (uint32_t i=0; i<memoryProperties.memoryTypeCount; ++i)
        {
            memoryProperties.memoryTypes[i].heapIndex = vk_physicalDeviceMemoryProperties.memoryProperties.memoryTypes[i].heapIndex;
            memoryProperties.memoryTypes[i].propertyFlags = getMemoryPropertyFlagsFromVkMemoryPropertyFlags(vk_physicalDeviceMemoryProperties.memoryProperties.memoryTypes[i].propertyFlags);
        }
        memoryProperties.memoryHeapCount = vk_physicalDeviceMemoryProperties.memoryProperties.memoryHeapCount;
        for (uint32_t i=0; i<memoryProperties.memoryHeapCount; ++i)
        {
            memoryProperties.memoryHeaps[i].size = vk_physicalDeviceMemoryProperties.memoryProperties.memoryHeaps[i].size;
            memoryProperties.memoryHeaps[i].flags = static_cast<IDeviceMemoryAllocation::E_MEMORY_HEAP_FLAGS>(vk_physicalDeviceMemoryProperties.memoryProperties.memoryHeaps[i].flags);
        }
    }
        
    // and family props
    {
        core::vector<VkQueueFamilyProperties2> qfamprops;
        {
            uint32_t qfamCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties2(vk_physicalDevice, &qfamCount, nullptr);
            if (qfamCount>ILogicalDevice::MaxQueueFamilies)
                qfamCount = ILogicalDevice::MaxQueueFamilies;
            qfamprops.resize(qfamCount, { VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2,nullptr });
            vkGetPhysicalDeviceQueueFamilyProperties2(vk_physicalDevice, &qfamCount, qfamprops.data());
        }
        core::vector<SQueueFamilyProperties> qfamPropertiesMutable(qfamprops.size());
        auto outIt = qfamPropertiesMutable.begin();
        for (auto in : qfamprops)
        {
            const auto& vkqf = in.queueFamilyProperties;
            outIt->queueCount = vkqf.queueCount;
            outIt->queueFlags = static_cast<IQueue::FAMILY_FLAGS>(vkqf.queueFlags);
            outIt->timestampValidBits = vkqf.timestampValidBits;
            outIt->minImageTransferGranularity = { vkqf.minImageTransferGranularity.width, vkqf.minImageTransferGranularity.height, vkqf.minImageTransferGranularity.depth };
            outIt++;
        }
        initData.qfamProperties = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<const SQueueFamilyProperties>>(qfamPropertiesMutable);
        // added lots of utils to smart_refctd_ptr and `refctd_dynamic_array` but to no avail, have to do manually
        //initData.qfamProperties = core::smart_refctd_dynamic_array<const SQueueFamilyProperties>(reinterpret_cast<core::refctd_dynamic_array<SQueueFamilyProperties>*>(qfamPropertiesMutable.get()));
        //initData.qfamProperties = core::smart_refctd_dynamic_array<const SQueueFamilyProperties>(qfamPropertiesMutable.get());
    }

    // Set Format Usages
    auto anyFlag = [](const VkFormatFeatureFlagBits2 features, const VkFormatFeatureFlagBits2 flags)->bool{return features&flags;};
    auto convert = [anyFlag](const VkFormatFeatureFlagBits2 features)->SFormatImageUsages::SUsage
    {
        SFormatImageUsages::SUsage retval = {};
        retval.sampledImage = anyFlag(features, VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT);
        retval.linearlySampledImage = anyFlag(features, VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_LINEAR_BIT);
        retval.storageImage = anyFlag(features, VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT);
        retval.storageImageAtomic = anyFlag(features, VK_FORMAT_FEATURE_2_STORAGE_IMAGE_ATOMIC_BIT);
        retval.attachment = anyFlag(features, VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT);
        retval.attachmentBlend = anyFlag(features, VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BLEND_BIT);
        retval.blitSrc = anyFlag(features, VK_FORMAT_FEATURE_2_BLIT_SRC_BIT);
        retval.blitDst = anyFlag(features, VK_FORMAT_FEATURE_2_BLIT_DST_BIT);
        retval.transferSrc = anyFlag(features, VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT);
        retval.transferDst = anyFlag(features, VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT);
//        retval.videoDecodeOutput = anyFlag(features, VK_FORMAT_FEATURE_2_VIDEO_DECODE_OUTPUT_BIT_KHR);
//        retval.videoDecodeDPB = anyFlag(features, VK_FORMAT_FEATURE_2_VIDEO_DECODE_DPB_BIT_KHR);
//        retval.videoEncodeInput = anyFlag(features, VK_FORMAT_FEATURE_2_VIDEO_ENCODE_INPUT_BIT_KHR);
//        retval.videoEncodeDPB = anyFlag(features, VK_FORMAT_FEATURE_2_VIDEO_ENCODE_DPB_BIT_KHR);
        retval.storageImageLoadWithoutFormat = anyFlag(features, VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT);
        retval.storageImageStoreWithoutFormat = anyFlag(features, VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT);
        retval.depthCompareSampledImage = anyFlag(features, VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT);
//        retval.hostImageTransfer = anyFlag(features, VK_FORMAT_FEATURE_2_HOST_IMAGE_TRANSFER_BIT);
        //retval.log2MaxSmples = ; // Todo(Erfan)
        return retval;
    };
    for (uint32_t i=0; i<asset::EF_COUNT; ++i)
    {
        const asset::E_FORMAT format = static_cast<asset::E_FORMAT>(i);
        bool skip = false;
        switch (format)
        {
            case asset::EF_PVRTC1_2BPP_UNORM_BLOCK_IMG:
            case asset::EF_PVRTC1_4BPP_UNORM_BLOCK_IMG:
            case asset::EF_PVRTC2_2BPP_UNORM_BLOCK_IMG:
            case asset::EF_PVRTC2_4BPP_UNORM_BLOCK_IMG:
            case asset::EF_PVRTC1_2BPP_SRGB_BLOCK_IMG:
            case asset::EF_PVRTC1_4BPP_SRGB_BLOCK_IMG:
            case asset::EF_PVRTC2_2BPP_SRGB_BLOCK_IMG:
            case asset::EF_PVRTC2_4BPP_SRGB_BLOCK_IMG:
                if (!isExtensionSupported(VK_IMG_FORMAT_PVRTC_EXTENSION_NAME))
                    skip = true;
                break;
        }
        if (skip)
            continue;

        VkFormatProperties3 vk_formatProps = {VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3,nullptr};
        VkFormatProperties2 dummy = {VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2,&vk_formatProps};
        vkGetPhysicalDeviceFormatProperties2(vk_physicalDevice,getVkFormatFromFormat(format),&dummy);

        initData.linearTilingUsages[format] = convert(vk_formatProps.linearTilingFeatures);
        initData.optimalTilingUsages[format] = convert(vk_formatProps.optimalTilingFeatures);

        auto& bufferUsages = initData.bufferUsages[format];
        const VkFormatFeatureFlags2 bufferFeatures = vk_formatProps.bufferFeatures;
        bufferUsages = {};
        bufferUsages.vertexAttribute = anyFlag(bufferFeatures,VK_FORMAT_FEATURE_2_VERTEX_BUFFER_BIT);
        bufferUsages.bufferView = anyFlag(bufferFeatures,VK_FORMAT_FEATURE_2_UNIFORM_TEXEL_BUFFER_BIT);
        bufferUsages.storageBufferView = anyFlag(bufferFeatures,VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_BIT);
        bufferUsages.storageBufferViewAtomic = anyFlag(bufferFeatures,VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_ATOMIC_BIT);
        bufferUsages.accelerationStructureVertex = anyFlag(bufferFeatures,VK_FORMAT_FEATURE_2_ACCELERATION_STRUCTURE_VERTEX_BUFFER_BIT_KHR);
        bufferUsages.storageBufferViewLoadWithoutFormat = anyFlag(bufferFeatures,VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT);
        bufferUsages.storageBufferViewStoreWithoutFormat = anyFlag(bufferFeatures, VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT);
//        bufferUsages.opticalFlowImage = anyFlag(bufferFeatures,VK_FORMAT_FEATURE_2_OPTICAL_FLOW_IMAGE_BIT_NV);
//        bufferUsages.opticalFlowVector = anyFlag(bufferFeatures,VK_FORMAT_FEATURE_2_OPTICAL_FLOW_VECTOR_BIT_NV);
//        bufferUsages.opticalFlowCost = anyFlag(bufferFeatures,VK_FORMAT_FEATURE_2_OPTICAL_FLOW_COST_BIT_NV);
    }
      
    return std::unique_ptr<CVulkanPhysicalDevice>(new CVulkanPhysicalDevice(std::move(initData),rdoc,vk_physicalDevice,std::move(availableFeatureSet)));
}


core::smart_refctd_ptr<ILogicalDevice> CVulkanPhysicalDevice::createLogicalDevice_impl(ILogicalDevice::SCreationParams&& params)
{
    // We might alter it to account for dependancies.
    resolveFeatureDependencies(params.featuresToEnable);
    SFeatures& enabledFeatures = params.featuresToEnable;

    // Important notes on extension dependancies, both instance and device
    /*
        If an extension is supported (as queried by vkEnumerateInstanceExtensionProperties or vkEnumerateDeviceExtensionProperties),
        then required extensions of that extension must also be supported for the same instance or physical device.

        Any device extension that has an instance extension dependency that is not enabled by vkCreateInstance is considered to be unsupported,
        hence it must not be returned by vkEnumerateDeviceExtensionProperties for any VkPhysicalDevice child of the instance. Instance extensions do not have dependencies on device extensions.

        Conclusion: We don't need to specifically check instance extension dependancies but we can do it through apiConnection->getEnableFeatures to hint the user on what might be wrong
    */
    VkDevice vk_device = VK_NULL_HANDLE;
    {
        // required
        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT    shaderAtomicFloatFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT, nullptr };
        VkBaseInStructure* featuresTail = reinterpret_cast<VkBaseInStructure*>(&shaderAtomicFloatFeatures);
        VkPhysicalDeviceVulkan13Features                vulkan13Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &shaderAtomicFloatFeatures };
        VkPhysicalDeviceVulkan12Features                vulkan12Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &vulkan13Features };
        VkPhysicalDeviceVulkan11Features                vulkan11Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &vulkan12Features };
        VkPhysicalDeviceFeatures2                       vk_deviceFeatures2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &vulkan11Features };

        core::unordered_set<core::string> extensionsToEnable;
        // Vulkan has problems with having features in the feature chain that have all values set to false.
        // For example having an empty "RayTracingPipelineFeaturesKHR" in the chain will lead to validation errors for RayQueryONLY applications.
        auto addStructToChain = [](VkBaseInStructure* &tail, void* pStruct) -> void
        {
            VkBaseInStructure* toAdd = reinterpret_cast<VkBaseInStructure*>(pStruct);
            tail->pNext = toAdd;
            tail = toAdd;
        };
        auto enableExtensionIfAvailable = [&](const char* extName, void* pFeatureStruct=nullptr) -> bool
        {
            if (m_extensions.find(extName)!=m_extensions.end())
            {
                extensionsToEnable.insert(extName);
                if (pFeatureStruct)
                    addStructToChain(featuresTail,pFeatureStruct);
                return true;
            }
            else
                return false;
        };
        #define REQUIRE_EXTENSION_IF(COND,.../*NAME,FEATURES=nullptr*/) \
        if ((COND) && !enableExtensionIfAvailable(__VA_ARGS__)) \
            return nullptr


        // Extensions required by Nabla Core Profile
#ifdef _NBL_WINDOWS_API_
        extensionsToEnable.insert(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME); // All Requirements Exist in Vulkan 1.1 (including instance extensions)
#endif
        enableExtensionIfAvailable(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
#ifdef _NBL_WINDOWS_API_
        enableExtensionIfAvailable(VK_KHR_WIN32_KEYED_MUTEX_EXTENSION_NAME);
        extensionsToEnable.insert(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME); // All Requirements Exist in Vulkan 1.1 (including instance extensions)
#endif
        enableExtensionIfAvailable(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
#ifdef _NBL_WINDOWS_API_
        extensionsToEnable.insert(VK_KHR_EXTERNAL_FENCE_WIN32_EXTENSION_NAME); // All Requirements Exist in Vulkan 1.1 (including instance extensions)
#endif
        enableExtensionIfAvailable(VK_KHR_EXTERNAL_FENCE_FD_EXTENSION_NAME);
        enableExtensionIfAvailable(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
        extensionsToEnable.insert(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);

        //! required but has overhead so conditional
        VkPhysicalDeviceRobustness2FeaturesEXT robustness2Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT,nullptr };
        REQUIRE_EXTENSION_IF(enabledFeatures.robustBufferAccess2||enabledFeatures.robustImageAccess2||enabledFeatures.nullDescriptor,VK_EXT_ROBUSTNESS_2_EXTENSION_NAME,&robustness2Features);


        // extensions
        const bool swapchainEnabled = enabledFeatures.swapchainMode.hasFlags(E_SWAPCHAIN_MODE::ESM_SURFACE);
        REQUIRE_EXTENSION_IF(swapchainEnabled,VK_KHR_SWAPCHAIN_EXTENSION_NAME,nullptr);
        {
            // If we reach here then the instance extension VK_KHR_Surface was definitely enabled otherwise the extension wouldn't be reported by physical device
            REQUIRE_EXTENSION_IF(swapchainEnabled,VK_KHR_SWAPCHAIN_MUTABLE_FORMAT_EXTENSION_NAME);
            // TODO: https://github.com/Devsh-Graphics-Programming/Nabla/issues/508
        }

        enableExtensionIfAvailable(VK_AMD_SHADER_TRINARY_MINMAX_EXTENSION_NAME);

        enableExtensionIfAvailable(VK_AMD_SHADER_EXPLICIT_VERTEX_PARAMETER_EXTENSION_NAME);

        enableExtensionIfAvailable(VK_AMD_GPU_SHADER_HALF_FLOAT_EXTENSION_NAME);

        REQUIRE_EXTENSION_IF(enabledFeatures.shaderInfoAMD,VK_AMD_SHADER_INFO_EXTENSION_NAME);

        enableExtensionIfAvailable(VK_AMD_SHADER_IMAGE_LOAD_STORE_LOD_EXTENSION_NAME);

        VkPhysicalDeviceASTCDecodeFeaturesEXT astcDecodeFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ASTC_DECODE_FEATURES_EXT,nullptr };
        enableExtensionIfAvailable(VK_EXT_ASTC_DECODE_MODE_EXTENSION_NAME,&astcDecodeFeatures);

        VkPhysicalDeviceConditionalRenderingFeaturesEXT conditionalRenderingFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT,nullptr };
        REQUIRE_EXTENSION_IF(enabledFeatures.conditionalRendering,VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME,&conditionalRenderingFeatures); // feature dependency taken care of

        enableExtensionIfAvailable(VK_GOOGLE_DISPLAY_TIMING_EXTENSION_NAME);

        REQUIRE_EXTENSION_IF(enabledFeatures.geometryShaderPassthrough,VK_NV_GEOMETRY_SHADER_PASSTHROUGH_EXTENSION_NAME); // feature dependency taken care of

        enableExtensionIfAvailable(VK_EXT_DISCARD_RECTANGLES_EXTENSION_NAME);

        enableExtensionIfAvailable(VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME);

        REQUIRE_EXTENSION_IF(enabledFeatures.hdrMetadata,VK_EXT_HDR_METADATA_EXTENSION_NAME); // feature dependency taken care of

        VkPhysicalDevicePerformanceQueryFeaturesKHR performanceQueryFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PERFORMANCE_QUERY_FEATURES_KHR,nullptr };
        REQUIRE_EXTENSION_IF(enabledFeatures.performanceCounterQueryPools,VK_KHR_PERFORMANCE_QUERY_EXTENSION_NAME,&performanceQueryFeatures); // feature dependency taken care of

        enableExtensionIfAvailable(VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME);

        // thou shalt not refactor, the short-ciruiting and its order is on purpose!
        if (enabledFeatures.mixedAttachmentSamples && (!enableExtensionIfAvailable(VK_NV_FRAMEBUFFER_MIXED_SAMPLES_EXTENSION_NAME) || !enableExtensionIfAvailable(VK_AMD_MIXED_ATTACHMENT_SAMPLES_EXTENSION_NAME)))
            return nullptr;

        enableExtensionIfAvailable(VK_EXT_SHADER_STENCIL_EXPORT_EXTENSION_NAME);

        VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,nullptr };
        REQUIRE_EXTENSION_IF(enabledFeatures.accelerationStructure,VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,&accelerationStructureFeatures); // feature dependency taken care of

        VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipelineFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,nullptr };
        REQUIRE_EXTENSION_IF(enabledFeatures.rayTracingPipeline,VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,&rayTracingPipelineFeatures); // feature dependency taken care of

        VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,nullptr };
        REQUIRE_EXTENSION_IF(enabledFeatures.rayQuery,VK_KHR_RAY_QUERY_EXTENSION_NAME,&rayQueryFeatures); // feature dependency taken care of

        VkPhysicalDeviceShaderSMBuiltinsFeaturesNV shaderSMBuiltinsFeaturesNV = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_FEATURES_NV,nullptr };
        enableExtensionIfAvailable(VK_NV_SHADER_SM_BUILTINS_EXTENSION_NAME,&shaderSMBuiltinsFeaturesNV);

        enableExtensionIfAvailable(VK_EXT_POST_DEPTH_COVERAGE_EXTENSION_NAME);

        VkPhysicalDeviceRepresentativeFragmentTestFeaturesNV representativeFragmentTestFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_REPRESENTATIVE_FRAGMENT_TEST_FEATURES_NV,nullptr };
        REQUIRE_EXTENSION_IF(enabledFeatures.representativeFragmentTest,VK_NV_REPRESENTATIVE_FRAGMENT_TEST_EXTENSION_NAME,&representativeFragmentTestFeatures);

        REQUIRE_EXTENSION_IF(enabledFeatures.bufferMarkerAMD,VK_AMD_BUFFER_MARKER_EXTENSION_NAME);

        VkPhysicalDeviceShaderClockFeaturesKHR shaderClockFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR,nullptr };
        enableExtensionIfAvailable(VK_KHR_SHADER_CLOCK_EXTENSION_NAME,&shaderClockFeatures);

        VkPhysicalDeviceComputeShaderDerivativesFeaturesNV computeShaderDerivativesFeaturesNV = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_NV,nullptr };
        enableExtensionIfAvailable(VK_NV_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME,&computeShaderDerivativesFeaturesNV);

        VkPhysicalDeviceShaderImageFootprintFeaturesNV shaderImageFootprintFeaturesNV = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_FOOTPRINT_FEATURES_NV,nullptr };
        enableExtensionIfAvailable(VK_NV_SHADER_IMAGE_FOOTPRINT_EXTENSION_NAME,&shaderImageFootprintFeaturesNV);

        VkPhysicalDeviceShaderIntegerFunctions2FeaturesINTEL shaderIntegerFunctions2Intel = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_FUNCTIONS_2_FEATURES_INTEL, nullptr };
        enableExtensionIfAvailable(VK_INTEL_SHADER_INTEGER_FUNCTIONS_2_EXTENSION_NAME,&shaderIntegerFunctions2Intel);

        enableExtensionIfAvailable(VK_EXT_PCI_BUS_INFO_EXTENSION_NAME);

        VkPhysicalDeviceFragmentDensityMapFeaturesEXT fragmentDensityMapFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_FEATURES_EXT,nullptr };
        REQUIRE_EXTENSION_IF(enabledFeatures.fragmentDensityMap,VK_EXT_FRAGMENT_DENSITY_MAP_EXTENSION_NAME,&fragmentDensityMapFeatures); // feature dependency taken care of

        enableExtensionIfAvailable(VK_GOOGLE_DECORATE_STRING_EXTENSION_NAME);

        VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT shaderImageAtomicInt64Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES_EXT,nullptr };
        enableExtensionIfAvailable(VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME,&shaderImageAtomicInt64Features);

        VkPhysicalDeviceCoherentMemoryFeaturesAMD coherentMemoryFeaturesAMD = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COHERENT_MEMORY_FEATURES_AMD,nullptr };
        REQUIRE_EXTENSION_IF(enabledFeatures.deviceCoherentMemory,VK_AMD_DEVICE_COHERENT_MEMORY_EXTENSION_NAME,&coherentMemoryFeaturesAMD);

        VkPhysicalDeviceMemoryPriorityFeaturesEXT memoryPriorityFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT,nullptr };
        REQUIRE_EXTENSION_IF(enabledFeatures.memoryPriority,VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME,&memoryPriorityFeatures);

        VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT fragmentShaderInterlockFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT,nullptr };
        REQUIRE_EXTENSION_IF(
            enabledFeatures.fragmentShaderSampleInterlock||
            enabledFeatures.fragmentShaderPixelInterlock||
            enabledFeatures.fragmentShaderShadingRateInterlock,
            VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME,
            &fragmentShaderInterlockFeatures
        );

        VkPhysicalDeviceLineRasterizationFeaturesEXT lineRasterizationFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_FEATURES_EXT,nullptr };
        REQUIRE_EXTENSION_IF(
            enabledFeatures.rectangularLines||
            enabledFeatures.bresenhamLines||
            enabledFeatures.smoothLines||
            enabledFeatures.stippledRectangularLines||
            enabledFeatures.stippledBresenhamLines||
            enabledFeatures.stippledSmoothLines,
            VK_EXT_LINE_RASTERIZATION_EXTENSION_NAME,
            &lineRasterizationFeatures
        );

        VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT shaderAtomicFloat2Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES_EXT,nullptr };
        enableExtensionIfAvailable(VK_EXT_SHADER_ATOMIC_FLOAT_2_EXTENSION_NAME, &shaderAtomicFloat2Features);

        VkPhysicalDeviceIndexTypeUint8FeaturesEXT indexTypeUint8Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_EXT,nullptr };
        REQUIRE_EXTENSION_IF(enabledFeatures.indexTypeUint8,VK_EXT_INDEX_TYPE_UINT8_EXTENSION_NAME,&indexTypeUint8Features);

        REQUIRE_EXTENSION_IF(enabledFeatures.deferredHostOperations,VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);

        VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pipelineExecutablePropertiesFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR,nullptr };
        REQUIRE_EXTENSION_IF(enabledFeatures.pipelineExecutableInfo,VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME,&pipelineExecutablePropertiesFeatures);

        VkPhysicalDeviceDeviceGeneratedCommandsFeaturesNV deviceGeneratedCommandsFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_FEATURES_NV,nullptr };
        REQUIRE_EXTENSION_IF(enabledFeatures.deviceGeneratedCommands,VK_NV_DEVICE_GENERATED_COMMANDS_EXTENSION_NAME,&deviceGeneratedCommandsFeatures);

        VkPhysicalDeviceDeviceMemoryReportFeaturesEXT deviceMemoryReportFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_MEMORY_REPORT_FEATURES_EXT,nullptr };
        enableExtensionIfAvailable(VK_EXT_DEVICE_MEMORY_REPORT_EXTENSION_NAME,&deviceMemoryReportFeatures);

        enableExtensionIfAvailable(VK_AMD_SHADER_EARLY_AND_LATE_FRAGMENT_TESTS_EXTENSION_NAME);

        enableExtensionIfAvailable(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);

        enableExtensionIfAvailable(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME);

        VkPhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR subgroupUniformControlFlowFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_FEATURES_KHR,nullptr };
        enableExtensionIfAvailable(VK_KHR_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_EXTENSION_NAME,&subgroupUniformControlFlowFeatures);

        VkPhysicalDeviceRayTracingMotionBlurFeaturesNV rayTracingMotionBlurFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_MOTION_BLUR_FEATURES_NV,nullptr };
        REQUIRE_EXTENSION_IF(enabledFeatures.rayTracingMotionBlur,VK_NV_RAY_TRACING_MOTION_BLUR_EXTENSION_NAME,&rayTracingMotionBlurFeatures); // feature dependency taken care of

        VkPhysicalDeviceFragmentDensityMap2FeaturesEXT fragmentDensityMap2Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_2_FEATURES_EXT,nullptr };
        REQUIRE_EXTENSION_IF(enabledFeatures.fragmentDensityMapDeferred, VK_EXT_FRAGMENT_DENSITY_MAP_2_EXTENSION_NAME, &fragmentDensityMap2Features);

        VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR workgroupMemoryExplicitLayoutFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR,nullptr };
        enableExtensionIfAvailable(VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME,&workgroupMemoryExplicitLayoutFeatures);

        VkPhysicalDeviceRasterizationOrderAttachmentAccessFeaturesARM rasterizationOrderAttachmentAccessFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_FEATURES_ARM,nullptr };
        REQUIRE_EXTENSION_IF(
            enabledFeatures.rasterizationOrderColorAttachmentAccess||
            enabledFeatures.rasterizationOrderDepthAttachmentAccess||
            enabledFeatures.rasterizationOrderStencilAttachmentAccess,
            VK_ARM_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_EXTENSION_NAME,
            &rasterizationOrderAttachmentAccessFeatures
        );

        VkPhysicalDeviceColorWriteEnableFeaturesEXT colorWriteEnableFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COLOR_WRITE_ENABLE_FEATURES_EXT,nullptr };
        enableExtensionIfAvailable(VK_EXT_COLOR_WRITE_ENABLE_EXTENSION_NAME,&colorWriteEnableFeatures);
#if 0
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperativeMatrixFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,nullptr };
        REQUIRE_EXTENSION_IF(enabledFeatures.cooperativeMatrixRobustBufferAccess,VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME,&cooperativeMatrixFeatures);
#endif

        #undef REQUIRE_EXTENSION_IF


        // prime ourselves with good defaults, we actually re-query all available Vulkan <= MinimumApiVersion features so that by default they're all enabled unless we explicitly disable
        vkGetPhysicalDeviceFeatures2(m_vkPhysicalDevice,&vk_deviceFeatures2);
        const auto& limits = m_initData.properties.limits;

        /* Vulkan 1.0 Core  */
        vk_deviceFeatures2.features.robustBufferAccess = enabledFeatures.robustBufferAccess;
        vk_deviceFeatures2.features.fullDrawIndexUint32 = true; // ROADMAP 2022
        vk_deviceFeatures2.features.imageCubeArray = true; // ROADMAP 2022
        vk_deviceFeatures2.features.independentBlend = true; // ROADMAP 2022
        vk_deviceFeatures2.features.geometryShader = enabledFeatures.geometryShader;
        vk_deviceFeatures2.features.tessellationShader = enabledFeatures.tessellationShader;
        vk_deviceFeatures2.features.sampleRateShading = true; // ROADMAP 2022
        vk_deviceFeatures2.features.dualSrcBlend = true; // good device support
        vk_deviceFeatures2.features.logicOp = limits.logicOp;
        vk_deviceFeatures2.features.multiDrawIndirect = true; // ROADMAP 2022
        vk_deviceFeatures2.features.drawIndirectFirstInstance = true; // ROADMAP 2022
        vk_deviceFeatures2.features.depthClamp = true; // ROADMAP 2022
        vk_deviceFeatures2.features.depthBiasClamp = true; // ROADMAP 2022
        vk_deviceFeatures2.features.fillModeNonSolid = true; // good device support
        vk_deviceFeatures2.features.depthBounds = enabledFeatures.depthBounds;
        vk_deviceFeatures2.features.wideLines = enabledFeatures.wideLines;
        vk_deviceFeatures2.features.largePoints = enabledFeatures.largePoints;
        vk_deviceFeatures2.features.alphaToOne = enabledFeatures.alphaToOne;
        vk_deviceFeatures2.features.multiViewport = true; // good device support
        vk_deviceFeatures2.features.samplerAnisotropy = true; // ROADMAP 2022
        // leave defaulted, enable if supported
        //vk_deviceFeatures2.features.textureCompressionETC2;
        //vk_deviceFeatures2.features.textureCompressionASTC_LDR;
        //vk_deviceFeatures2.features.textureCompressionBC;
        vk_deviceFeatures2.features.occlusionQueryPrecise = true; // ROADMAP 2022
        vk_deviceFeatures2.features.pipelineStatisticsQuery = enabledFeatures.pipelineStatisticsQuery;
        vk_deviceFeatures2.features.vertexPipelineStoresAndAtomics = limits.vertexPipelineStoresAndAtomics;
        vk_deviceFeatures2.features.fragmentStoresAndAtomics = limits.fragmentStoresAndAtomics;
        vk_deviceFeatures2.features.shaderTessellationAndGeometryPointSize = limits.shaderTessellationAndGeometryPointSize;
        vk_deviceFeatures2.features.shaderImageGatherExtended = true; // ubi
        vk_deviceFeatures2.features.shaderStorageImageExtendedFormats = true; // ROADMAP 2022
        vk_deviceFeatures2.features.shaderStorageImageMultisample = limits.shaderStorageImageMultisample;
        vk_deviceFeatures2.features.shaderStorageImageReadWithoutFormat = limits.shaderStorageImageReadWithoutFormat;
        vk_deviceFeatures2.features.shaderStorageImageWriteWithoutFormat = true; // ubi
        vk_deviceFeatures2.features.shaderUniformBufferArrayDynamicIndexing = true; // ROADMAP 2022
        vk_deviceFeatures2.features.shaderSampledImageArrayDynamicIndexing = true; // ROADMAP 2022
        vk_deviceFeatures2.features.shaderStorageBufferArrayDynamicIndexing = true; // ROADMAP 2022
        vk_deviceFeatures2.features.shaderStorageImageArrayDynamicIndexing = limits.shaderStorageImageArrayDynamicIndexing;
        vk_deviceFeatures2.features.shaderClipDistance = true; // good device support
        vk_deviceFeatures2.features.shaderCullDistance = enabledFeatures.shaderCullDistance;
        vk_deviceFeatures2.features.shaderFloat64 = limits.shaderFloat64;
        vk_deviceFeatures2.features.shaderInt64 = true; // always enable
        vk_deviceFeatures2.features.shaderInt16 = true; // always enable
        vk_deviceFeatures2.features.shaderResourceResidency = enabledFeatures.shaderResourceResidency;
        vk_deviceFeatures2.features.shaderResourceMinLod = enabledFeatures.shaderResourceMinLod;
        vk_deviceFeatures2.features.sparseBinding = false; // not implemented yet
        vk_deviceFeatures2.features.sparseResidencyBuffer = false; // not implemented yet
        vk_deviceFeatures2.features.sparseResidencyImage2D = false; // not implemented yet
        vk_deviceFeatures2.features.sparseResidencyImage3D = false; // not implemented yet
        vk_deviceFeatures2.features.sparseResidency2Samples = false; // not implemented yet
        vk_deviceFeatures2.features.sparseResidency4Samples = false; // not implemented yet
        vk_deviceFeatures2.features.sparseResidency8Samples = false; // not implemented yet
        vk_deviceFeatures2.features.sparseResidency16Samples = false; // not implemented yet
        vk_deviceFeatures2.features.sparseResidencyAliased = false; // not implemented yet
        vk_deviceFeatures2.features.variableMultisampleRate = limits.variableMultisampleRate;
        vk_deviceFeatures2.features.inheritedQueries = true; // required

        /* Vulkan 1.1 Core */
        vulkan11Features.storageBuffer16BitAccess = true; // ubi
        vulkan11Features.uniformAndStorageBuffer16BitAccess = true; // ubi
        vulkan11Features.storagePushConstant16 = limits.storagePushConstant16;
        vulkan11Features.storageInputOutput16 = limits.storageInputOutput16;
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#features-requirements
        vulkan11Features.multiview = true; // required
        vulkan11Features.multiviewGeometryShader = limits.multiviewGeometryShader;
        vulkan11Features.multiviewTessellationShader = limits.multiviewTessellationShader;
        vulkan11Features.variablePointers = true; // require for future HLSL references and pointers
        vulkan11Features.variablePointersStorageBuffer = true; // require for future HLSL references and pointers
        // not yet
        vulkan11Features.protectedMemory = false; // not implemented yet
        vulkan11Features.samplerYcbcrConversion = false; // not implemented yet
        vulkan11Features.shaderDrawParameters = true; // ubi
            
        /* Vulkan 1.2 Core */
        vulkan12Features.samplerMirrorClampToEdge = true; // ubiquitous
        vulkan12Features.drawIndirectCount = limits.drawIndirectCount;
        vulkan12Features.storageBuffer8BitAccess = true; // ubiquitous
        vulkan12Features.uniformAndStorageBuffer8BitAccess = true; // ubiquitous
        vulkan12Features.storagePushConstant8 = limits.storagePushConstant8;
        vulkan12Features.shaderBufferInt64Atomics = limits.shaderBufferInt64Atomics;
        vulkan12Features.shaderSharedInt64Atomics = limits.shaderSharedInt64Atomics;
        vulkan12Features.shaderFloat16 = limits.shaderFloat16;
        vulkan12Features.shaderInt8 = true; // ubiquitous
        vulkan12Features.descriptorIndexing = true; // ROADMAP 2022
        vulkan12Features.shaderInputAttachmentArrayDynamicIndexing = limits.shaderInputAttachmentArrayDynamicIndexing;
        vulkan12Features.shaderUniformTexelBufferArrayDynamicIndexing = true; // implied by `descriptorIndexing`
        vulkan12Features.shaderStorageTexelBufferArrayDynamicIndexing = true; // implied by `descriptorIndexing`
        vulkan12Features.shaderUniformBufferArrayNonUniformIndexing = limits.shaderUniformBufferArrayNonUniformIndexing;
        vulkan12Features.shaderSampledImageArrayNonUniformIndexing = true; // implied by `descriptorIndexing`
        vulkan12Features.shaderStorageBufferArrayNonUniformIndexing = true; // implied by `descriptorIndexing`
        vulkan12Features.shaderStorageImageArrayNonUniformIndexing = true; // require
        vulkan12Features.shaderInputAttachmentArrayNonUniformIndexing = limits.shaderInputAttachmentArrayNonUniformIndexing;
        vulkan12Features.shaderUniformTexelBufferArrayNonUniformIndexing = true; // implied by `descriptorIndexing`
        vulkan12Features.shaderStorageTexelBufferArrayNonUniformIndexing = true; // ubiquitous
        vulkan12Features.descriptorBindingUniformBufferUpdateAfterBind = limits.descriptorBindingUniformBufferUpdateAfterBind;
        vulkan12Features.descriptorBindingSampledImageUpdateAfterBind = true; // implied by `descriptorIndexing`
        vulkan12Features.descriptorBindingStorageImageUpdateAfterBind = true; // implied by `descriptorIndexing`
        vulkan12Features.descriptorBindingStorageBufferUpdateAfterBind = true; // implied by `descriptorIndexing`
        vulkan12Features.descriptorBindingUniformTexelBufferUpdateAfterBind = true; // implied by `descriptorIndexing`
        vulkan12Features.descriptorBindingStorageTexelBufferUpdateAfterBind = true; // implied by `descriptorIndexing`
        vulkan12Features.descriptorBindingUpdateUnusedWhilePending = true; // implied by `descriptorIndexing`
        vulkan12Features.descriptorBindingPartiallyBound = true; // implied by `descriptorIndexing`
        vulkan12Features.descriptorBindingVariableDescriptorCount = true; // ubiquitous
        vulkan12Features.runtimeDescriptorArray = true; // implied by `descriptorIndexing`
        vulkan12Features.samplerFilterMinmax = limits.samplerFilterMinmax;
        vulkan12Features.scalarBlockLayout = true; // ROADMAP 2022
        vulkan12Features.imagelessFramebuffer = false; // decided against
        vulkan12Features.uniformBufferStandardLayout = true; // required anyway
        vulkan12Features.shaderSubgroupExtendedTypes = true; // required anyway
        vulkan12Features.separateDepthStencilLayouts = true; // required anyway
        vulkan12Features.hostQueryReset = true; // required anyway
        vulkan12Features.timelineSemaphore = true; // required anyway
        vulkan12Features.bufferDeviceAddress = true; // Vulkan 1.3
        // Some capture tools need this but can't enable this when you set this to false (they're buggy probably, We shouldn't worry about this)
        vulkan12Features.bufferDeviceAddressCaptureReplay = m_rdoc_api!=nullptr;
        vulkan12Features.bufferDeviceAddressMultiDevice = enabledFeatures.bufferDeviceAddressMultiDevice;
        vulkan12Features.vulkanMemoryModel = true; // require
        vulkan12Features.vulkanMemoryModelDeviceScope = true; // require
        vulkan12Features.vulkanMemoryModelAvailabilityVisibilityChains = limits.vulkanMemoryModelAvailabilityVisibilityChains;
        vulkan12Features.shaderOutputViewportIndex = limits.shaderOutputViewportIndex;
        vulkan12Features.shaderOutputLayer = limits.shaderOutputLayer;
        vulkan12Features.subgroupBroadcastDynamicId = true; // ubiquitous
            
        /* Vulkan 1.3 Core */
        vulkan13Features.robustImageAccess = enabledFeatures.robustImageAccess;
        vulkan13Features.inlineUniformBlock = false; // decided against
        vulkan13Features.descriptorBindingInlineUniformBlockUpdateAfterBind = false; // decided against
        vulkan13Features.pipelineCreationCacheControl = true; // require
        vulkan13Features.privateData = false; // decided against
        vulkan13Features.shaderDemoteToHelperInvocation = limits.shaderDemoteToHelperInvocation;
        vulkan13Features.shaderTerminateInvocation = limits.shaderTerminateInvocation;
        vulkan13Features.subgroupSizeControl = true; // require
        vulkan13Features.computeFullSubgroups = true; // require
        vulkan13Features.synchronization2 = true; // require
        // leave defaulted, enable if supported
        //vulkan13Features.textureCompressionASTC_HDR;
        vulkan13Features.shaderZeroInitializeWorkgroupMemory = limits.shaderZeroInitializeWorkgroupMemory;
        vulkan13Features.dynamicRendering = false; // decided against
        vulkan13Features.shaderIntegerDotProduct = true; // require
        vulkan13Features.maintenance4 = true; // require

        /* Nabla Core Profile - Note that we can just set variables, if the extension is not enabled then the struct won't be in the pNext */
        // shaderAtomicFloat [LIMIT SO ENABLE EVERYTHING BY DEFAULT]

        robustness2Features.robustBufferAccess2 = enabledFeatures.robustBufferAccess2;
        robustness2Features.robustImageAccess2 = enabledFeatures.robustImageAccess2;
        robustness2Features.nullDescriptor = enabledFeatures.nullDescriptor;

        //astcDecodeFeatures.decodeModeSharedExponent [LIMIT SO ENABLE EVERYTHING BY DEFAULT]

        conditionalRenderingFeatures.conditionalRendering = enabledFeatures.conditionalRendering;
        conditionalRenderingFeatures.inheritedConditionalRendering = enabledFeatures.inheritedConditionalRendering;

        performanceQueryFeatures.performanceCounterQueryPools = enabledFeatures.performanceCounterQueryPools;
        performanceQueryFeatures.performanceCounterMultipleQueryPools = enabledFeatures.performanceCounterMultipleQueryPools;

        accelerationStructureFeatures.accelerationStructure = enabledFeatures.accelerationStructure;
        accelerationStructureFeatures.accelerationStructureCaptureReplay = m_rdoc_api!=nullptr;
        accelerationStructureFeatures.accelerationStructureIndirectBuild = enabledFeatures.accelerationStructureIndirectBuild;
        accelerationStructureFeatures.accelerationStructureHostCommands = enabledFeatures.accelerationStructureHostCommands;
        accelerationStructureFeatures.descriptorBindingAccelerationStructureUpdateAfterBind = enabledFeatures.accelerationStructure;

        rayTracingPipelineFeatures.rayTracingPipeline = enabledFeatures.rayTracingPipeline;
        rayTracingPipelineFeatures.rayTracingPipelineShaderGroupHandleCaptureReplay = m_rdoc_api!=nullptr;
        rayTracingPipelineFeatures.rayTracingPipelineShaderGroupHandleCaptureReplayMixed = m_rdoc_api!=nullptr;
        rayTracingPipelineFeatures.rayTracingPipelineTraceRaysIndirect = enabledFeatures.rayTracingPipeline;
        rayTracingPipelineFeatures.rayTraversalPrimitiveCulling = enabledFeatures.rayTraversalPrimitiveCulling;

        rayQueryFeatures.rayQuery = enabledFeatures.rayQuery;

        //shaderSMBuiltinsFeaturesNV [LIMIT SO ENABLE EVERYTHING BY DEFAULT]

        representativeFragmentTestFeatures.representativeFragmentTest = enabledFeatures.representativeFragmentTest;

        //shaderClockFeatures [LIMIT SO ENABLE EVERYTHING BY DEFAULT]

        //computeShaderDerivativesFeaturesNV [LIMIT SO ENABLE EVERYTHING BY DEFAULT]

        //shaderImageFootprintFeaturesNV [LIMIT SO ENABLE EVERYTHING BY DEFAULT] 

        //shaderIntegerFunctions2Intel [LIMIT SO ENABLE EVERYTHING BY DEFAULT]

        fragmentDensityMapFeatures.fragmentDensityMap = enabledFeatures.fragmentDensityMap;
        fragmentDensityMapFeatures.fragmentDensityMapDynamic = enabledFeatures.fragmentDensityMapDynamic;
        fragmentDensityMapFeatures.fragmentDensityMapNonSubsampledImages = enabledFeatures.fragmentDensityMapNonSubsampledImages;

        //shaderImageAtomicInt64Features [LIMIT SO ENABLE EVERYTHING BY DEFAULT]

        coherentMemoryFeaturesAMD.deviceCoherentMemory = enabledFeatures.deviceCoherentMemory;

        memoryPriorityFeatures.memoryPriority = enabledFeatures.memoryPriority;

        fragmentShaderInterlockFeatures.fragmentShaderSampleInterlock = enabledFeatures.fragmentShaderSampleInterlock;
        fragmentShaderInterlockFeatures.fragmentShaderPixelInterlock = enabledFeatures.fragmentShaderPixelInterlock;
        fragmentShaderInterlockFeatures.fragmentShaderShadingRateInterlock = enabledFeatures.fragmentShaderShadingRateInterlock;

        lineRasterizationFeatures.rectangularLines = enabledFeatures.rectangularLines;
        lineRasterizationFeatures.bresenhamLines = enabledFeatures.bresenhamLines;
        lineRasterizationFeatures.smoothLines = enabledFeatures.smoothLines;
        lineRasterizationFeatures.stippledRectangularLines = enabledFeatures.stippledRectangularLines;
        lineRasterizationFeatures.stippledBresenhamLines = enabledFeatures.stippledBresenhamLines;
        lineRasterizationFeatures.stippledSmoothLines = enabledFeatures.stippledSmoothLines;

        //shaderAtomicFloat2Features [LIMIT SO ENABLE EVERYTHING BY DEFAULT]

        indexTypeUint8Features.indexTypeUint8 = enabledFeatures.indexTypeUint8;

        pipelineExecutablePropertiesFeatures.pipelineExecutableInfo = enabledFeatures.pipelineExecutableInfo;

        deviceGeneratedCommandsFeatures.deviceGeneratedCommands = enabledFeatures.deviceGeneratedCommands;

        //deviceMemoryReportFeatures [LIMIT SO ENABLE EVERYTHING BY DEFAULT]

        //subgroupUniformControlFlowFeatures [LIMIT SO ENABLE EVERYTHING BY DEFAULT]

        rayTracingMotionBlurFeatures.rayTracingMotionBlur = enabledFeatures.rayTracingMotionBlur;
        rayTracingMotionBlurFeatures.rayTracingMotionBlurPipelineTraceRaysIndirect = enabledFeatures.rayTracingMotionBlurPipelineTraceRaysIndirect;

        fragmentDensityMap2Features.fragmentDensityMapDeferred = enabledFeatures.fragmentDensityMapDeferred;

        //workgroupMemoryExplicitLayoutFeatures [LIMIT SO ENABLE EVERYTHING BY DEFAULT]

        rasterizationOrderAttachmentAccessFeatures.rasterizationOrderColorAttachmentAccess = enabledFeatures.rasterizationOrderColorAttachmentAccess;
        rasterizationOrderAttachmentAccessFeatures.rasterizationOrderDepthAttachmentAccess = enabledFeatures.rasterizationOrderDepthAttachmentAccess;
        rasterizationOrderAttachmentAccessFeatures.rasterizationOrderStencilAttachmentAccess = enabledFeatures.rasterizationOrderStencilAttachmentAccess;

        //colorWriteEnableFeatures [LIMIT SO ENABLE EVERYTHING BY DEFAULT]
#if 0
        cooperativeMatrixFeatures.cooperativeMatrix = true;
        cooperativeMatrixFeatures.cooperativeMatrixRobustBufferAccess = enabledFeatures.cooperativeMatrixRobustBufferAccess;
#endif

        // convert a set into a vector
        core::vector<const char*> extensionStrings(extensionsToEnable.size());
        {
            uint32_t i = 0u;
            for (const auto& feature : extensionsToEnable)
                extensionStrings[i++] = feature.c_str();
        }

        // Create Device
        VkDeviceCreateInfo vk_createInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
        vk_createInfo.pNext = &vk_deviceFeatures2;
        // Vulkan >= 1.1 Device uses createInfo.pNext to use features
        vk_createInfo.pEnabledFeatures = nullptr;
        vk_createInfo.flags = static_cast<VkDeviceCreateFlags>(0); // reserved for future use, by Vulkan

        core::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        for (auto i=0u; i<ILogicalDevice::MaxQueueFamilies; i++)
        if (const auto& qparams=params.queueParams[i]; qparams.count)
        {
            auto& qci = queueCreateInfos.emplace_back();           
            qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            qci.pNext = nullptr;
            qci.queueCount = qparams.count;
            qci.queueFamilyIndex = i;
            qci.flags = static_cast<VkDeviceQueueCreateFlags>(qparams.flags.value);
            qci.pQueuePriorities = qparams.priorities.data();
        }
        vk_createInfo.queueCreateInfoCount = queueCreateInfos.size();
        vk_createInfo.pQueueCreateInfos = queueCreateInfos.data();
    
        vk_createInfo.enabledExtensionCount = static_cast<uint32_t>(extensionStrings.size());
        vk_createInfo.ppEnabledExtensionNames = extensionStrings.data();
        
        if (vkCreateDevice(m_vkPhysicalDevice,&vk_createInfo,nullptr,&vk_device)!=VK_SUCCESS)
            return nullptr;
    }

    if (!params.compilerSet)
        params.compilerSet = core::make_smart_refctd_ptr<asset::CCompilerSet>(core::smart_refctd_ptr(m_initData.system));

    return core::make_smart_refctd_ptr<CVulkanLogicalDevice>(core::smart_refctd_ptr<const IAPIConnection>(m_initData.api),m_rdoc_api,this,vk_device,params);
}

}