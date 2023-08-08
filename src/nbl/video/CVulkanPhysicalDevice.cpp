#include "nbl/video/CVulkanPhysicalDevice.h"
#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{
	
std::unique_ptr<CVulkanPhysicalDevice> CVulkanPhysicalDevice::create(core::smart_refctd_ptr<system::ISystem>&& sys, IAPIConnection* const api, renderdoc_api_t* const rdoc, const VkPhysicalDevice vk_physicalDevice)
{
    IPhysicalDevice::SProperties properties = {};

    // First call just with Vulkan 1.0 API because:
    // "The value of apiVersion may be different than the version returned by vkEnumerateInstanceVersion; either higher or lower.
    //  In such cases, the application must not use functionality that exceeds the version of Vulkan associated with a given object.
    //  The pApiVersion parameter returned by vkEnumerateInstanceVersion is the version associated with a VkInstance and its children,
    //  except for a VkPhysicalDevice and its children. VkPhysicalDeviceProperties::apiVersion is the version associated with a VkPhysicalDevice and its children."
    VkPhysicalDeviceProperties vk_deviceProperties;
    {
        vkGetPhysicalDeviceProperties(vk_physicalDevice,&vk_deviceProperties);
        if (vk_deviceProperties.apiVersion<MinimumVulkanApiVersion)
            return nullptr;

        properties.apiVersion.major = VK_API_VERSION_MAJOR(vk_deviceProperties.apiVersion);
        properties.apiVersion.minor = VK_API_VERSION_MINOR(vk_deviceProperties.apiVersion);
        properties.apiVersion.subminor = 0;
        properties.apiVersion.patch = VK_API_VERSION_PATCH(vk_deviceProperties.apiVersion);

        properties.driverVersion = vk_deviceProperties.driverVersion;
        properties.vendorID = vk_deviceProperties.vendorID;
        properties.deviceID = vk_deviceProperties.deviceID;
        switch(vk_deviceProperties.deviceType)
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
        

        // just enforcing the spec a bit (no point doing this right now since they're the same as the spec, but we raise these limits higher)
        constexpr uint32_t MaxRoadmap2022ImageDim = 8192u;
        constexpr uint32_t MaxRoadmap2022ImageLayers = 2048u;
        // TODO: should we even care about LLVMpipe and SwiftShader if they don't have our pre-requisite features and extensions?
        constexpr uint32_t MaxSwiftShader3DImageDim = 1024u;
        if (vk_deviceProperties.limits.maxImageDimension1D<MaxRoadmap2022ImageDim)
            return nullptr;
        properties.limits.maxImageDimension1D = vk_deviceProperties.limits.maxImageDimension1D;
        if (vk_deviceProperties.limits.maxImageDimension2D<MaxRoadmap2022ImageDim)
            return nullptr;
        properties.limits.maxImageDimension2D = vk_deviceProperties.limits.maxImageDimension2D;
        if (vk_deviceProperties.limits.maxImageDimension3D<MaxSwiftShader3DImageDim)
            return nullptr;
        properties.limits.maxImageDimension3D = vk_deviceProperties.limits.maxImageDimension3D;
        if (vk_deviceProperties.limits.maxImageDimensionCube<MaxRoadmap2022ImageDim)
            return nullptr;
        properties.limits.maxImageDimensionCube = vk_deviceProperties.limits.maxImageDimensionCube;
        if (vk_deviceProperties.limits.maxImageArrayLayers<MaxRoadmap2022ImageLayers)
            return nullptr;
        properties.limits.maxImageArrayLayers = vk_deviceProperties.limits.maxImageArrayLayers;
        if (vk_deviceProperties.limits.maxTexelBufferElements<0x10000u)
            return nullptr;
        properties.limits.maxBufferViewTexels = vk_deviceProperties.limits.maxTexelBufferElements;

        constexpr uint32_t MaxRoadmap2022UniformBufferRange = 0x10000u;
        if (vk_deviceProperties.limits.maxUniformBufferRange<MaxRoadmap2022UniformBufferRange)
            return nullptr;
        properties.limits.maxUBOSize = vk_deviceProperties.limits.maxUniformBufferRange;
        if (vk_deviceProperties.limits.maxStorageBufferRange<(128u<<20u))
            return nullptr;
        properties.limits.maxSSBOSize = vk_deviceProperties.limits.maxStorageBufferRange;
        if (vk_deviceProperties.limits.maxPushConstantsSize<IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE)
            return nullptr;
        properties.limits.maxPushConstantsSize = vk_deviceProperties.limits.maxPushConstantsSize;
        if (vk_deviceProperties.limits.maxMemoryAllocationCount<4096u)
            return nullptr;
        properties.limits.maxMemoryAllocationCount = vk_deviceProperties.limits.maxMemoryAllocationCount;
        if (vk_deviceProperties.limits.maxSamplerAllocationCount<4000u)
            return nullptr;
        properties.limits.maxSamplerAllocationCount = vk_deviceProperties.limits.maxSamplerAllocationCount;

        constexpr uint32_t MinRoadmap2022BufferImageGranularity = 4096u;
        if (vk_deviceProperties.limits.bufferImageGranularity>MinRoadmap2022BufferImageGranularity)
            return nullptr;
        properties.limits.bufferImageGranularity = vk_deviceProperties.limits.bufferImageGranularity;

        // we hardcoded this in the engine to 4
        if (vk_deviceProperties.limits.maxBoundDescriptorSets<4u)
            return nullptr;

        // Max PerStage Descriptors
        //constexpr uint32_t MaxRoadmap2022PerStageSamplers = 64u;
        constexpr uint32_t MaxRoadmap2022PerStageUBOs = 15u;
        constexpr uint32_t MaxRoadmap2022PerStageSSBOs = 30u;
        //constexpr uint32_t MaxRoadmap2022PerStageSampledImages = 200u;
        //constexpr uint32_t MaxRoadmap2022PerStageStorageImages = 16u;
        constexpr uint32_t MaxIntelARCInputAttachments = 7u;
        constexpr uint32_t MaxApplePerStageSamplers = 64u;
        constexpr uint32_t MaxA11PerStageSampledImages = 96u;
        constexpr uint32_t MaxApplePerStageStorageImages = 8u;
        if (vk_deviceProperties.limits.maxPerStageDescriptorSamplers<MaxApplePerStageSamplers)
            return nullptr;
        properties.limits.maxPerStageDescriptorSamplers = vk_deviceProperties.limits.maxPerStageDescriptorSamplers;
        if (vk_deviceProperties.limits.maxPerStageDescriptorUniformBuffers<MaxRoadmap2022PerStageUBOs)
            return nullptr;
        properties.limits.maxPerStageDescriptorUBOs = vk_deviceProperties.limits.maxPerStageDescriptorUniformBuffers;
        if (vk_deviceProperties.limits.maxPerStageDescriptorStorageBuffers<MaxRoadmap2022PerStageSSBOs)
            return nullptr;
        properties.limits.maxPerStageDescriptorSSBOs = vk_deviceProperties.limits.maxPerStageDescriptorStorageBuffers;
        if (vk_deviceProperties.limits.maxPerStageDescriptorSampledImages<MaxA11PerStageSampledImages)
            return nullptr;
        properties.limits.maxPerStageDescriptorImages = vk_deviceProperties.limits.maxPerStageDescriptorSampledImages;
        if (vk_deviceProperties.limits.maxPerStageDescriptorStorageImages<MaxApplePerStageStorageImages)
            return nullptr;
        properties.limits.maxPerStageDescriptorStorageImages = vk_deviceProperties.limits.maxPerStageDescriptorStorageImages;
        if (vk_deviceProperties.limits.maxPerStageDescriptorInputAttachments<MaxIntelARCInputAttachments)
            return nullptr;
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
        if (vk_deviceProperties.limits.maxDescriptorSetInputAttachments<MaxIntelARCInputAttachments)
            return nullptr;
        properties.limits.maxDescriptorSetInputAttachments = vk_deviceProperties.limits.maxDescriptorSetInputAttachments;

        constexpr uint32_t MaxA11InterStageComponents = 124u;
        // we don't even store this as everyone does these limits and we've hardcoded that
        if (vk_deviceProperties.limits.maxVertexInputAttributes<16u || vk_deviceProperties.limits.maxVertexInputAttributeOffset<2047u)
            return nullptr;
        if (vk_deviceProperties.limits.maxVertexInputBindings<16u || vk_deviceProperties.limits.maxVertexInputBindingStride<2048u)
            return nullptr;
        if (vk_deviceProperties.limits.maxVertexOutputComponents<MaxA11InterStageComponents)
            return nullptr;
        properties.limits.maxVertexOutputComponents = vk_deviceProperties.limits.maxVertexOutputComponents;

        // these are conditional on the feature being enabled, not checking
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
        
        constexpr uint32_t MaxRoadmap2022ColorAttachments = 7u;
        constexpr uint32_t MaxRoadmapFragmentCombinedOutputResources = 16u;
        if (vk_deviceProperties.limits.maxFragmentInputComponents<MaxA11InterStageComponents)
            return nullptr;
        properties.limits.maxFragmentInputComponents = vk_deviceProperties.limits.maxFragmentInputComponents;
        if (vk_deviceProperties.limits.maxFragmentOutputAttachments<MaxRoadmap2022ColorAttachments)
            return nullptr;
        properties.limits.maxFragmentOutputAttachments = vk_deviceProperties.limits.maxFragmentOutputAttachments;
        // not everyone supports dual source blending so not checking
        properties.limits.maxFragmentDualSrcAttachments = vk_deviceProperties.limits.maxFragmentDualSrcAttachments;
        if (vk_deviceProperties.limits.maxFragmentCombinedOutputResources<MaxRoadmapFragmentCombinedOutputResources)
            return nullptr;
        properties.limits.maxFragmentCombinedOutputResources = vk_deviceProperties.limits.maxFragmentCombinedOutputResources;

        constexpr uint32_t MaxRoadmap2022ComputeWorkGroupInvocations = 256u;
        constexpr uint32_t MaxRoadmap2022ComputeWorkGroupSize[3] = {MaxRoadmap2022ComputeWorkGroupInvocations,MaxRoadmap2022ComputeWorkGroupInvocations,64u};
        if (vk_deviceProperties.limits.maxComputeSharedMemorySize<0x4000u)
            return nullptr;
        properties.limits.maxComputeSharedMemorySize = vk_deviceProperties.limits.maxComputeSharedMemorySize;
        for (auto i=0u; i<3u; i++)
        if (vk_deviceProperties.limits.maxComputeWorkGroupCount[i]<0xffffu)
            return nullptr;
        properties.limits.maxComputeWorkGroupCount[0] = vk_deviceProperties.limits.maxComputeWorkGroupCount[0];
        properties.limits.maxComputeWorkGroupCount[1] = vk_deviceProperties.limits.maxComputeWorkGroupCount[1];
        properties.limits.maxComputeWorkGroupCount[2] = vk_deviceProperties.limits.maxComputeWorkGroupCount[2];
        if (vk_deviceProperties.limits.maxComputeWorkGroupInvocations<MaxRoadmap2022ComputeWorkGroupInvocations)
            return nullptr;
        properties.limits.maxComputeWorkGroupInvocations = vk_deviceProperties.limits.maxComputeWorkGroupInvocations;
        for (auto i=0u; i<3u; i++)
        if (vk_deviceProperties.limits.maxComputeWorkGroupSize[i]<MaxRoadmap2022ComputeWorkGroupSize[i])
            return nullptr;
        properties.limits.maxWorkgroupSize[0] = vk_deviceProperties.limits.maxComputeWorkGroupSize[0];
        properties.limits.maxWorkgroupSize[1] = vk_deviceProperties.limits.maxComputeWorkGroupSize[1];
        properties.limits.maxWorkgroupSize[2] = vk_deviceProperties.limits.maxComputeWorkGroupSize[2];

        properties.limits.subPixelPrecisionBits = vk_deviceProperties.limits.subPixelPrecisionBits;
        // TODO: report and check `subTexelPrecisionBits` & `mipmapPrecisionBits` ?

        if (vk_deviceProperties.limits.maxDrawIndexedIndexValue!=0xffFFffFFu)
            return nullptr;
        constexpr uint32_t ObservedMaxDrawIndirectCount = 0x1u<<30;
        if (vk_deviceProperties.limits.maxDrawIndirectCount<ObservedMaxDrawIndirectCount)
            return nullptr;
        properties.limits.maxDrawIndirectCount = vk_deviceProperties.limits.maxDrawIndirectCount;
        
        constexpr float MaxRoadmap2022Anisotropy = 16.f;
        //constexpr float MaxRoadmap2022SamplerLodBias = 14.f;
        constexpr float MaxAppleSamplerLodBias = 14.f;
        if (vk_deviceProperties.limits.maxSamplerLodBias<MaxAppleSamplerLodBias)
            return nullptr;
        properties.limits.maxSamplerLodBias = vk_deviceProperties.limits.maxSamplerLodBias;
        if (vk_deviceProperties.limits.maxSamplerAnisotropy<MaxRoadmap2022Anisotropy)
            return nullptr;
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
        properties.limits.maxFramebufferWidth = vk_deviceProperties.limits.maxFramebufferWidth;
        properties.limits.maxFramebufferHeight = vk_deviceProperties.limits.maxFramebufferHeight;
        properties.limits.maxFramebufferLayers = vk_deviceProperties.limits.maxFramebufferLayers;
        properties.limits.framebufferColorSampleCounts = static_cast<asset::IImage::E_SAMPLE_COUNT_FLAGS>(vk_deviceProperties.limits.framebufferColorSampleCounts);
        properties.limits.framebufferDepthSampleCounts = static_cast<asset::IImage::E_SAMPLE_COUNT_FLAGS>(vk_deviceProperties.limits.framebufferDepthSampleCounts);
        properties.limits.framebufferStencilSampleCounts = static_cast<asset::IImage::E_SAMPLE_COUNT_FLAGS>(vk_deviceProperties.limits.framebufferStencilSampleCounts);
        properties.limits.framebufferNoAttachmentsSampleCounts = static_cast<asset::IImage::E_SAMPLE_COUNT_FLAGS>(vk_deviceProperties.limits.framebufferNoAttachmentsSampleCounts);
        if (vk_deviceProperties.limits.maxColorAttachments<MaxRoadmap2022ColorAttachments)
            return nullptr;
        properties.limits.maxColorAttachments = vk_deviceProperties.limits.maxColorAttachments;
        properties.limits.sampledImageColorSampleCounts = static_cast<asset::IImage::E_SAMPLE_COUNT_FLAGS>(vk_deviceProperties.limits.sampledImageColorSampleCounts);
        properties.limits.sampledImageIntegerSampleCounts = static_cast<asset::IImage::E_SAMPLE_COUNT_FLAGS>(vk_deviceProperties.limits.sampledImageIntegerSampleCounts);
        properties.limits.sampledImageDepthSampleCounts = static_cast<asset::IImage::E_SAMPLE_COUNT_FLAGS>(vk_deviceProperties.limits.sampledImageDepthSampleCounts);
        properties.limits.sampledImageStencilSampleCounts = static_cast<asset::IImage::E_SAMPLE_COUNT_FLAGS>(vk_deviceProperties.limits.sampledImageStencilSampleCounts);
        properties.limits.storageImageSampleCounts = static_cast<asset::IImage::E_SAMPLE_COUNT_FLAGS>(vk_deviceProperties.limits.storageImageSampleCounts);
        properties.limits.maxSampleMaskWords = vk_deviceProperties.limits.maxSampleMaskWords;
        properties.limits.timestampComputeAndGraphics = vk_deviceProperties.limits.timestampComputeAndGraphics;
        properties.limits.timestampPeriodInNanoSeconds = vk_deviceProperties.limits.timestampPeriod;
        properties.limits.maxClipDistances = vk_deviceProperties.limits.maxClipDistances;
        properties.limits.maxCullDistances = vk_deviceProperties.limits.maxCullDistances;
        properties.limits.maxCombinedClipAndCullDistances = vk_deviceProperties.limits.maxCombinedClipAndCullDistances;
        properties.limits.discreteQueuePriorities = vk_deviceProperties.limits.discreteQueuePriorities;

        // don't check yet cause Apple
        //constexpr float MinRoadmap2022PointSizeGranularity = 0.125f;
        //constexpr float MinRoadmap2022LineWidthGranularity = 0.5f;
        constexpr float MinApplePointSizeGranularity = 1.f;
        properties.limits.pointSizeRange[0] = vk_deviceProperties.limits.pointSizeRange[0];
        properties.limits.pointSizeRange[1] = vk_deviceProperties.limits.pointSizeRange[1];
        properties.limits.lineWidthRange[0] = vk_deviceProperties.limits.lineWidthRange[0];
        properties.limits.lineWidthRange[1] = vk_deviceProperties.limits.lineWidthRange[1];
        //if (vk_deviceProperties.limits.pointSizeGranularity>MinRoadmap2022PointSizeGranularity)
        //    return nullptr;
        properties.limits.pointSizeGranularity = vk_deviceProperties.limits.pointSizeGranularity;
        if (vk_deviceProperties.limits.pointSizeGranularity>MinApplePointSizeGranularity)
            return nullptr;
        properties.limits.lineWidthGranularity = vk_deviceProperties.limits.lineWidthGranularity;
        properties.limits.strictLines = vk_deviceProperties.limits.strictLines;

        if (!vk_deviceProperties.limits.standardSampleLocations)
            return nullptr;

        properties.limits.optimalBufferCopyOffsetAlignment = vk_deviceProperties.limits.optimalBufferCopyOffsetAlignment;
        properties.limits.optimalBufferCopyRowPitchAlignment = vk_deviceProperties.limits.optimalBufferCopyRowPitchAlignment;
        properties.limits.nonCoherentAtomSize = vk_deviceProperties.limits.nonCoherentAtomSize;

        constexpr uint32_t MaxRoadmap2022PerStageResources = 200u;
        const uint32_t MaxResources = core::min(MaxRoadmap2022PerStageResources,
            properties.limits.maxPerStageDescriptorUBOs+properties.limits.maxPerStageDescriptorSSBOs+
            properties.limits.maxPerStageDescriptorImages+properties.limits.maxPerStageDescriptorStorageImages+
            properties.limits.maxPerStageDescriptorInputAttachments+properties.limits.maxColorAttachments
        );
        if (vk_deviceProperties.limits.maxPerStageResources<MaxResources)
            return nullptr;
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

    {
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
            VkPhysicalDeviceMaintenance4Properties                  maintanance4Properties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES };
            VkPhysicalDeviceSubgroupSizeControlProperties           subgroupSizeControlProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES };
            VkPhysicalDeviceTexelBufferAlignmentProperties          texelBufferAlignmentProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXEL_BUFFER_ALIGNMENT_PROPERTIES };
            VkPhysicalDeviceShaderIntegerDotProductProperties       shaderIntegerDotProductProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES };
            //! Extensions
            VkPhysicalDeviceConservativeRasterizationPropertiesEXT  conservativeRasterizationProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONSERVATIVE_RASTERIZATION_PROPERTIES_EXT };
            VkPhysicalDeviceDiscardRectanglePropertiesEXT           discardRectangleProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DISCARD_RECTANGLE_PROPERTIES_EXT };
            VkPhysicalDeviceExternalMemoryHostPropertiesEXT         externalMemoryHostPropertiesEXT = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT };
            VkPhysicalDevicePCIBusInfoPropertiesEXT                 PCIBusInfoProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT };
            VkPhysicalDeviceSampleLocationsPropertiesEXT            sampleLocationsProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLE_LOCATIONS_PROPERTIES_EXT };
            VkPhysicalDeviceCooperativeMatrixPropertiesNV           cooperativeMatrixProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_NV };
            VkPhysicalDeviceAccelerationStructurePropertiesKHR      accelerationStructureProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR };
            VkPhysicalDeviceRayTracingPipelinePropertiesKHR         rayTracingPipelineProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR };
            VkPhysicalDeviceFragmentDensityMapPropertiesEXT         fragmentDensityMapProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_PROPERTIES_EXT };
            VkPhysicalDeviceFragmentDensityMap2PropertiesEXT        fragmentDensityMap2Properties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_2_PROPERTIES_EXT };
            VkPhysicalDeviceLineRasterizationPropertiesEXT          lineRasterizationPropertiesEXT = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_PROPERTIES_EXT };
            VkPhysicalDeviceVertexAttributeDivisorPropertiesEXT     vertexAttributeDivisorPropertiesEXT = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_PROPERTIES_EXT };
            VkPhysicalDeviceSubpassShadingPropertiesHUAWEI          subpassShadingPropertiesHUAWEI = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBPASS_SHADING_PROPERTIES_HUAWEI };
            VkPhysicalDeviceShaderSMBuiltinsPropertiesNV            shaderSMBuiltinsProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_PROPERTIES_NV };
            VkPhysicalDeviceShaderCoreProperties2AMD                shaderCoreProperties2AMD = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_2_AMD };
            //! This is only written for convenience to avoid getting validation errors otherwise vulkan will just skip any strutctures it doesn't recognize
            if (vk_deviceProperties.apiVersion>=VK_MAKE_API_VERSION(0,1,3,0))
            {
                addToPNextChain(&vulkan13Properties); // old validation layers might complain about this struct in pNext, all is fine
                addToPNextChain(&maintanance4Properties); // old validation layers might complain about this struct in pNext, all is fine
                addToPNextChain(&subgroupSizeControlProperties);
                addToPNextChain(&texelBufferAlignmentProperties);
                addToPNextChain(&shaderIntegerDotProductProperties); // old validation layers might complain about this struct in pNext, all is fine
            }
            else
            {
                if (isExtensionSupported(VK_KHR_MAINTENANCE_4_EXTENSION_NAME))
                    addToPNextChain(&maintanance4Properties);
                if (isExtensionSupported(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME))
                    addToPNextChain(&subgroupSizeControlProperties);
                if (isExtensionSupported(VK_EXT_TEXEL_BUFFER_ALIGNMENT_EXTENSION_NAME))
                    addToPNextChain(&texelBufferAlignmentProperties);
                if (isExtensionSupported(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME))
                    addToPNextChain(&shaderIntegerDotProductProperties);
            }
            if (isExtensionSupported(VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME))
                addToPNextChain(&conservativeRasterizationProperties);
            if (isExtensionSupported(VK_EXT_DISCARD_RECTANGLES_EXTENSION_NAME))
                addToPNextChain(&discardRectangleProperties);
            if (isExtensionSupported(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME))
                addToPNextChain(&externalMemoryHostPropertiesEXT);
            if (isExtensionSupported(VK_EXT_PCI_BUS_INFO_EXTENSION_NAME))
                addToPNextChain(&PCIBusInfoProperties);
            if (isExtensionSupported(VK_EXT_SAMPLE_LOCATIONS_EXTENSION_NAME))
                addToPNextChain(&sampleLocationsProperties);
            if (isExtensionSupported(VK_NV_COOPERATIVE_MATRIX_EXTENSION_NAME))
                addToPNextChain(&cooperativeMatrixProperties);
            if (isExtensionSupported(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME))
                addToPNextChain(&accelerationStructureProperties);
            if (isExtensionSupported(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME))
                addToPNextChain(&rayTracingPipelineProperties);
            if (isExtensionSupported(VK_EXT_FRAGMENT_DENSITY_MAP_EXTENSION_NAME))
                addToPNextChain(&fragmentDensityMapProperties);
            if (isExtensionSupported(VK_EXT_FRAGMENT_DENSITY_MAP_2_EXTENSION_NAME))
                addToPNextChain(&fragmentDensityMap2Properties);
            if (isExtensionSupported(VK_EXT_LINE_RASTERIZATION_EXTENSION_NAME))
                addToPNextChain(&lineRasterizationPropertiesEXT);
            if (isExtensionSupported(VK_EXT_VERTEX_ATTRIBUTE_DIVISOR_EXTENSION_NAME))
                addToPNextChain(&vertexAttributeDivisorPropertiesEXT);
            if (isExtensionSupported(VK_HUAWEI_SUBPASS_SHADING_EXTENSION_NAME))
                addToPNextChain(&subpassShadingPropertiesHUAWEI);
            if (isExtensionSupported(VK_NV_SHADER_SM_BUILTINS_EXTENSION_NAME))
                addToPNextChain(&shaderSMBuiltinsProperties);
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
            properties.limits.maxPerSetDescriptors = vulkan11Properties.maxPerSetDescriptors;
            properties.limits.maxMemoryAllocationSize = vulkan11Properties.maxMemoryAllocationSize;

            /* SubgroupProperties */
            constexpr uint32_t Roadmap2022SubgroupSize = 4u;
            constexpr uint32_t NablaSubgroupOperationMask = VK_SUBGROUP_FEATURE_BASIC_BIT|VK_SUBGROUP_FEATURE_VOTE_BIT|VK_SUBGROUP_FEATURE_BALLOT_BIT|VK_SUBGROUP_FEATURE_SHUFFLE_BIT|VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT;
            if (vulkan11Properties.subgroupSize<Roadmap2022SubgroupSize)
                return nullptr;
            properties.limits.subgroupSize = vulkan11Properties.subgroupSize;
            properties.limits.subgroupOpsShaderStages = static_cast<IGPUShader::E_SHADER_STAGE>(vulkan11Properties.subgroupSupportedStages);
            if (!properties.limits.subgroupOpsShaderStages.hasFlags(IGPUShader::ESS_COMPUTE|IGPUShader::ESS_FRAGMENT))
                return nullptr;
            if ((vulkan11Properties.subgroupSupportedOperations&NablaSubgroupOperationMask)!=NablaSubgroupOperationMask)
                return nullptr;
            properties.limits.shaderSubgroupArithmetic = vulkan11Properties.subgroupSupportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
            properties.limits.shaderSubgroupClustered = vulkan11Properties.subgroupSupportedOperations & VK_SUBGROUP_FEATURE_CLUSTERED_BIT;
            properties.limits.shaderSubgroupQuad = vulkan11Properties.subgroupSupportedOperations & VK_SUBGROUP_FEATURE_QUAD_BIT;
            properties.limits.shaderSubgroupQuadAllStages = vulkan11Properties.subgroupQuadOperationsInAllStages;

            properties.limits.pointClippingBehavior = static_cast<SLimits::E_POINT_CLIPPING_BEHAVIOR>(vulkan11Properties.pointClippingBehavior);

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

            // float control
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
            // TODO: Roadmap2022 says 7, but what is the support in the wild?
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

            properties.limits.filterMinmaxSingleComponentFormats = vulkan12Properties.filterMinmaxSingleComponentFormats;
            properties.limits.filterMinmaxImageComponentMapping = vulkan12Properties.filterMinmaxImageComponentMapping;

            /* Vulkan 1.3 Core  */
            if(vk_deviceProperties.apiVersion>=VK_MAKE_API_VERSION(0,1,3,0)||isExtensionSupported(VK_KHR_MAINTENANCE_4_EXTENSION_NAME))
                properties.limits.maxBufferSize = maintanance4Properties.maxBufferSize;
            else
                properties.limits.maxBufferSize = vulkan11Properties.maxMemoryAllocationSize;

            constexpr uint32_t Roadmap2022MaxSubgroupSize = 4u;
            if(vk_deviceProperties.apiVersion>=VK_MAKE_API_VERSION(0,1,3,0)||isExtensionSupported(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME))
            {
                properties.limits.minSubgroupSize = subgroupSizeControlProperties.minSubgroupSize;
                properties.limits.maxSubgroupSize = subgroupSizeControlProperties.maxSubgroupSize;
                properties.limits.maxComputeWorkgroupSubgroups = subgroupSizeControlProperties.maxComputeWorkgroupSubgroups;
                properties.limits.requiredSubgroupSizeStages = static_cast<asset::IShader::E_SHADER_STAGE>(subgroupSizeControlProperties.requiredSubgroupSizeStages);
            }
            else
            {
                getMinMaxSubgroupSizeFromDriverID(properties.driverID, properties.limits.minSubgroupSize, properties.limits.maxSubgroupSize);
            }
            if (properties.limits.maxSubgroupSize<Roadmap2022MaxSubgroupSize)
                return nullptr;

            if (vk_deviceProperties.apiVersion>=VK_MAKE_API_VERSION(0,1,3,0)||isExtensionSupported(VK_EXT_TEXEL_BUFFER_ALIGNMENT_EXTENSION_NAME))
            {
                properties.limits.storageTexelBufferOffsetAlignmentBytes = texelBufferAlignmentProperties.storageTexelBufferOffsetAlignmentBytes;
                properties.limits.uniformTexelBufferOffsetAlignmentBytes = texelBufferAlignmentProperties.uniformTexelBufferOffsetAlignmentBytes;
            }
            else
            {
                properties.limits.storageTexelBufferOffsetAlignmentBytes = properties.limits.bufferViewAlignment;
                properties.limits.uniformTexelBufferOffsetAlignmentBytes = properties.limits.bufferViewAlignment;
            }

            if (vk_deviceProperties.apiVersion>=VK_MAKE_API_VERSION(0,1,3,0)||isExtensionSupported(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME))
            {
                properties.limits.integerDotProduct8BitUnsignedAccelerated = shaderIntegerDotProductProperties.integerDotProduct8BitUnsignedAccelerated;
                properties.limits.integerDotProduct8BitSignedAccelerated = shaderIntegerDotProductProperties.integerDotProduct8BitSignedAccelerated;
                properties.limits.integerDotProduct8BitMixedSignednessAccelerated = shaderIntegerDotProductProperties.integerDotProduct8BitMixedSignednessAccelerated;
                properties.limits.integerDotProduct4x8BitPackedUnsignedAccelerated = shaderIntegerDotProductProperties.integerDotProduct4x8BitPackedUnsignedAccelerated;
                properties.limits.integerDotProduct4x8BitPackedSignedAccelerated = shaderIntegerDotProductProperties.integerDotProduct4x8BitPackedSignedAccelerated;
                properties.limits.integerDotProduct4x8BitPackedMixedSignednessAccelerated = shaderIntegerDotProductProperties.integerDotProduct4x8BitPackedMixedSignednessAccelerated;
                properties.limits.integerDotProduct16BitUnsignedAccelerated = shaderIntegerDotProductProperties.integerDotProduct16BitUnsignedAccelerated;
                properties.limits.integerDotProduct16BitSignedAccelerated = shaderIntegerDotProductProperties.integerDotProduct16BitSignedAccelerated;
                properties.limits.integerDotProduct16BitMixedSignednessAccelerated = shaderIntegerDotProductProperties.integerDotProduct16BitMixedSignednessAccelerated;
                properties.limits.integerDotProduct32BitUnsignedAccelerated = shaderIntegerDotProductProperties.integerDotProduct32BitUnsignedAccelerated;
                properties.limits.integerDotProduct32BitSignedAccelerated = shaderIntegerDotProductProperties.integerDotProduct32BitSignedAccelerated;
                properties.limits.integerDotProduct32BitMixedSignednessAccelerated = shaderIntegerDotProductProperties.integerDotProduct32BitMixedSignednessAccelerated;
                properties.limits.integerDotProduct64BitUnsignedAccelerated = shaderIntegerDotProductProperties.integerDotProduct64BitUnsignedAccelerated;
                properties.limits.integerDotProduct64BitSignedAccelerated = shaderIntegerDotProductProperties.integerDotProduct64BitSignedAccelerated;
                properties.limits.integerDotProduct64BitMixedSignednessAccelerated = shaderIntegerDotProductProperties.integerDotProduct64BitMixedSignednessAccelerated;
                properties.limits.integerDotProductAccumulatingSaturating8BitUnsignedAccelerated = shaderIntegerDotProductProperties.integerDotProductAccumulatingSaturating8BitUnsignedAccelerated;
                properties.limits.integerDotProductAccumulatingSaturating8BitSignedAccelerated = shaderIntegerDotProductProperties.integerDotProductAccumulatingSaturating8BitSignedAccelerated;
                properties.limits.integerDotProductAccumulatingSaturating8BitMixedSignednessAccelerated = shaderIntegerDotProductProperties.integerDotProductAccumulatingSaturating8BitMixedSignednessAccelerated;
                properties.limits.integerDotProductAccumulatingSaturating4x8BitPackedUnsignedAccelerated = shaderIntegerDotProductProperties.integerDotProductAccumulatingSaturating4x8BitPackedUnsignedAccelerated;
                properties.limits.integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated = shaderIntegerDotProductProperties.integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated;
                properties.limits.integerDotProductAccumulatingSaturating4x8BitPackedMixedSignednessAccelerated = shaderIntegerDotProductProperties.integerDotProductAccumulatingSaturating4x8BitPackedMixedSignednessAccelerated;
                properties.limits.integerDotProductAccumulatingSaturating16BitUnsignedAccelerated = shaderIntegerDotProductProperties.integerDotProductAccumulatingSaturating16BitUnsignedAccelerated;
                properties.limits.integerDotProductAccumulatingSaturating16BitSignedAccelerated = shaderIntegerDotProductProperties.integerDotProductAccumulatingSaturating16BitSignedAccelerated;
                properties.limits.integerDotProductAccumulatingSaturating16BitMixedSignednessAccelerated = shaderIntegerDotProductProperties.integerDotProductAccumulatingSaturating16BitMixedSignednessAccelerated;
                properties.limits.integerDotProductAccumulatingSaturating32BitUnsignedAccelerated = shaderIntegerDotProductProperties.integerDotProductAccumulatingSaturating32BitUnsignedAccelerated;
                properties.limits.integerDotProductAccumulatingSaturating32BitSignedAccelerated = shaderIntegerDotProductProperties.integerDotProductAccumulatingSaturating32BitSignedAccelerated;
                properties.limits.integerDotProductAccumulatingSaturating32BitMixedSignednessAccelerated = shaderIntegerDotProductProperties.integerDotProductAccumulatingSaturating32BitMixedSignednessAccelerated;
                properties.limits.integerDotProductAccumulatingSaturating64BitUnsignedAccelerated = shaderIntegerDotProductProperties.integerDotProductAccumulatingSaturating64BitUnsignedAccelerated;
                properties.limits.integerDotProductAccumulatingSaturating64BitSignedAccelerated = shaderIntegerDotProductProperties.integerDotProductAccumulatingSaturating64BitSignedAccelerated;
                properties.limits.integerDotProductAccumulatingSaturating64BitMixedSignednessAccelerated = shaderIntegerDotProductProperties.integerDotProductAccumulatingSaturating64BitMixedSignednessAccelerated;
            }


            //! Extensions
            
            /* ConservativeRasterizationPropertiesEXT */
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
            
            if (isExtensionSupported(VK_EXT_DISCARD_RECTANGLES_EXTENSION_NAME))
            {
                properties.limits.maxDiscardRectangles = discardRectangleProperties.maxDiscardRectangles;
            }

            if (isExtensionSupported(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME))
            {
                properties.limits.minImportedHostPointerAlignment = externalMemoryHostPropertiesEXT.minImportedHostPointerAlignment;
            }
            
            if (isExtensionSupported(VK_EXT_PCI_BUS_INFO_EXTENSION_NAME))
            {
                properties.limits.pciDomain   = PCIBusInfoProperties.pciDomain;
                properties.limits.pciBus      = PCIBusInfoProperties.pciBus;
                properties.limits.pciDevice   = PCIBusInfoProperties.pciDevice;
                properties.limits.pciFunction = PCIBusInfoProperties.pciFunction;
            }
            
            if (isExtensionSupported(VK_EXT_SAMPLE_LOCATIONS_EXTENSION_NAME))
            {
                properties.limits.sampleLocationSampleCounts          = static_cast<asset::IImage::E_SAMPLE_COUNT_FLAGS>(sampleLocationsProperties.sampleLocationSampleCounts);
                properties.limits.maxSampleLocationGridSize           = sampleLocationsProperties.maxSampleLocationGridSize;
                properties.limits.sampleLocationCoordinateRange[0]    = sampleLocationsProperties.sampleLocationCoordinateRange[0];
                properties.limits.sampleLocationCoordinateRange[1]    = sampleLocationsProperties.sampleLocationCoordinateRange[1];
                properties.limits.sampleLocationSubPixelBits          = sampleLocationsProperties.sampleLocationSubPixelBits;
                properties.limits.variableSampleLocations             = sampleLocationsProperties.variableSampleLocations;
            }
            
            if (isExtensionSupported(VK_NV_COOPERATIVE_MATRIX_EXTENSION_NAME))
            {
                properties.limits.cooperativeMatrixSupportedStages   = static_cast<asset::IShader::E_SHADER_STAGE>(cooperativeMatrixProperties.cooperativeMatrixSupportedStages);
            }
            
            /* AccelerationStructurePropertiesKHR */
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

            /* RayTracingPipelinePropertiesKHR */
            if (isExtensionSupported(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME))
            {
                properties.limits.shaderGroupHandleSize = rayTracingPipelineProperties.shaderGroupHandleSize;
                properties.limits.maxRayRecursionDepth = rayTracingPipelineProperties.maxRayRecursionDepth;
                properties.limits.maxShaderGroupStride = rayTracingPipelineProperties.maxShaderGroupStride;
                properties.limits.shaderGroupBaseAlignment = rayTracingPipelineProperties.shaderGroupBaseAlignment;
                properties.limits.maxRayDispatchInvocationCount = rayTracingPipelineProperties.maxRayDispatchInvocationCount;
                properties.limits.shaderGroupHandleAlignment = rayTracingPipelineProperties.shaderGroupHandleAlignment;
                properties.limits.maxRayHitAttributeSize = rayTracingPipelineProperties.maxRayHitAttributeSize;
            }

            /* VkPhysicalDeviceFragmentDensityMapPropertiesEXT */
            if (isExtensionSupported(VK_EXT_FRAGMENT_DENSITY_MAP_EXTENSION_NAME))
            {
                properties.limits.minFragmentDensityTexelSize = fragmentDensityMapProperties.minFragmentDensityTexelSize;
                properties.limits.maxFragmentDensityTexelSize = fragmentDensityMapProperties.maxFragmentDensityTexelSize;
                properties.limits.fragmentDensityInvocations = fragmentDensityMapProperties.fragmentDensityInvocations;
            }

            /* VkPhysicalDeviceFragmentDensityMapPropertiesEXT */
            if (isExtensionSupported(VK_EXT_FRAGMENT_DENSITY_MAP_2_EXTENSION_NAME))
            {
                properties.limits.subsampledLoads = fragmentDensityMap2Properties.subsampledLoads;
                properties.limits.subsampledCoarseReconstructionEarlyAccess = fragmentDensityMap2Properties.subsampledCoarseReconstructionEarlyAccess;
                properties.limits.maxSubsampledArrayLayers = fragmentDensityMap2Properties.maxSubsampledArrayLayers;
                properties.limits.maxDescriptorSetSubsampledSamplers = fragmentDensityMap2Properties.maxDescriptorSetSubsampledSamplers;
            }

            /* VkPhysicalDeviceLineRasterizationPropertiesEXT */
            if (isExtensionSupported(VK_EXT_LINE_RASTERIZATION_EXTENSION_NAME))
            {
                properties.limits.lineSubPixelPrecisionBits = lineRasterizationPropertiesEXT.lineSubPixelPrecisionBits;
            }

            /* VkPhysicalDeviceVertexAttributeDivisorPropertiesEXT */
            if (isExtensionSupported(VK_EXT_VERTEX_ATTRIBUTE_DIVISOR_EXTENSION_NAME))
            {
                properties.limits.maxVertexAttribDivisor = vertexAttributeDivisorPropertiesEXT.maxVertexAttribDivisor;
            }

            /* VkPhysicalDeviceSubpassShadingPropertiesHUAWEI */
            if (isExtensionSupported(VK_HUAWEI_SUBPASS_SHADING_EXTENSION_NAME))
            {
                properties.limits.maxSubpassShadingWorkgroupSizeAspectRatio = subpassShadingPropertiesHUAWEI.maxSubpassShadingWorkgroupSizeAspectRatio;
            }

            //! Nabla
            if (isExtensionSupported(VK_NV_SHADER_SM_BUILTINS_EXTENSION_NAME))
                properties.limits.computeUnits = shaderSMBuiltinsProperties.shaderSMCount;
            else if(isExtensionSupported(VK_AMD_SHADER_CORE_PROPERTIES_2_EXTENSION_NAME))
                properties.limits.computeUnits = shaderCoreProperties2AMD.activeComputeUnitCount;
            else 
                properties.limits.computeUnits = getMaxComputeUnitsFromDriverID(properties.driverID);
            
            properties.limits.dispatchBase = true;
            properties.limits.allowCommandBufferQueryCopies = true; // TODO: REDO WE NOW SUPPORT PERF QUERIES always true in vk for all query types instead of PerformanceQuery which we don't support at the moment (have VkPhysicalDevicePerformanceQueryPropertiesKHR::allowCommandBufferQueryCopies in mind)
            properties.limits.maxOptimallyResidentWorkgroupInvocations = core::min(core::roundDownToPoT(properties.limits.maxComputeWorkGroupInvocations),512u);
            
            auto invocationsPerComputeUnit = getMaxInvocationsPerComputeUnitsFromDriverID(properties.driverID);
            if(isExtensionSupported(VK_NV_SHADER_SM_BUILTINS_EXTENSION_NAME))
            {
                constexpr auto invocationsPerWarp = 32u; // unless Nvidia changed something recently
                invocationsPerComputeUnit = shaderSMBuiltinsProperties.shaderWarpsPerSM*invocationsPerWarp;
            }

            properties.limits.maxResidentInvocations = properties.limits.computeUnits*invocationsPerComputeUnit;
            
            // constexpr auto beefyGPUWorkgroupMaxOccupancy = 256u; // TODO: find a way to query and report this somehow, persistent threads are very useful!
            // properties.limits.maxResidentInvocations = beefyGPUWorkgroupMaxOccupancy*properties.limits.maxOptimallyResidentWorkgroupInvocations;

            /*
                [NO NABLA SUPPORT] Vulkan 1.0 implementation must support the 1.0 version of SPIR-V and the 1.0 version of the SPIR-V Extended Instructions for GLSL. If the VK_KHR_spirv_1_4 extension is enabled, the implementation must additionally support the 1.4 version of SPIR-V.
                [NO NABLA SUPPORT] A Vulkan 1.1 implementation must support the 1.0, 1.1, 1.2, and 1.3 versions of SPIR-V and the 1.0 version of the SPIR-V Extended Instructions for GLSL.
                A Vulkan 1.2 implementation must support the 1.0, 1.1, 1.2, 1.3, 1.4, and 1.5 versions of SPIR-V and the 1.0 version of the SPIR-V Extended Instructions for GLSL.
                A Vulkan 1.3 implementation must support the 1.0, 1.1, 1.2, 1.3, 1.4, and 1.5 versions of SPIR-V and the 1.0 version of the SPIR-V Extended Instructions for GLSL.
            */
            properties.limits.spirvVersion = asset::IShaderCompiler::E_SPIRV_VERSION::ESV_1_5;
            if (VK_API_VERSION_MINOR(vk_deviceProperties.apiVersion)>=3)
                    properties.limits.spirvVersion = asset::IShaderCompiler::E_SPIRV_VERSION::ESV_1_6;
        }


        // Get physical device's features
        SPhysicalDeviceFeatures features = {};
        
        // ! In Vulkan: These will be reported based on availability of an extension and will be enabled by enabling an extension
        // Table 51. Extension Feature Aliases (vkspec 1.3.211)
        // Extension                               Feature(s)
        // VK_KHR_shader_draw_parameters           shaderDrawParameters
        // VK_KHR_draw_indirect_count              drawIndirectCount
        // VK_KHR_sampler_mirror_clamp_to_edge     samplerMirrorClampToEdge
        // VK_EXT_descriptor_indexing              descriptorIndexing
        // VK_EXT_sampler_filter_minmax            samplerFilterMinmax
        // VK_EXT_shader_viewport_index_layer      shaderOutputViewportIndex, shaderOutputLayer

        // Extensions
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
            VkPhysicalDeviceSubgroupSizeControlFeaturesEXT                  subgroupSizeControlFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT };
            VkPhysicalDeviceShaderTerminateInvocationFeatures               shaderTerminateInvocationFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_TERMINATE_INVOCATION_FEATURES };
            VkPhysicalDeviceShaderIntegerDotProductFeatures                 shaderIntegerDotProductFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES };
            VkPhysicalDeviceBufferDeviceAddressFeaturesKHR                  bufferDeviceAddressFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_KHR };
            VkPhysicalDeviceImageRobustnessFeatures                         imageRobustnessFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES };
            VkPhysicalDevicePipelineCreationCacheControlFeatures            pipelineCreationCacheControlFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_CREATION_CACHE_CONTROL_FEATURES };
            VkPhysicalDeviceShaderDemoteToHelperInvocationFeaturesEXT       shaderDemoteToHelperInvocationFeaturesEXT = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES_EXT };
            VkPhysicalDeviceMaintenance4Features                            maintenance4Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES };
            //! Extensions
            VkPhysicalDeviceCooperativeMatrixFeaturesNV                     cooperativeMatrixFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_NV };
            VkPhysicalDeviceRayQueryFeaturesKHR                             rayQueryFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR };
            VkPhysicalDeviceAccelerationStructureFeaturesKHR                accelerationFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR };
            VkPhysicalDeviceRayTracingPipelineFeaturesKHR                   rayTracingPipelineFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR };
            VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT              fragmentShaderInterlockFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT };
            VkPhysicalDeviceRayTracingMotionBlurFeaturesNV                  rayTracingMotionBlurFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_MOTION_BLUR_FEATURES_NV };
            VkPhysicalDeviceRasterizationOrderAttachmentAccessFeaturesARM   rasterizationOrderAttachmentAccessFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_FEATURES_ARM };
            VkPhysicalDeviceShaderAtomicFloatFeaturesEXT                    shaderAtomicFloatFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT };
            VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT                   shaderAtomicFloat2Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES_EXT };
            VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT               shaderImageAtomicInt64Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES_EXT };
            VkPhysicalDeviceIndexTypeUint8FeaturesEXT                       indexTypeUint8Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_EXT };
            VkPhysicalDeviceShaderClockFeaturesKHR                          shaderClockFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR };
            VkPhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR     subgroupUniformControlFlowFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_FEATURES_KHR };
            VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR        workgroupMemoryExplicitLayout = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR };
            VkPhysicalDeviceComputeShaderDerivativesFeaturesNV              computeShaderDerivativesFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_NV };
            VkPhysicalDeviceCoverageReductionModeFeaturesNV                 coverageReductionModeFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COVERAGE_REDUCTION_MODE_FEATURES_NV };

            VkPhysicalDeviceDeviceGeneratedCommandsFeaturesNV               deviceGeneratedCommandsFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_FEATURES_NV };
            VkPhysicalDeviceMeshShaderFeaturesNV                            meshShaderFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_NV };
            VkPhysicalDeviceRepresentativeFragmentTestFeaturesNV            representativeFragmentTestFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_REPRESENTATIVE_FRAGMENT_TEST_FEATURES_NV };
            VkPhysicalDeviceColorWriteEnableFeaturesEXT                     colorWriteEnableFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COLOR_WRITE_ENABLE_FEATURES_EXT };
            VkPhysicalDeviceConditionalRenderingFeaturesEXT                 conditionalRenderingFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT };
            VkPhysicalDeviceDeviceMemoryReportFeaturesEXT                   deviceMemoryReportFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_MEMORY_REPORT_FEATURES_EXT };
            VkPhysicalDeviceFragmentDensityMapFeaturesEXT                   fragmentDensityMapFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_FEATURES_EXT };
            VkPhysicalDeviceFragmentDensityMap2FeaturesEXT                  fragmentDensityMap2Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_2_FEATURES_EXT };
            VkPhysicalDeviceLineRasterizationFeaturesEXT                    lineRasterizationFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_FEATURES_EXT };
            VkPhysicalDeviceMemoryPriorityFeaturesEXT                       memoryPriorityFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT };
            VkPhysicalDeviceRobustness2FeaturesEXT                          robustness2Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT };
            VkPhysicalDevicePerformanceQueryFeaturesKHR                     performanceQueryFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PERFORMANCE_QUERY_FEATURES_KHR };
            VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR         pipelineExecutablePropertiesFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR };
            VkPhysicalDeviceCoherentMemoryFeaturesAMD                       coherentMemoryFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COHERENT_MEMORY_FEATURES_AMD };
            VkPhysicalDeviceShaderIntegerFunctions2FeaturesINTEL            intelShaderIntegerFunctions2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_FUNCTIONS_2_FEATURES_INTEL };
            VkPhysicalDeviceShaderImageFootprintFeaturesNV                  shaderImageFootprintFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_FOOTPRINT_FEATURES_NV };
            VkPhysicalDeviceTexelBufferAlignmentFeaturesEXT                 texelBufferAlignmentFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXEL_BUFFER_ALIGNMENT_FEATURES_EXT };
            VkPhysicalDeviceGlobalPriorityQueryFeaturesKHR                  globalPriorityFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GLOBAL_PRIORITY_QUERY_FEATURES_KHR };
            VkPhysicalDeviceASTCDecodeFeaturesEXT                           astcDecodeFeaturesEXT = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ASTC_DECODE_FEATURES_EXT };
            VkPhysicalDeviceShaderSMBuiltinsFeaturesNV                      shaderSMBuiltinsFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_FEATURES_NV };
            if (vk_deviceProperties.apiVersion>=VK_MAKE_API_VERSION(0,1,3,0))
            {
                addToPNextChain(&vulkan13Features);
                addToPNextChain(&subgroupSizeControlFeatures);
                addToPNextChain(&shaderTerminateInvocationFeatures);
                addToPNextChain(&shaderIntegerDotProductFeatures);
                addToPNextChain(&bufferDeviceAddressFeatures);
                addToPNextChain(&imageRobustnessFeatures);
                addToPNextChain(&pipelineCreationCacheControlFeatures);
                addToPNextChain(&shaderDemoteToHelperInvocationFeaturesEXT);
                addToPNextChain(&maintenance4Features);
            }
            else
            {
                if (isExtensionSupported(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME))
                    addToPNextChain(&subgroupSizeControlFeatures);
                if (isExtensionSupported(VK_KHR_SHADER_TERMINATE_INVOCATION_EXTENSION_NAME))
                    addToPNextChain(&shaderTerminateInvocationFeatures);
                if (isExtensionSupported(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME))
                    addToPNextChain(&shaderIntegerDotProductFeatures);
                if (isExtensionSupported(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME))
                    addToPNextChain(&bufferDeviceAddressFeatures);
                if (isExtensionSupported(VK_EXT_IMAGE_ROBUSTNESS_EXTENSION_NAME))
                    addToPNextChain(&imageRobustnessFeatures);
                if (isExtensionSupported(VK_EXT_PIPELINE_CREATION_CACHE_CONTROL_EXTENSION_NAME))
                    addToPNextChain(&pipelineCreationCacheControlFeatures);
                if (isExtensionSupported(VK_EXT_SHADER_DEMOTE_TO_HELPER_INVOCATION_EXTENSION_NAME))
                    addToPNextChain(&shaderDemoteToHelperInvocationFeaturesEXT);
                if (isExtensionSupported(VK_KHR_MAINTENANCE_4_EXTENSION_NAME))
                    addToPNextChain(&maintenance4Features);
            }
            if (isExtensionSupported(VK_NV_COOPERATIVE_MATRIX_EXTENSION_NAME))
                addToPNextChain(&cooperativeMatrixFeatures);
            if (isExtensionSupported(VK_KHR_RAY_QUERY_EXTENSION_NAME))
                addToPNextChain(&rayQueryFeatures);
            if (isExtensionSupported(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME))
                addToPNextChain(&accelerationFeatures);
            if (isExtensionSupported(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME))
                addToPNextChain(&rayTracingPipelineFeatures);
            if (isExtensionSupported(VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME))
                addToPNextChain(&fragmentShaderInterlockFeatures);
            if (isExtensionSupported(VK_NV_RAY_TRACING_MOTION_BLUR_EXTENSION_NAME))
                addToPNextChain(&rayTracingMotionBlurFeatures);
            if (isExtensionSupported(VK_ARM_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_EXTENSION_NAME))
                addToPNextChain(&rasterizationOrderAttachmentAccessFeatures);
            if (isExtensionSupported(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME))
                addToPNextChain(&shaderAtomicFloatFeatures);
            if (isExtensionSupported(VK_EXT_SHADER_ATOMIC_FLOAT_2_EXTENSION_NAME))
                addToPNextChain(&shaderAtomicFloat2Features);
            if (isExtensionSupported(VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME))
                addToPNextChain(&shaderImageAtomicInt64Features);
            if (isExtensionSupported(VK_EXT_INDEX_TYPE_UINT8_EXTENSION_NAME))
                addToPNextChain(&indexTypeUint8Features);
            if (isExtensionSupported(VK_KHR_SHADER_CLOCK_EXTENSION_NAME))
                addToPNextChain(&shaderClockFeatures);
            if (isExtensionSupported(VK_KHR_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_EXTENSION_NAME))
                addToPNextChain(&subgroupUniformControlFlowFeatures);
            if (isExtensionSupported(VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME))
                addToPNextChain(&workgroupMemoryExplicitLayout);
            if (isExtensionSupported(VK_NV_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME))
                addToPNextChain(&computeShaderDerivativesFeatures);
            if (isExtensionSupported(VK_NV_COVERAGE_REDUCTION_MODE_EXTENSION_NAME))
                addToPNextChain(&coverageReductionModeFeatures);
            if (isExtensionSupported(VK_NV_DEVICE_GENERATED_COMMANDS_EXTENSION_NAME))
                addToPNextChain(&deviceGeneratedCommandsFeatures);
            if (isExtensionSupported(VK_NV_MESH_SHADER_EXTENSION_NAME))
                addToPNextChain(&meshShaderFeatures);
            if (isExtensionSupported(VK_NV_REPRESENTATIVE_FRAGMENT_TEST_EXTENSION_NAME))
                addToPNextChain(&representativeFragmentTestFeatures);
            if (isExtensionSupported(VK_EXT_COLOR_WRITE_ENABLE_EXTENSION_NAME))
                addToPNextChain(&colorWriteEnableFeatures);
            if (isExtensionSupported(VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME))
                addToPNextChain(&conditionalRenderingFeatures);
            if (isExtensionSupported(VK_EXT_DEVICE_MEMORY_REPORT_EXTENSION_NAME))
                addToPNextChain(&deviceMemoryReportFeatures);
            if (isExtensionSupported(VK_EXT_FRAGMENT_DENSITY_MAP_EXTENSION_NAME))
                addToPNextChain(&fragmentDensityMapFeatures);
            if (isExtensionSupported(VK_EXT_FRAGMENT_DENSITY_MAP_2_EXTENSION_NAME))
                addToPNextChain(&fragmentDensityMap2Features);
            if (isExtensionSupported(VK_EXT_LINE_RASTERIZATION_EXTENSION_NAME))
                addToPNextChain(&lineRasterizationFeatures);
            if (isExtensionSupported(VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME))
                addToPNextChain(&memoryPriorityFeatures);
            if (isExtensionSupported(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME))
                addToPNextChain(&robustness2Features);
            if (isExtensionSupported(VK_KHR_PERFORMANCE_QUERY_EXTENSION_NAME))
                addToPNextChain(&performanceQueryFeatures);
            if (isExtensionSupported(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME))
                addToPNextChain(&pipelineExecutablePropertiesFeatures);
            if (isExtensionSupported(VK_AMD_DEVICE_COHERENT_MEMORY_EXTENSION_NAME))
                addToPNextChain(&coherentMemoryFeatures);
            if (isExtensionSupported(VK_INTEL_SHADER_INTEGER_FUNCTIONS_2_EXTENSION_NAME))
                addToPNextChain(&intelShaderIntegerFunctions2);
            if (isExtensionSupported(VK_NV_SHADER_IMAGE_FOOTPRINT_EXTENSION_NAME))
                addToPNextChain(&shaderImageFootprintFeatures);
            if (isExtensionSupported(VK_EXT_TEXEL_BUFFER_ALIGNMENT_EXTENSION_NAME))
                addToPNextChain(&texelBufferAlignmentFeatures);
            if (isExtensionSupported(VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME))
                addToPNextChain(&globalPriorityFeatures);
            if (isExtensionSupported(VK_NV_SHADER_SM_BUILTINS_EXTENSION_NAME))
                addToPNextChain(&shaderSMBuiltinsFeatures);
            // call
            finalizePNextChain();
            vkGetPhysicalDeviceFeatures2(vk_physicalDevice,&deviceFeatures);

            /* Vulkan 1.0 Core  */
            features.robustBufferAccess = deviceFeatures.features.robustBufferAccess;
            features.geometryShader = deviceFeatures.features.geometryShader;
            features.tessellationShader = deviceFeatures.features.tessellationShader;
            features.dualSrcBlend = deviceFeatures.features.dualSrcBlend;
            features.logicOp = deviceFeatures.features.logicOp;
            features.fillModeNonSolid = deviceFeatures.features.fillModeNonSolid;
            features.depthBounds = deviceFeatures.features.depthBounds;
            features.wideLines = deviceFeatures.features.wideLines;
            features.largePoints = deviceFeatures.features.largePoints;
            features.alphaToOne = deviceFeatures.features.alphaToOne;
            features.multiViewport = deviceFeatures.features.multiViewport;
            features.pipelineStatisticsQuery = deviceFeatures.features.pipelineStatisticsQuery;
            features.shaderStorageImageReadWithoutFormat = deviceFeatures.features.shaderStorageImageReadWithoutFormat;
            features.shaderStorageImageWriteWithoutFormat = deviceFeatures.features.shaderStorageImageWriteWithoutFormat;
            features.shaderClipDistance = deviceFeatures.features.shaderClipDistance;
            features.shaderCullDistance = deviceFeatures.features.shaderCullDistance;
            features.shaderResourceResidency = deviceFeatures.features.shaderResourceResidency;
            features.shaderResourceMinLod = deviceFeatures.features.shaderResourceMinLod; 
            features.variableMultisampleRate = deviceFeatures.features.variableMultisampleRate;
            features.inheritedQueries = deviceFeatures.features.inheritedQueries;
            
            /* Vulkan 1.1 Core  */
            features.shaderDrawParameters = vulkan11Features.shaderDrawParameters;
            
            /* Vulkan 1.2 Core  */
            properties.limits.storageBuffer8BitAccess = vulkan12Features.storageBuffer8BitAccess;
            properties.limits.uniformAndStorageBuffer8BitAccess = vulkan12Features.uniformAndStorageBuffer8BitAccess;
            properties.limits.storagePushConstant8 = vulkan12Features.storagePushConstant8;

            properties.limits.shaderBufferInt64Atomics = vulkan12Features.shaderBufferInt64Atomics;
            properties.limits.shaderSharedInt64Atomics = vulkan12Features.shaderSharedInt64Atomics;

            properties.limits.shaderFloat16 = vulkan12Features.shaderFloat16;
            properties.limits.shaderInt8 = vulkan12Features.shaderInt8;
            
            features.descriptorIndexing = vulkan12Features.descriptorIndexing;
            features.shaderInputAttachmentArrayDynamicIndexing = vulkan12Features.shaderInputAttachmentArrayDynamicIndexing;
            features.shaderUniformTexelBufferArrayDynamicIndexing = vulkan12Features.shaderUniformTexelBufferArrayDynamicIndexing;
            features.shaderStorageTexelBufferArrayDynamicIndexing = vulkan12Features.shaderStorageTexelBufferArrayDynamicIndexing;
            features.shaderUniformBufferArrayNonUniformIndexing = vulkan12Features.shaderUniformBufferArrayNonUniformIndexing;
            features.shaderSampledImageArrayNonUniformIndexing = vulkan12Features.shaderSampledImageArrayNonUniformIndexing;
            features.shaderStorageBufferArrayNonUniformIndexing = vulkan12Features.shaderStorageBufferArrayNonUniformIndexing;
            features.shaderStorageImageArrayNonUniformIndexing = vulkan12Features.shaderStorageImageArrayNonUniformIndexing;
            features.shaderInputAttachmentArrayNonUniformIndexing = vulkan12Features.shaderInputAttachmentArrayNonUniformIndexing;
            features.shaderUniformTexelBufferArrayNonUniformIndexing = vulkan12Features.shaderUniformTexelBufferArrayNonUniformIndexing;
            features.shaderStorageTexelBufferArrayNonUniformIndexing = vulkan12Features.shaderStorageTexelBufferArrayNonUniformIndexing;
            features.descriptorBindingUniformBufferUpdateAfterBind = vulkan12Features.descriptorBindingUniformBufferUpdateAfterBind;
            features.descriptorBindingSampledImageUpdateAfterBind = vulkan12Features.descriptorBindingSampledImageUpdateAfterBind;
            features.descriptorBindingStorageImageUpdateAfterBind = vulkan12Features.descriptorBindingStorageImageUpdateAfterBind;
            features.descriptorBindingStorageBufferUpdateAfterBind = vulkan12Features.descriptorBindingStorageBufferUpdateAfterBind;
            features.descriptorBindingUniformTexelBufferUpdateAfterBind = vulkan12Features.descriptorBindingUniformTexelBufferUpdateAfterBind;
            features.descriptorBindingStorageTexelBufferUpdateAfterBind = vulkan12Features.descriptorBindingStorageTexelBufferUpdateAfterBind;
            features.descriptorBindingUpdateUnusedWhilePending = vulkan12Features.descriptorBindingUpdateUnusedWhilePending;
            features.descriptorBindingPartiallyBound = vulkan12Features.descriptorBindingPartiallyBound;
            features.descriptorBindingVariableDescriptorCount = vulkan12Features.descriptorBindingVariableDescriptorCount;
            features.runtimeDescriptorArray = vulkan12Features.runtimeDescriptorArray;

            features.samplerFilterMinmax = vulkan12Features.samplerFilterMinmax;

            features.bufferDeviceAddress = bufferDeviceAddressFeatures.bufferDeviceAddress;
            features.bufferDeviceAddressMultiDevice = bufferDeviceAddressFeatures.bufferDeviceAddressMultiDevice;

            properties.limits.vulkanMemoryModel = vulkan12Features.vulkanMemoryModel;
            properties.limits.vulkanMemoryModelDeviceScope = vulkan12Features.vulkanMemoryModelDeviceScope;
            properties.limits.vulkanMemoryModelAvailabilityVisibilityChains = vulkan12Features.vulkanMemoryModelAvailabilityVisibilityChains;

            /* Vulkan 1.3 Core  */
            if(vk_deviceProperties.apiVersion>=VK_MAKE_API_VERSION(0,1,3,0)||isExtensionSupported(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME))
            {
                features.subgroupSizeControl  = subgroupSizeControlFeatures.subgroupSizeControl;
                features.computeFullSubgroups = subgroupSizeControlFeatures.computeFullSubgroups;
            }

            if(vk_deviceProperties.apiVersion>=VK_MAKE_API_VERSION(0,1,3,0)||isExtensionSupported(VK_KHR_SHADER_TERMINATE_INVOCATION_EXTENSION_NAME))
            {
                features.shaderTerminateInvocation = shaderTerminateInvocationFeatures.shaderTerminateInvocation;
            }

            if(vk_deviceProperties.apiVersion>=VK_MAKE_API_VERSION(0,1,3,0)||isExtensionSupported(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME))
            {
                features.shaderIntegerDotProduct = shaderIntegerDotProductFeatures.shaderIntegerDotProduct;
                // [TODO] there's a bunch of fields! https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VkPhysicalDeviceShaderIntegerDotProductPropertiesKHR.html
                    // RESPONSE FROM ERFAN: That's not features, that's properties
            }

            if(vk_deviceProperties.apiVersion>=VK_MAKE_API_VERSION(0,1,3,0)||isExtensionSupported(VK_EXT_IMAGE_ROBUSTNESS_EXTENSION_NAME))
            {
                features.robustImageAccess = imageRobustnessFeatures.robustImageAccess;
            }

            if(vk_deviceProperties.apiVersion>=VK_MAKE_API_VERSION(0,1,3,0)||isExtensionSupported(VK_EXT_PIPELINE_CREATION_CACHE_CONTROL_EXTENSION_NAME))
            {
                features.pipelineCreationCacheControl = pipelineCreationCacheControlFeatures.pipelineCreationCacheControl;
            }

            features.shaderDemoteToHelperInvocation = isExtensionSupported(VK_EXT_SHADER_DEMOTE_TO_HELPER_INVOCATION_EXTENSION_NAME);

            if (isExtensionSupported(VK_KHR_MAINTENANCE_4_EXTENSION_NAME))
            {
                properties.limits.workgroupSizeFromSpecConstant = maintenance4Features.maintenance4;
            }

            /* Vulkan Extensions */
            
            if (isExtensionSupported(VK_NV_COOPERATIVE_MATRIX_EXTENSION_NAME))
            {
                features.cooperativeMatrix = cooperativeMatrixFeatures.cooperativeMatrix;
                features.cooperativeMatrixRobustBufferAccess = cooperativeMatrixFeatures.cooperativeMatrixRobustBufferAccess;
            }

            /* RayQueryFeaturesKHR */
            if (isExtensionSupported(VK_KHR_RAY_QUERY_EXTENSION_NAME))
                features.rayQuery = rayQueryFeatures.rayQuery;
            
            /* AccelerationStructureFeaturesKHR */
            if (isExtensionSupported(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME))
            {
                features.accelerationStructure = accelerationFeatures.accelerationStructure;
                features.accelerationStructureIndirectBuild = accelerationFeatures.accelerationStructureIndirectBuild;
                features.accelerationStructureHostCommands = accelerationFeatures.accelerationStructureHostCommands;
                features.descriptorBindingAccelerationStructureUpdateAfterBind = accelerationFeatures.descriptorBindingAccelerationStructureUpdateAfterBind;
            }
            
            /* RayTracingPipelineFeaturesKHR */
            if (isExtensionSupported(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME))
            {
                features.rayTracingPipeline = rayTracingPipelineFeatures.rayTracingPipeline;
                features.rayTracingPipelineTraceRaysIndirect = rayTracingPipelineFeatures.rayTracingPipelineTraceRaysIndirect;
                features.rayTraversalPrimitiveCulling = rayTracingPipelineFeatures.rayTraversalPrimitiveCulling;
            }
            
            /* FragmentShaderInterlockFeaturesEXT */
            if (isExtensionSupported(VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME))
            {
                features.fragmentShaderPixelInterlock = fragmentShaderInterlockFeatures.fragmentShaderPixelInterlock;
                features.fragmentShaderSampleInterlock = fragmentShaderInterlockFeatures.fragmentShaderSampleInterlock;
                features.fragmentShaderShadingRateInterlock = fragmentShaderInterlockFeatures.fragmentShaderShadingRateInterlock;
            }

            /* RayTracingMotionBlurFeaturesNV */
            if (isExtensionSupported(VK_NV_RAY_TRACING_MOTION_BLUR_EXTENSION_NAME))
            {
                features.rayTracingMotionBlur = rayTracingMotionBlurFeatures.rayTracingMotionBlur;
                features.rayTracingMotionBlurPipelineTraceRaysIndirect = rayTracingMotionBlurFeatures.rayTracingMotionBlurPipelineTraceRaysIndirect;
            }

            /* VkPhysicalDeviceRasterizationOrderAttachmentAccessFeaturesARM */
            if (isExtensionSupported(VK_ARM_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_EXTENSION_NAME))
            {
                features.rasterizationOrderColorAttachmentAccess = rasterizationOrderAttachmentAccessFeatures.rasterizationOrderColorAttachmentAccess;
                features.rasterizationOrderDepthAttachmentAccess = rasterizationOrderAttachmentAccessFeatures.rasterizationOrderDepthAttachmentAccess;
                features.rasterizationOrderStencilAttachmentAccess = rasterizationOrderAttachmentAccessFeatures.rasterizationOrderStencilAttachmentAccess;
            }

            /* VkPhysicalDeviceShaderAtomicFloatFeaturesEXT */
            if (isExtensionSupported(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME))
            {
                features.shaderBufferFloat32Atomics = shaderAtomicFloatFeatures.shaderBufferFloat32Atomics;
                features.shaderBufferFloat32AtomicAdd = shaderAtomicFloatFeatures.shaderBufferFloat32AtomicAdd;
                features.shaderBufferFloat64Atomics = shaderAtomicFloatFeatures.shaderBufferFloat64Atomics;
                features.shaderBufferFloat64AtomicAdd = shaderAtomicFloatFeatures.shaderBufferFloat64AtomicAdd;
                features.shaderSharedFloat32Atomics = shaderAtomicFloatFeatures.shaderSharedFloat32Atomics;
                features.shaderSharedFloat32AtomicAdd = shaderAtomicFloatFeatures.shaderSharedFloat32AtomicAdd;
                features.shaderSharedFloat64Atomics = shaderAtomicFloatFeatures.shaderSharedFloat64Atomics;
                features.shaderSharedFloat64AtomicAdd = shaderAtomicFloatFeatures.shaderSharedFloat64AtomicAdd;
                features.shaderImageFloat32Atomics = shaderAtomicFloatFeatures.shaderImageFloat32Atomics;
                features.shaderImageFloat32AtomicAdd = shaderAtomicFloatFeatures.shaderImageFloat32AtomicAdd;
                features.sparseImageFloat32Atomics = shaderAtomicFloatFeatures.sparseImageFloat32Atomics;
                features.sparseImageFloat32AtomicAdd = shaderAtomicFloatFeatures.sparseImageFloat32AtomicAdd;
            }

            /* VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT */
            if (isExtensionSupported(VK_EXT_SHADER_ATOMIC_FLOAT_2_EXTENSION_NAME))
            {
                features.shaderBufferFloat16Atomics = shaderAtomicFloat2Features.shaderBufferFloat16Atomics;
                features.shaderBufferFloat16AtomicAdd = shaderAtomicFloat2Features.shaderBufferFloat16AtomicAdd;
                features.shaderBufferFloat16AtomicMinMax = shaderAtomicFloat2Features.shaderBufferFloat16AtomicMinMax;
                features.shaderBufferFloat32AtomicMinMax = shaderAtomicFloat2Features.shaderBufferFloat32AtomicMinMax;
                features.shaderBufferFloat64AtomicMinMax = shaderAtomicFloat2Features.shaderBufferFloat64AtomicMinMax;
                features.shaderSharedFloat16Atomics = shaderAtomicFloat2Features.shaderSharedFloat16Atomics;
                features.shaderSharedFloat16AtomicAdd = shaderAtomicFloat2Features.shaderSharedFloat16AtomicAdd;
                features.shaderSharedFloat16AtomicMinMax = shaderAtomicFloat2Features.shaderSharedFloat16AtomicMinMax;
                features.shaderSharedFloat32AtomicMinMax = shaderAtomicFloat2Features.shaderSharedFloat32AtomicMinMax;
                features.shaderSharedFloat64AtomicMinMax = shaderAtomicFloat2Features.shaderSharedFloat64AtomicMinMax;
                features.shaderImageFloat32AtomicMinMax = shaderAtomicFloat2Features.shaderImageFloat32AtomicMinMax;
                features.sparseImageFloat32AtomicMinMax = shaderAtomicFloat2Features.sparseImageFloat32AtomicMinMax;
            }

            /* VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT */
            if (isExtensionSupported(VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME))
            {
                features.shaderImageInt64Atomics = shaderImageAtomicInt64Features.shaderImageInt64Atomics;
                features.sparseImageInt64Atomics = shaderImageAtomicInt64Features.sparseImageInt64Atomics;
            }

            /* VkPhysicalDeviceIndexTypeUint8FeaturesEXT */
            if (isExtensionSupported(VK_EXT_INDEX_TYPE_UINT8_EXTENSION_NAME))
            {
                features.indexTypeUint8 = indexTypeUint8Features.indexTypeUint8;
            }

            /* VkPhysicalDeviceShaderClockFeaturesKHR */
            if (isExtensionSupported(VK_KHR_SHADER_CLOCK_EXTENSION_NAME))
            {
                features.shaderDeviceClock = shaderClockFeatures.shaderDeviceClock;
            }

            /* VkPhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR */
            if (isExtensionSupported(VK_KHR_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_EXTENSION_NAME))
            {
                features.shaderSubgroupUniformControlFlow = subgroupUniformControlFlowFeatures.shaderSubgroupUniformControlFlow;
            }

            /* VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR */
            if (isExtensionSupported(VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME))
            {
                features.workgroupMemoryExplicitLayout = workgroupMemoryExplicitLayout.workgroupMemoryExplicitLayout;
                features.workgroupMemoryExplicitLayoutScalarBlockLayout = workgroupMemoryExplicitLayout.workgroupMemoryExplicitLayoutScalarBlockLayout;
                features.workgroupMemoryExplicitLayout8BitAccess = workgroupMemoryExplicitLayout.workgroupMemoryExplicitLayout8BitAccess;
                features.workgroupMemoryExplicitLayout16BitAccess = workgroupMemoryExplicitLayout.workgroupMemoryExplicitLayout16BitAccess;
            }

            /* VkPhysicalDeviceComputeShaderDerivativesFeaturesNV */
            if (isExtensionSupported(VK_NV_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME))
            {
                features.computeDerivativeGroupQuads = computeShaderDerivativesFeatures.computeDerivativeGroupQuads;
                features.computeDerivativeGroupLinear = computeShaderDerivativesFeatures.computeDerivativeGroupLinear;
            }

            /* VkPhysicalDeviceCoverageReductionModeFeaturesNV  */
            if (isExtensionSupported(VK_NV_COVERAGE_REDUCTION_MODE_EXTENSION_NAME))
            {
                features.coverageReductionMode = coverageReductionModeFeatures.coverageReductionMode;
            }

            /* VkPhysicalDeviceDeviceGeneratedCommandsFeaturesNV  */
            if (isExtensionSupported(VK_NV_DEVICE_GENERATED_COMMANDS_EXTENSION_NAME))
            {
                features.deviceGeneratedCommands = deviceGeneratedCommandsFeatures.deviceGeneratedCommands;
            }

            /* VkPhysicalDeviceMeshShaderFeaturesNV  */
            if (isExtensionSupported(VK_NV_MESH_SHADER_EXTENSION_NAME))
            {
                features.meshShader = meshShaderFeatures.meshShader;
                features.taskShader = meshShaderFeatures.taskShader;
            }

            /* VkPhysicalDeviceRepresentativeFragmentTestFeaturesNV  */
            if (isExtensionSupported(VK_NV_REPRESENTATIVE_FRAGMENT_TEST_EXTENSION_NAME))
            {
                features.representativeFragmentTest = representativeFragmentTestFeatures.representativeFragmentTest;
            }

            if (isExtensionSupported(VK_AMD_MIXED_ATTACHMENT_SAMPLES_EXTENSION_NAME) || isExtensionSupported(VK_NV_FRAMEBUFFER_MIXED_SAMPLES_EXTENSION_NAME))
            {
                features.mixedAttachmentSamples = true;
            }

            if (isExtensionSupported(VK_EXT_HDR_METADATA_EXTENSION_NAME))
            {
                features.hdrMetadata = true;
            }

            if (isExtensionSupported(VK_GOOGLE_DISPLAY_TIMING_EXTENSION_NAME))
            {
                features.displayTiming = true;
            }

            if (isExtensionSupported(VK_AMD_RASTERIZATION_ORDER_EXTENSION_NAME))
            {
                features.rasterizationOrder = true;
            }

            if (isExtensionSupported(VK_AMD_SHADER_EXPLICIT_VERTEX_PARAMETER_EXTENSION_NAME))
            {
                features.shaderExplicitVertexParameter = true;
            }

            if (isExtensionSupported(VK_AMD_SHADER_INFO_EXTENSION_NAME))
            {
                features.shaderInfoAMD = true;
            }

            if (isExtensionSupported(VK_AMD_BUFFER_MARKER_EXTENSION_NAME))
            {
                features.bufferMarkerAMD = true;
            }

            /* VkPhysicalDeviceColorWriteEnableFeaturesEXT */
            if (isExtensionSupported(VK_EXT_COLOR_WRITE_ENABLE_EXTENSION_NAME))
            {
                features.colorWriteEnable = colorWriteEnableFeatures.colorWriteEnable;
            }

            /* VkPhysicalDeviceConditionalRenderingFeaturesEXT */
            if (isExtensionSupported(VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME))
            {
                features.conditionalRendering = conditionalRenderingFeatures.conditionalRendering;
                features.inheritedConditionalRendering = conditionalRenderingFeatures.inheritedConditionalRendering;
            }

            /* VkPhysicalDeviceDeviceMemoryReportFeaturesEXT */
            if (isExtensionSupported(VK_EXT_DEVICE_MEMORY_REPORT_EXTENSION_NAME))
            {
                features.deviceMemoryReport = deviceMemoryReportFeatures.deviceMemoryReport;
            }

            /* VkPhysicalDeviceFragmentDensityMapFeaturesEXT */
            if (isExtensionSupported(VK_EXT_FRAGMENT_DENSITY_MAP_EXTENSION_NAME))
            {
                features.fragmentDensityMap = fragmentDensityMapFeatures.fragmentDensityMap;
                features.fragmentDensityMapDynamic = fragmentDensityMapFeatures.fragmentDensityMapDynamic;
                features.fragmentDensityMapNonSubsampledImages = fragmentDensityMapFeatures.fragmentDensityMapNonSubsampledImages;
            }

            /* VkPhysicalDeviceFragmentDensityMap2FeaturesEXT */
            if (isExtensionSupported(VK_EXT_FRAGMENT_DENSITY_MAP_2_EXTENSION_NAME))
            {
                features.fragmentDensityMapDeferred = fragmentDensityMap2Features.fragmentDensityMapDeferred;
            }

            /* VkPhysicalDeviceLineRasterizationFeaturesEXT */
            if (isExtensionSupported(VK_EXT_LINE_RASTERIZATION_EXTENSION_NAME))
            {
                features.rectangularLines = lineRasterizationFeatures.rectangularLines;
                features.bresenhamLines = lineRasterizationFeatures.bresenhamLines;
                features.smoothLines = lineRasterizationFeatures.smoothLines;
                features.stippledRectangularLines = lineRasterizationFeatures.stippledRectangularLines;
                features.stippledBresenhamLines = lineRasterizationFeatures.stippledBresenhamLines;
                features.stippledSmoothLines = lineRasterizationFeatures.stippledSmoothLines;
            }

            /* VkPhysicalDeviceMemoryPriorityFeaturesEXT */
            if (isExtensionSupported(VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME))
            {
                features.memoryPriority = memoryPriorityFeatures.memoryPriority;
            }

            /* VkPhysicalDeviceRobustness2FeaturesEXT */
            if (isExtensionSupported(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME))
            {
                features.robustBufferAccess2 = robustness2Features.robustBufferAccess2;
                features.robustImageAccess2 = robustness2Features.robustImageAccess2;
                features.nullDescriptor = robustness2Features.nullDescriptor;
            }

            /* VkPhysicalDevicePerformanceQueryFeaturesKHR */
            if (isExtensionSupported(VK_KHR_PERFORMANCE_QUERY_EXTENSION_NAME))
            {
                features.performanceCounterQueryPools = performanceQueryFeatures.performanceCounterQueryPools;
                features.performanceCounterMultipleQueryPools = performanceQueryFeatures.performanceCounterMultipleQueryPools;
            }

            /* VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR */
            if (isExtensionSupported(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME))
            {
                features.pipelineExecutableInfo = pipelineExecutablePropertiesFeatures.pipelineExecutableInfo;
            }

            /* VkPhysicalDeviceCoherentMemoryFeaturesAMD */
            if (isExtensionSupported(VK_AMD_DEVICE_COHERENT_MEMORY_EXTENSION_NAME))
            {
                features.deviceCoherentMemory = coherentMemoryFeatures.deviceCoherentMemory;
            }

            features.swapchainMode = static_cast<E_SWAPCHAIN_MODE>(E_SWAPCHAIN_MODE::ESM_NONE);
            if(isExtensionSupported(VK_KHR_SWAPCHAIN_EXTENSION_NAME))
                features.swapchainMode |= E_SWAPCHAIN_MODE::ESM_SURFACE;

            features.deferredHostOperations = isExtensionSupported(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);

            /*
                !! Enabled by Default, Exposed as Limits:
            */
            
            properties.limits.vertexPipelineStoresAndAtomics = deviceFeatures.features.vertexPipelineStoresAndAtomics;
            properties.limits.fragmentStoresAndAtomics = deviceFeatures.features.fragmentStoresAndAtomics;
            properties.limits.shaderTessellationAndGeometryPointSize = deviceFeatures.features.shaderTessellationAndGeometryPointSize;
            properties.limits.shaderImageGatherExtended = deviceFeatures.features.shaderImageGatherExtended;
            properties.limits.shaderFloat64 = deviceFeatures.features.shaderFloat64;
            properties.limits.shaderInt64 = deviceFeatures.features.shaderInt64;
            properties.limits.shaderInt16 = deviceFeatures.features.shaderInt16;
            
            properties.limits.storageBuffer16BitAccess = vulkan11Features.storageBuffer16BitAccess;
            properties.limits.uniformAndStorageBuffer16BitAccess = vulkan11Features.uniformAndStorageBuffer16BitAccess;
            properties.limits.storagePushConstant16 = vulkan11Features.storagePushConstant16;
            properties.limits.storageInputOutput16 = vulkan11Features.storageInputOutput16;
            properties.limits.variablePointers = vulkan11Features.variablePointers;

            properties.limits.vulkanMemoryModel = vulkan12Features.vulkanMemoryModel;
            properties.limits.vulkanMemoryModelDeviceScope = vulkan12Features.vulkanMemoryModelDeviceScope;
            properties.limits.vulkanMemoryModelAvailabilityVisibilityChains = vulkan12Features.vulkanMemoryModelAvailabilityVisibilityChains;
            properties.limits.shaderOutputViewportIndex = vulkan12Features.shaderOutputViewportIndex;
            properties.limits.shaderOutputLayer = vulkan12Features.shaderOutputLayer;
            
            if (isExtensionSupported(VK_INTEL_SHADER_INTEGER_FUNCTIONS_2_EXTENSION_NAME))
            {
                properties.limits.shaderIntegerFunctions2 = intelShaderIntegerFunctions2.shaderIntegerFunctions2;
            }

            if (isExtensionSupported(VK_KHR_SHADER_CLOCK_EXTENSION_NAME))
            {
                properties.limits.shaderSubgroupClock = shaderClockFeatures.shaderSubgroupClock;
            }
            
            if (isExtensionSupported(VK_NV_SHADER_IMAGE_FOOTPRINT_EXTENSION_NAME))
            {
                properties.limits.imageFootprint = shaderImageFootprintFeatures.imageFootprint;
            }

            if(isExtensionSupported(VK_EXT_TEXEL_BUFFER_ALIGNMENT_EXTENSION_NAME))
            {
                properties.limits.texelBufferAlignment = texelBufferAlignmentFeatures.texelBufferAlignment;
            }

            if(isExtensionSupported(VK_NV_SHADER_SM_BUILTINS_EXTENSION_NAME))
            {
                properties.limits.shaderSMBuiltins = shaderSMBuiltinsFeatures.shaderSMBuiltins;
            }

            properties.limits.shaderSubgroupPartitioned = isExtensionSupported(VK_NV_SHADER_SUBGROUP_PARTITIONED_EXTENSION_NAME);
            properties.limits.gcnShader = isExtensionSupported(VK_AMD_GCN_SHADER_EXTENSION_NAME);
            properties.limits.gpuShaderHalfFloat = isExtensionSupported(VK_AMD_GPU_SHADER_HALF_FLOAT_EXTENSION_NAME);
            properties.limits.shaderBallot = isExtensionSupported(VK_AMD_SHADER_BALLOT_EXTENSION_NAME);
            properties.limits.shaderImageLoadStoreLod = isExtensionSupported(VK_AMD_SHADER_IMAGE_LOAD_STORE_LOD_EXTENSION_NAME);
            properties.limits.shaderTrinaryMinmax = isExtensionSupported(VK_AMD_SHADER_TRINARY_MINMAX_EXTENSION_NAME);
            properties.limits.postDepthCoverage = isExtensionSupported(VK_EXT_POST_DEPTH_COVERAGE_EXTENSION_NAME);
            properties.limits.shaderStencilExport = isExtensionSupported(VK_EXT_SHADER_STENCIL_EXPORT_EXTENSION_NAME);
            properties.limits.decorateString = isExtensionSupported(VK_GOOGLE_DECORATE_STRING_EXTENSION_NAME);

            properties.limits.shaderNonSemanticInfo = isExtensionSupported(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);
            properties.limits.fragmentShaderBarycentric = isExtensionSupported(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME);
            properties.limits.geometryShaderPassthrough = isExtensionSupported(VK_NV_GEOMETRY_SHADER_PASSTHROUGH_EXTENSION_NAME);
            properties.limits.viewportSwizzle = isExtensionSupported(VK_NV_VIEWPORT_SWIZZLE_EXTENSION_NAME);
        }

        // Get physical device's memory properties
        {
            m_memoryProperties = SMemoryProperties();
            VkPhysicalDeviceMemoryProperties2 vk_physicalDeviceMemoryProperties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,nullptr};
            vkGetPhysicalDeviceMemoryProperties2(vk_physicalDevice, &vk_physicalDeviceMemoryProperties);
            m_memoryProperties.memoryTypeCount = vk_physicalDeviceMemoryProperties.memoryProperties.memoryTypeCount;
            for(uint32_t i = 0; i < m_memoryProperties.memoryTypeCount; ++i)
            {
                m_memoryProperties.memoryTypes[i].heapIndex = vk_physicalDeviceMemoryProperties.memoryProperties.memoryTypes[i].heapIndex;
                m_memoryProperties.memoryTypes[i].propertyFlags = getMemoryPropertyFlagsFromVkMemoryPropertyFlags(vk_physicalDeviceMemoryProperties.memoryProperties.memoryTypes[i].propertyFlags);
            }
            m_memoryProperties.memoryHeapCount = vk_physicalDeviceMemoryProperties.memoryProperties.memoryHeapCount;
            for(uint32_t i = 0; i < m_memoryProperties.memoryHeapCount; ++i)
            {
                m_memoryProperties.memoryHeaps[i].size = vk_physicalDeviceMemoryProperties.memoryProperties.memoryHeaps[i].size;
                m_memoryProperties.memoryHeaps[i].flags = static_cast<IDeviceMemoryAllocation::E_MEMORY_HEAP_FLAGS>(vk_physicalDeviceMemoryProperties.memoryProperties.memoryHeaps[i].flags);
            }
        }
                
        uint32_t qfamCount = 0u;
        vkGetPhysicalDeviceQueueFamilyProperties2(m_vkPhysicalDevice, &qfamCount, nullptr);
        core::vector<VkQueueFamilyProperties2> qfamprops(qfamCount,{VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2,nullptr});
        vkGetPhysicalDeviceQueueFamilyProperties2(m_vkPhysicalDevice, &qfamCount, qfamprops.data());

        m_qfamProperties = core::make_refctd_dynamic_array<qfam_props_array_t>(qfamCount);
        for (uint32_t i = 0u; i < qfamCount; ++i)
        {
            const auto& vkqf = qfamprops[i].queueFamilyProperties;
            auto& qf = (*m_qfamProperties)[i];
                    
            qf.queueCount = vkqf.queueCount;
            qf.queueFlags = static_cast<IQueue::FAMILY_FLAGS>(vkqf.queueFlags);
            qf.timestampValidBits = vkqf.timestampValidBits;
            qf.minImageTransferGranularity = { vkqf.minImageTransferGranularity.width, vkqf.minImageTransferGranularity.height, vkqf.minImageTransferGranularity.depth };
        }

        // Set Format Usages
        for(uint32_t i = 0; i < asset::EF_COUNT; ++i)
        {
            const asset::E_FORMAT format = static_cast<asset::E_FORMAT>(i);
            bool skip = false;
            switch (format)
            {
                case asset::EF_B4G4R4A4_UNORM_PACK16:
                case asset::EF_R4G4B4A4_UNORM_PACK16:
                    if (vk_deviceProperties.apiVersion<VK_MAKE_API_VERSION(0,1,3,0) && !isExtensionSupported(VK_EXT_4444_FORMATS_EXTENSION_NAME))
                        skip = true;
                    break;
                // TODO: ASTC HDR stuff
                //case asset::EF_ASTC__SFLOAT
                    //if (vk_deviceProperties.apiVersion<VK_MAKE_API_VERSION(0,1,3,0) && !isExtensionSupported(VK_EXT_TEXTURE_COMPRESSION_ASTC_HDR_EXTENSION_NAME))
                    //    skip = true;
                    //break;
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

            VkFormatProperties2 vk_formatProps = {VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2,nullptr};
            vkGetPhysicalDeviceFormatProperties2(m_vkPhysicalDevice, getVkFormatFromFormat(format), &vk_formatProps);

            // TODO: Upgrade to `VkFormatFeatureFlags2` when `VK_KHR_format_feature_flags2` support is ubiquitous
            const VkFormatFeatureFlags linearTilingFeatures = vk_formatProps.formatProperties.linearTilingFeatures;
            const VkFormatFeatureFlags optimalTilingFeatures = vk_formatProps.formatProperties.optimalTilingFeatures;
            const VkFormatFeatureFlags bufferFeatures = vk_formatProps.formatProperties.bufferFeatures;

            m_linearTilingUsages[format] = {};
            m_linearTilingUsages[format].sampledImage = (linearTilingFeatures & (VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) ? 1 : 0;
            m_linearTilingUsages[format].storageImage = (linearTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) ? 1 : 0;
            m_linearTilingUsages[format].storageImageAtomic = (linearTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT) ? 1 : 0;
            m_linearTilingUsages[format].attachment = (linearTilingFeatures & (VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) ? 1 : 0;
            m_linearTilingUsages[format].attachmentBlend = (linearTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT) ? 1 : 0;
            m_linearTilingUsages[format].blitSrc = (linearTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) ? 1 : 0;
            m_linearTilingUsages[format].blitDst = (linearTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT) ? 1 : 0;
            m_linearTilingUsages[format].transferSrc = (linearTilingFeatures & VK_FORMAT_FEATURE_TRANSFER_SRC_BIT) ? 1 : 0;
            m_linearTilingUsages[format].transferDst = (linearTilingFeatures & VK_FORMAT_FEATURE_TRANSFER_DST_BIT) ? 1 : 0;
            // m_linearTilingUsages[format].log2MaxSmples = ; // Todo(Erfan)
            
            m_optimalTilingUsages[format] = {};
            m_optimalTilingUsages[format].sampledImage = optimalTilingFeatures & (VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) ? 1 : 0;
            m_optimalTilingUsages[format].storageImage = optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT ? 1 : 0;
            m_optimalTilingUsages[format].storageImageAtomic = optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT ? 1 : 0;
            m_optimalTilingUsages[format].attachment = optimalTilingFeatures & (VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) ? 1 : 0;
            m_optimalTilingUsages[format].attachmentBlend = optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT ? 1 : 0;
            m_optimalTilingUsages[format].blitSrc = optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT ? 1 : 0;
            m_optimalTilingUsages[format].blitDst = optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT ? 1 : 0;
            m_optimalTilingUsages[format].transferSrc = optimalTilingFeatures & VK_FORMAT_FEATURE_TRANSFER_SRC_BIT ? 1 : 0;
            m_optimalTilingUsages[format].transferDst = optimalTilingFeatures & VK_FORMAT_FEATURE_TRANSFER_DST_BIT ? 1 : 0;
            // m_optimalTilingUsages[format].log2MaxSmples = ; // Todo(Erfan)
            
            m_bufferUsages[format] = {};
            m_bufferUsages[format].vertexAttribute = (bufferFeatures & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT) ? 1 : 0;
            m_bufferUsages[format].bufferView = (bufferFeatures & VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT) ? 1 : 0;
            m_bufferUsages[format].storageBufferView = (bufferFeatures & VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT) ? 1 : 0;
            m_bufferUsages[format].storageBufferViewAtomic = (bufferFeatures & VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_ATOMIC_BIT) ? 1 : 0;
            m_bufferUsages[format].accelerationStructureVertex = (bufferFeatures & VK_FORMAT_FEATURE_ACCELERATION_STRUCTURE_VERTEX_BUFFER_BIT_KHR) ? 1 : 0;
        }
#endif        
    return std::unique_ptr<CVulkanPhysicalDevice>(new CVulkanPhysicalDevice(core::smart_refctd_ptr<system::ISystem>(sys),api,rdoc,vk_physicalDevice));
}


core::smart_refctd_ptr<ILogicalDevice> CVulkanPhysicalDevice::createLogicalDevice_impl(ILogicalDevice::SCreationParams&& params)
{
    // We might alter it to account for dependancies.
    resolveFeatureDependencies(params.featuresToEnable);
    SFeatures& enabledFeatures = params.featuresToEnable;

    core::unordered_set<core::string> extensionsToEnable;
    // Extensions we REQUIRE
    extensionsToEnable.insert(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
    VkPhysicalDeviceSynchronization2FeaturesKHR synchronization2Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR, nullptr };
    VkBaseInStructure* featuresTail = reinterpret_cast<VkBaseInStructure*>(&synchronization2Features);

#if 0
            //
            VkPhysicalDeviceVulkan12Features vulkan12Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &synchronization2Features };
            VkPhysicalDeviceVulkan11Features vulkan11Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &vulkan12Features };
            VkPhysicalDeviceFeatures2 vk_deviceFeatures2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &vulkan11Features };

            // Important notes on extension dependancies, both instance and device
            /*
                If an extension is supported (as queried by vkEnumerateInstanceExtensionProperties or vkEnumerateDeviceExtensionProperties), 
                then required extensions of that extension must also be supported for the same instance or physical device.

                Any device extension that has an instance extension dependency that is not enabled by vkCreateInstance is considered to be unsupported,
                hence it must not be returned by vkEnumerateDeviceExtensionProperties for any VkPhysicalDevice child of the instance. Instance extensions do not have dependencies on device extensions.

                Conclusion: We don't need to specifically check instance extension dependancies but we can do it through apiConnection->getEnableFeatures to hint the user on what might be wrong 
            */


            // Extensions promoted to 1.3 core
            VkPhysicalDeviceImageRobustnessFeaturesEXT                  imageRobustnessFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES_EXT, nullptr };
            VkPhysicalDevicePipelineCreationCacheControlFeaturesEXT     pipelineCreationCacheControlFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_CREATION_CACHE_CONTROL_FEATURES_EXT, nullptr };
            VkPhysicalDeviceShaderDemoteToHelperInvocationFeaturesEXT   shaderDemoteToHelperInvocationFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES_EXT, nullptr };
            VkPhysicalDeviceShaderTerminateInvocationFeaturesKHR        shaderTerminateInvocationFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_TERMINATE_INVOCATION_FEATURES_KHR, nullptr };
            VkPhysicalDeviceSubgroupSizeControlFeaturesEXT              subgroupSizeControlFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT, nullptr };
            VkPhysicalDeviceTextureCompressionASTCHDRFeaturesEXT        textureCompressionASTCHDRFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXTURE_COMPRESSION_ASTC_HDR_FEATURES_EXT, nullptr };

            // Real Extensions
            VkPhysicalDeviceTexelBufferAlignmentFeaturesEXT                 texelBufferAlignmentFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXEL_BUFFER_ALIGNMENT_FEATURES_EXT, nullptr };
            VkPhysicalDeviceColorWriteEnableFeaturesEXT                     colorWriteEnableFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COLOR_WRITE_ENABLE_FEATURES_EXT, nullptr };
            VkPhysicalDeviceConditionalRenderingFeaturesEXT                 conditionalRenderingFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT, nullptr };
            VkPhysicalDeviceDeviceMemoryReportFeaturesEXT                   deviceMemoryReportFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_MEMORY_REPORT_FEATURES_EXT, nullptr };
            VkPhysicalDeviceFragmentDensityMapFeaturesEXT                   fragmentDensityMapFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_FEATURES_EXT, nullptr };
            VkPhysicalDeviceFragmentDensityMap2FeaturesEXT                  fragmentDensityMap2Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_2_FEATURES_EXT, nullptr };
            VkPhysicalDeviceLineRasterizationFeaturesEXT                    lineRasterizationFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_FEATURES_EXT, nullptr };
            VkPhysicalDeviceMemoryPriorityFeaturesEXT                       memoryPriorityFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT, nullptr };
            VkPhysicalDeviceRobustness2FeaturesEXT                          robustness2Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT, nullptr };
            VkPhysicalDevicePerformanceQueryFeaturesKHR                     performanceQueryFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PERFORMANCE_QUERY_FEATURES_KHR, nullptr };
            VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR         pipelineExecutablePropertiesFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR, nullptr };
            VkPhysicalDeviceMaintenance4Features                            maintenance4Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES, nullptr };
            VkPhysicalDeviceCoherentMemoryFeaturesAMD                       coherentMemoryFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COHERENT_MEMORY_FEATURES_AMD, nullptr };
            VkPhysicalDeviceGlobalPriorityQueryFeaturesKHR                  globalPriorityFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GLOBAL_PRIORITY_QUERY_FEATURES_KHR, nullptr };
            VkPhysicalDeviceCoverageReductionModeFeaturesNV                 coverageReductionModeFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COVERAGE_REDUCTION_MODE_FEATURES_NV, nullptr };
            VkPhysicalDeviceDeviceGeneratedCommandsFeaturesNV               deviceGeneratedCommandsFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_FEATURES_NV, nullptr };
            VkPhysicalDeviceMeshShaderFeaturesNV                            meshShaderFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_NV, nullptr };
            VkPhysicalDeviceRepresentativeFragmentTestFeaturesNV            representativeFragmentTestFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_REPRESENTATIVE_FRAGMENT_TEST_FEATURES_NV, nullptr };
            VkPhysicalDeviceShaderImageFootprintFeaturesNV                  shaderImageFootprintFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_FOOTPRINT_FEATURES_NV, nullptr };
            VkPhysicalDeviceComputeShaderDerivativesFeaturesNV              computeShaderDerivativesFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_NV, nullptr };
            VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR        workgroupMemoryExplicitLayout = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR, nullptr };
            VkPhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR     subgroupUniformControlFlowFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_FEATURES_KHR, nullptr };
            VkPhysicalDeviceShaderClockFeaturesKHR                          shaderClockFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR, nullptr };
            VkPhysicalDeviceShaderIntegerFunctions2FeaturesINTEL            intelShaderIntegerFunctions2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_FUNCTIONS_2_FEATURES_INTEL, nullptr };
            VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT               shaderImageAtomicInt64Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES_EXT, nullptr };
            VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT                   shaderAtomicFloat2Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES_EXT, nullptr };
            VkPhysicalDeviceShaderAtomicFloatFeaturesEXT                    shaderAtomicFloatFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT, nullptr };
            VkPhysicalDeviceIndexTypeUint8FeaturesEXT                       indexTypeUint8Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_EXT, nullptr };
            VkPhysicalDeviceRasterizationOrderAttachmentAccessFeaturesARM   rasterizationOrderAttachmentAccessFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_FEATURES_ARM, nullptr };
            VkPhysicalDeviceShaderIntegerDotProductFeatures                 shaderIntegerDotProductFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES, nullptr };
            VkPhysicalDeviceRayTracingMotionBlurFeaturesNV                  rayTracingMotionBlurFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_MOTION_BLUR_FEATURES_NV, nullptr };
            VkPhysicalDeviceASTCDecodeFeaturesEXT                           astcDecodeFeaturesEXT = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ASTC_DECODE_FEATURES_EXT, nullptr };
            VkPhysicalDeviceShaderSMBuiltinsFeaturesNV                      shaderSMBuiltinsFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_FEATURES_NV, nullptr };
            VkPhysicalDeviceCooperativeMatrixFeaturesNV                     cooperativeMatrixFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_NV, nullptr };
            VkPhysicalDeviceRayTracingPipelineFeaturesKHR                   rayTracingPipelineFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR, nullptr };
            VkPhysicalDeviceAccelerationStructureFeaturesKHR                accelerationStructureFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR, nullptr };
            VkPhysicalDeviceRayQueryFeaturesKHR                             rayQueryFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR, nullptr };
            VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT              fragmentShaderInterlockFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT, nullptr };
        

            auto insertExtensionIfAvailable = [&](const char* extName) -> bool
            {
                if(isExtensionSupported(extName))
                {
                    extensionsToEnable.insert(extName);
                    return true;
                }
                else
                    return false;
            };

            auto uncondAddFeatureToChain = [&featuresTail](void* feature) -> void
            {
                VkBaseInStructure* toAdd = reinterpret_cast<VkBaseInStructure*>(feature);
                featuresTail->pNext = toAdd;
                featuresTail = toAdd;
            };

            // prime ourselves with good defaults
            {
                // special handling of texture compression/format extensions 
                if (insertExtensionIfAvailable(VK_EXT_TEXTURE_COMPRESSION_ASTC_HDR_EXTENSION_NAME))
                    uncondAddFeatureToChain(&textureCompressionASTCHDRFeatures);
                if (insertExtensionIfAvailable(VK_EXT_ASTC_DECODE_MODE_EXTENSION_NAME))
                    uncondAddFeatureToChain(&astcDecodeFeaturesEXT);

                // we actually re-query all available Vulkan <= MinimumApiVersion features so that by default they're all enabled unless we explicitly disable
                vkGetPhysicalDeviceFeatures2(m_vkPhysicalDevice,&vk_deviceFeatures2);
            }

            // Vulkan has problems with having features in the feature chain that have all values set to false.
            // For example having an empty "RayTracingPipelineFeaturesKHR" in the chain will lead to validation errors for RayQueryONLY applications.
            auto addFeatureToChain = [&featuresTail,uncondAddFeatureToChain](void* feature) -> void
            {
                VkBaseInStructure* toAdd = reinterpret_cast<VkBaseInStructure*>(feature);
                // For protecting against duplication of feature structures that may be requested to add to chain twice due to extension requirements
                if (toAdd->pNext==nullptr && toAdd!=featuresTail);
                    uncondAddFeatureToChain(toAdd);
            };

            // A. Enable by Default, exposed as limits : add names to string and structs to feature chain
            {
                if (insertExtensionIfAvailable(VK_INTEL_SHADER_INTEGER_FUNCTIONS_2_EXTENSION_NAME))
                {
                    // All Requirements Exist in Vulkan 1.1
                    intelShaderIntegerFunctions2.shaderIntegerFunctions2 = properties.limits.shaderIntegerFunctions2;
                    addFeatureToChain(&intelShaderIntegerFunctions2);
                }

                if (insertExtensionIfAvailable(VK_KHR_SHADER_CLOCK_EXTENSION_NAME))
                {
                    // All Requirements Exist in Vulkan 1.1
                    shaderClockFeatures.shaderSubgroupClock = properties.limits.shaderSubgroupClock;
                    addFeatureToChain(&shaderClockFeatures);
                }
            
                if (insertExtensionIfAvailable(VK_NV_SHADER_IMAGE_FOOTPRINT_EXTENSION_NAME))
                {
                    // All Requirements Exist in Vulkan 1.1
                    shaderImageFootprintFeatures.imageFootprint = properties.limits.imageFootprint;
                    addFeatureToChain(&shaderImageFootprintFeatures);
                }

                if(insertExtensionIfAvailable(VK_EXT_TEXEL_BUFFER_ALIGNMENT_EXTENSION_NAME))
                {
                    // All Requirements Exist in Vulkan 1.1
                    texelBufferAlignmentFeatures.texelBufferAlignment = properties.limits.texelBufferAlignment;
                    addFeatureToChain(&texelBufferAlignmentFeatures);
                }

                if(insertExtensionIfAvailable(VK_NV_SHADER_SM_BUILTINS_EXTENSION_NAME))
                {
                    // No Extension Requirements
                    shaderSMBuiltinsFeatures.shaderSMBuiltins = properties.limits.shaderSMBuiltins;
                    addFeatureToChain(&shaderSMBuiltinsFeatures);
                }

                if (insertExtensionIfAvailable(VK_KHR_MAINTENANCE_4_EXTENSION_NAME))
                {
                    // No Extension Requirements
                    maintenance4Features.maintenance4 = properties.limits.workgroupSizeFromSpecConstant;
                    addFeatureToChain(&maintenance4Features);
                }

                insertExtensionIfAvailable(VK_NV_SHADER_SUBGROUP_PARTITIONED_EXTENSION_NAME); // No Extension Requirements
                insertExtensionIfAvailable(VK_AMD_GCN_SHADER_EXTENSION_NAME); // No Extension Requirements
                insertExtensionIfAvailable(VK_AMD_GPU_SHADER_HALF_FLOAT_EXTENSION_NAME); // No Extension Requirements
                insertExtensionIfAvailable(VK_AMD_SHADER_BALLOT_EXTENSION_NAME); // No Extension Requirements
                insertExtensionIfAvailable(VK_AMD_SHADER_IMAGE_LOAD_STORE_LOD_EXTENSION_NAME); // No Extension Requirements
                insertExtensionIfAvailable(VK_AMD_SHADER_TRINARY_MINMAX_EXTENSION_NAME); // No Extension Requirements
                insertExtensionIfAvailable(VK_EXT_POST_DEPTH_COVERAGE_EXTENSION_NAME); // No Extension Requirements
                insertExtensionIfAvailable(VK_EXT_SHADER_STENCIL_EXPORT_EXTENSION_NAME); // No Extension Requirements
                insertExtensionIfAvailable(VK_GOOGLE_DECORATE_STRING_EXTENSION_NAME); // No Extension Requirements

                insertExtensionIfAvailable(VK_KHR_EXTERNAL_FENCE_FD_EXTENSION_NAME); 
    #ifdef _NBL_WINDOWS_API_
                insertExtensionIfAvailable(VK_KHR_EXTERNAL_FENCE_WIN32_EXTENSION_NAME); // All Requirements Exist in Vulkan 1.1 (including instance extensions)
    #endif
                insertExtensionIfAvailable(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME); 
    #ifdef _NBL_WINDOWS_API_
                insertExtensionIfAvailable(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME); // All Requirements Exist in Vulkan 1.1 (including instance extensions)
    #endif
                insertExtensionIfAvailable(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME); 
    #ifdef _NBL_WINDOWS_API_
                insertExtensionIfAvailable(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME); // All Requirements Exist in Vulkan 1.1 (including instance extensions)
    #endif

                insertExtensionIfAvailable(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME); // No Extension Requirements
                insertExtensionIfAvailable(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME); // No Extension Requirements
                insertExtensionIfAvailable(VK_NV_GEOMETRY_SHADER_PASSTHROUGH_EXTENSION_NAME); // No Extension Requirements
                insertExtensionIfAvailable(VK_NV_VIEWPORT_SWIZZLE_EXTENSION_NAME); // No Extension Requirements
            }

            // B. FeaturesToEnable: add names to strings and structs to feature chain
            
        /* Vulkan 1.0 Core  */
        vk_deviceFeatures2.features.robustBufferAccess = enabledFeatures.robustBufferAccess;
        vk_deviceFeatures2.features.fullDrawIndexUint32 = true; // ROADMAP 2022
        vk_deviceFeatures2.features.imageCubeArray = true; // ROADMAP 2022
        vk_deviceFeatures2.features.independentBlend = true; // ROADMAP 2022
        vk_deviceFeatures2.features.geometryShader = enabledFeatures.geometryShader;
        vk_deviceFeatures2.features.tessellationShader = enabledFeatures.tessellationShader;
        vk_deviceFeatures2.features.sampleRateShading = true; // ROADMAP 2022
        vk_deviceFeatures2.features.dualSrcBlend = enabledFeatures.dualSrcBlend;
        vk_deviceFeatures2.features.logicOp = enabledFeatures.logicOp;
        vk_deviceFeatures2.features.multiDrawIndirect = true; // ROADMAP 2022
        vk_deviceFeatures2.features.drawIndirectFirstInstance = true; // ROADMAP 2022
        vk_deviceFeatures2.features.depthClamp = true; // ROADMAP 2022
        vk_deviceFeatures2.features.depthBiasClamp = true; // ROADMAP 2022
        vk_deviceFeatures2.features.fillModeNonSolid = enabledFeatures.fillModeNonSolid;
        vk_deviceFeatures2.features.depthBounds = enabledFeatures.depthBounds;
        vk_deviceFeatures2.features.wideLines = enabledFeatures.wideLines;
        vk_deviceFeatures2.features.largePoints = enabledFeatures.largePoints;
        vk_deviceFeatures2.features.alphaToOne = enabledFeatures.alphaToOne;
        vk_deviceFeatures2.features.multiViewport = enabledFeatures.multiViewport;
        vk_deviceFeatures2.features.samplerAnisotropy = true; // ROADMAP
        // leave defaulted
        //vk_deviceFeatures2.features.textureCompressionETC2;
        //vk_deviceFeatures2.features.textureCompressionASTC_LDR;
        //vk_deviceFeatures2.features.textureCompressionBC;
        vk_deviceFeatures2.features.occlusionQueryPrecise = true; // ROADMAP 2022
        vk_deviceFeatures2.features.pipelineStatisticsQuery = enabledFeatures.pipelineStatisticsQuery;
        vk_deviceFeatures2.features.vertexPipelineStoresAndAtomics = properties.limits.vertexPipelineStoresAndAtomics;
        vk_deviceFeatures2.features.fragmentStoresAndAtomics = properties.limits.fragmentStoresAndAtomics;
        vk_deviceFeatures2.features.shaderTessellationAndGeometryPointSize = properties.limits.shaderTessellationAndGeometryPointSize;
        vk_deviceFeatures2.features.shaderImageGatherExtended = properties.limits.shaderImageGatherExtended;
        vk_deviceFeatures2.features.shaderStorageImageExtendedFormats = true; // ROADMAP 2022
        vk_deviceFeatures2.features.shaderStorageImageMultisample = properties.limits.shaderStorageImageMultisample;
        vk_deviceFeatures2.features.shaderStorageImageReadWithoutFormat = enabledFeatures.shaderStorageImageReadWithoutFormat;
        vk_deviceFeatures2.features.shaderStorageImageWriteWithoutFormat = enabledFeatures.shaderStorageImageWriteWithoutFormat;
        vk_deviceFeatures2.features.shaderUniformBufferArrayDynamicIndexing = true; // ROADMAP 2022
        vk_deviceFeatures2.features.shaderSampledImageArrayDynamicIndexing = true; // ROADMAP 2022
        vk_deviceFeatures2.features.shaderStorageBufferArrayDynamicIndexing = true; // ROADMAP 2022
        vk_deviceFeatures2.features.shaderStorageImageArrayDynamicIndexing = properties.limits.shaderStorageImageArrayDynamicIndexing;
        vk_deviceFeatures2.features.shaderClipDistance = enabledFeatures.shaderClipDistance;
        vk_deviceFeatures2.features.shaderCullDistance = enabledFeatures.shaderCullDistance;
        vk_deviceFeatures2.features.shaderInt64 = properties.limits.shaderInt64;
        vk_deviceFeatures2.features.shaderInt16 = properties.limits.shaderInt16;
        vk_deviceFeatures2.features.shaderFloat64 = properties.limits.shaderFloat64;
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
        vk_deviceFeatures2.features.variableMultisampleRate = enabledFeatures.variableMultisampleRate;
        vk_deviceFeatures2.features.inheritedQueries = enabledFeatures.inheritedQueries;

        /* Vulkan 1.1 Core */
        vulkan11Features.storageBuffer16BitAccess = properties.limits.storageBuffer16BitAccess;
        vulkan11Features.uniformAndStorageBuffer16BitAccess = properties.limits.uniformAndStorageBuffer16BitAccess;
        vulkan11Features.storagePushConstant16 = properties.limits.storagePushConstant16;
        vulkan11Features.storageInputOutput16 = properties.limits.storageInputOutput16;
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#features-requirements
        vulkan11Features.multiview = true;
        vulkan11Features.multiviewGeometryShader = false;// = TODO;
        vulkan11Features.multiviewTessellationShader = false;// = TODO;
        vulkan11Features.variablePointers = properties.limits.variablePointers;
        vulkan11Features.variablePointersStorageBuffer = vulkan11Features.variablePointers;
        // not yet
        vulkan11Features.protectedMemory = false;
        vulkan11Features.samplerYcbcrConversion = false;
        vulkan11Features.shaderDrawParameters = enabledFeatures.shaderDrawParameters;
            
        /* Vulkan 1.2 Core */
        vulkan12Features.samplerMirrorClampToEdge = true; // ubiquitous
        vulkan12Features.drawIndirectCount = properties.limits.drawIndirectCount;
        vulkan12Features.storageBuffer8BitAccess = properties.limits.storageBuffer8BitAccess;
        vulkan12Features.uniformAndStorageBuffer8BitAccess = properties.limits.uniformAndStorageBuffer8BitAccess;
        vulkan12Features.storagePushConstant8 = properties.limits.storagePushConstant8;
        vulkan12Features.shaderBufferInt64Atomics = properties.limits.shaderBufferInt64Atomics;
        vulkan12Features.shaderSharedInt64Atomics = properties.limits.shaderSharedInt64Atomics;
        vulkan12Features.shaderFloat16 = properties.limits.shaderFloat16;
        vulkan12Features.shaderInt8 = properties.limits.shaderInt8;
        vulkan12Features.descriptorIndexing = enabledFeatures.descriptorIndexing;
        vulkan12Features.shaderInputAttachmentArrayDynamicIndexing = enabledFeatures.shaderInputAttachmentArrayDynamicIndexing;
        vulkan12Features.shaderUniformTexelBufferArrayDynamicIndexing = enabledFeatures.shaderUniformTexelBufferArrayDynamicIndexing;
        vulkan12Features.shaderStorageTexelBufferArrayDynamicIndexing = enabledFeatures.shaderStorageTexelBufferArrayDynamicIndexing;
        vulkan12Features.shaderUniformBufferArrayNonUniformIndexing = enabledFeatures.shaderUniformBufferArrayNonUniformIndexing;
        vulkan12Features.shaderSampledImageArrayNonUniformIndexing = enabledFeatures.shaderSampledImageArrayNonUniformIndexing;
        vulkan12Features.shaderStorageBufferArrayNonUniformIndexing = enabledFeatures.shaderStorageBufferArrayNonUniformIndexing;
        vulkan12Features.shaderStorageImageArrayNonUniformIndexing = enabledFeatures.shaderStorageImageArrayNonUniformIndexing;
        vulkan12Features.shaderInputAttachmentArrayNonUniformIndexing = enabledFeatures.shaderInputAttachmentArrayNonUniformIndexing;
        vulkan12Features.shaderUniformTexelBufferArrayNonUniformIndexing = enabledFeatures.shaderUniformTexelBufferArrayNonUniformIndexing;
        vulkan12Features.shaderStorageTexelBufferArrayNonUniformIndexing = enabledFeatures.shaderStorageTexelBufferArrayNonUniformIndexing;
        vulkan12Features.descriptorBindingUniformBufferUpdateAfterBind = enabledFeatures.descriptorBindingUniformBufferUpdateAfterBind;
        vulkan12Features.descriptorBindingSampledImageUpdateAfterBind = enabledFeatures.descriptorBindingSampledImageUpdateAfterBind;
        vulkan12Features.descriptorBindingStorageImageUpdateAfterBind = enabledFeatures.descriptorBindingStorageImageUpdateAfterBind;
        vulkan12Features.descriptorBindingStorageBufferUpdateAfterBind = enabledFeatures.descriptorBindingStorageBufferUpdateAfterBind;
        vulkan12Features.descriptorBindingUniformTexelBufferUpdateAfterBind = enabledFeatures.descriptorBindingUniformTexelBufferUpdateAfterBind;
        vulkan12Features.descriptorBindingStorageTexelBufferUpdateAfterBind = enabledFeatures.descriptorBindingStorageTexelBufferUpdateAfterBind;
        vulkan12Features.descriptorBindingUpdateUnusedWhilePending = enabledFeatures.descriptorBindingUpdateUnusedWhilePending;
        vulkan12Features.descriptorBindingPartiallyBound = enabledFeatures.descriptorBindingPartiallyBound;
        vulkan12Features.descriptorBindingVariableDescriptorCount = enabledFeatures.descriptorBindingVariableDescriptorCount;
        vulkan12Features.runtimeDescriptorArray = enabledFeatures.runtimeDescriptorArray;
        vulkan12Features.samplerFilterMinmax = enabledFeatures.samplerFilterMinmax;
        vulkan12Features.scalarBlockLayout = true; // ROADMAP 2022
        vulkan12Features.imagelessFramebuffer = false; // decided against
        vulkan12Features.uniformBufferStandardLayout = true; // required anyway
        vulkan12Features.shaderSubgroupExtendedTypes = true; // required anyway
        vulkan12Features.separateDepthStencilLayouts = true; // required anyway
        vulkan12Features.hostQueryReset = true; // required anyway
        vulkan12Features.timelineSemaphore = true; // required anyway
        vulkan12Features.bufferDeviceAddress = enabledFeatures.bufferDeviceAddress || enabledFeatures.bufferDeviceAddressMultiDevice;
        // Some capture tools need this but can't enable this when you set this to false (they're buggy probably, We shouldn't worry about this)
        vulkan12Features.bufferDeviceAddressCaptureReplay = bool(vulkan12Features.bufferDeviceAddress) && (m_rdoc_api!=nullptr);
        vulkan12Features.bufferDeviceAddressMultiDevice = enabledFeatures.bufferDeviceAddressMultiDevice;
        vulkan12Features.vulkanMemoryModel = properties.limits.vulkanMemoryModel;
        vulkan12Features.vulkanMemoryModelDeviceScope = properties.limits.vulkanMemoryModelDeviceScope;
        vulkan12Features.vulkanMemoryModelAvailabilityVisibilityChains = properties.limits.vulkanMemoryModelAvailabilityVisibilityChains;
        vulkan12Features.shaderOutputViewportIndex = properties.limits.shaderOutputViewportIndex;
        vulkan12Features.shaderOutputLayer = properties.limits.shaderOutputLayer;
        vulkan12Features.subgroupBroadcastDynamicId = true; // ubiquitous

            
#define CHECK_VULKAN_EXTENTION_FOR_SINGLE_VAR_FEATURE(VAR_NAME, EXT_NAME, FEATURE_STRUCT)           \
        if(enabledFeatures.VAR_NAME)                                                    \
        {                                                                               \
            insertExtensionIfAvailable(EXT_NAME);                                       \
            FEATURE_STRUCT.VAR_NAME = enabledFeatures.VAR_NAME;                         \
            addFeatureToChain(&FEATURE_STRUCT);                                         \
        }

        /* Vulkan 1.3 Core */
// TODO: robustImageAccess
// TODO: pipelineCreationCacheControl
        CHECK_VULKAN_EXTENTION_FOR_SINGLE_VAR_FEATURE(shaderDemoteToHelperInvocation, VK_EXT_SHADER_DEMOTE_TO_HELPER_INVOCATION_EXTENSION_NAME, shaderDemoteToHelperInvocationFeatures);
        CHECK_VULKAN_EXTENTION_FOR_SINGLE_VAR_FEATURE(shaderTerminateInvocation, VK_KHR_SHADER_TERMINATE_INVOCATION_EXTENSION_NAME, shaderTerminateInvocationFeatures);         
        // Instead of checking and enabling individual features like below, I can do awesome things like this:
        /*
            CHECK_VULKAN_EXTENTION(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME, subgroupSizeControlFeatures,
                    subgroupSizeControl,
                    computeFullSubgroups);
        */
        // But I would need to enable /Zc:preprocessor in compiler So I could use __VA_OPT__ :D
        if (enabledFeatures.subgroupSizeControl ||
            enabledFeatures.computeFullSubgroups)
        {
            // All Requirements Exist in Vulkan 1.1
            insertExtensionIfAvailable(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
            subgroupSizeControlFeatures.subgroupSizeControl = enabledFeatures.subgroupSizeControl;
            subgroupSizeControlFeatures.computeFullSubgroups = enabledFeatures.computeFullSubgroups;
            addFeatureToChain(&subgroupSizeControlFeatures);
        }
        //leave defaulted
        //textureCompressionASTCHDRFeatures;
// TODO: shaderZeroInitializeWorkgroupMemory
            
        CHECK_VULKAN_EXTENTION_FOR_SINGLE_VAR_FEATURE(shaderIntegerDotProduct, VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME, shaderIntegerDotProductFeatures);
            
        if (enabledFeatures.rasterizationOrderColorAttachmentAccess ||
            enabledFeatures.rasterizationOrderDepthAttachmentAccess ||
            enabledFeatures.rasterizationOrderStencilAttachmentAccess)
        {
            // All Requirements Exist in Vulkan 1.1
            insertExtensionIfAvailable(VK_ARM_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_EXTENSION_NAME);
            rasterizationOrderAttachmentAccessFeatures.rasterizationOrderColorAttachmentAccess = enabledFeatures.rasterizationOrderColorAttachmentAccess;
            rasterizationOrderAttachmentAccessFeatures.rasterizationOrderDepthAttachmentAccess = enabledFeatures.rasterizationOrderDepthAttachmentAccess;
            rasterizationOrderAttachmentAccessFeatures.rasterizationOrderStencilAttachmentAccess = enabledFeatures.rasterizationOrderStencilAttachmentAccess;
            addFeatureToChain(&rasterizationOrderAttachmentAccessFeatures);
        }

        if (enabledFeatures.fragmentShaderSampleInterlock ||
            enabledFeatures.fragmentShaderPixelInterlock ||
            enabledFeatures.fragmentShaderShadingRateInterlock)
        {
            // All Requirements Exist in Vulkan 1.1
            insertExtensionIfAvailable(VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME);
            fragmentShaderInterlockFeatures.fragmentShaderSampleInterlock = enabledFeatures.fragmentShaderSampleInterlock;
            fragmentShaderInterlockFeatures.fragmentShaderPixelInterlock = enabledFeatures.fragmentShaderPixelInterlock;
            fragmentShaderInterlockFeatures.fragmentShaderShadingRateInterlock = enabledFeatures.fragmentShaderShadingRateInterlock;
            addFeatureToChain(&fragmentShaderInterlockFeatures);
        }
            
        CHECK_VULKAN_EXTENTION_FOR_SINGLE_VAR_FEATURE(indexTypeUint8, VK_EXT_INDEX_TYPE_UINT8_EXTENSION_NAME, indexTypeUint8Features);

        if (enabledFeatures.shaderBufferFloat32Atomics ||
            enabledFeatures.shaderBufferFloat32AtomicAdd ||
            enabledFeatures.shaderBufferFloat64Atomics ||
            enabledFeatures.shaderBufferFloat64AtomicAdd ||
            enabledFeatures.shaderSharedFloat32Atomics ||
            enabledFeatures.shaderSharedFloat32AtomicAdd ||
            enabledFeatures.shaderSharedFloat64Atomics ||
            enabledFeatures.shaderSharedFloat64AtomicAdd ||
            enabledFeatures.shaderImageFloat32Atomics ||
            enabledFeatures.shaderImageFloat32AtomicAdd ||
            enabledFeatures.sparseImageFloat32Atomics ||
            enabledFeatures.sparseImageFloat32AtomicAdd)
        {
            insertExtensionIfAvailable(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
            shaderAtomicFloatFeatures.shaderBufferFloat32Atomics   = enabledFeatures.shaderBufferFloat32Atomics;
            shaderAtomicFloatFeatures.shaderBufferFloat32AtomicAdd = enabledFeatures.shaderBufferFloat32AtomicAdd;
            shaderAtomicFloatFeatures.shaderBufferFloat64Atomics   = enabledFeatures.shaderBufferFloat64Atomics;
            shaderAtomicFloatFeatures.shaderBufferFloat64AtomicAdd = enabledFeatures.shaderBufferFloat64AtomicAdd;
            shaderAtomicFloatFeatures.shaderSharedFloat32Atomics   = enabledFeatures.shaderSharedFloat32Atomics;
            shaderAtomicFloatFeatures.shaderSharedFloat32AtomicAdd = enabledFeatures.shaderSharedFloat32AtomicAdd;
            shaderAtomicFloatFeatures.shaderSharedFloat64Atomics   = enabledFeatures.shaderSharedFloat64Atomics;
            shaderAtomicFloatFeatures.shaderSharedFloat64AtomicAdd = enabledFeatures.shaderSharedFloat64AtomicAdd;
            shaderAtomicFloatFeatures.shaderImageFloat32Atomics    = enabledFeatures.shaderImageFloat32Atomics;
            shaderAtomicFloatFeatures.shaderImageFloat32AtomicAdd  = enabledFeatures.shaderImageFloat32AtomicAdd;
            shaderAtomicFloatFeatures.sparseImageFloat32Atomics    = enabledFeatures.sparseImageFloat32Atomics;
            shaderAtomicFloatFeatures.sparseImageFloat32AtomicAdd  = enabledFeatures.sparseImageFloat32AtomicAdd;
            addFeatureToChain(&shaderAtomicFloatFeatures);
        }
            
        if (enabledFeatures.shaderBufferFloat16Atomics ||
            enabledFeatures.shaderBufferFloat16AtomicAdd ||
            enabledFeatures.shaderBufferFloat16AtomicMinMax ||
            enabledFeatures.shaderBufferFloat32AtomicMinMax ||
            enabledFeatures.shaderBufferFloat64AtomicMinMax ||
            enabledFeatures.shaderSharedFloat16Atomics ||
            enabledFeatures.shaderSharedFloat16AtomicAdd ||
            enabledFeatures.shaderSharedFloat16AtomicMinMax ||
            enabledFeatures.shaderSharedFloat32AtomicMinMax ||
            enabledFeatures.shaderSharedFloat64AtomicMinMax ||
            enabledFeatures.shaderImageFloat32AtomicMinMax ||
            enabledFeatures.sparseImageFloat32AtomicMinMax)
        {
            insertExtensionIfAvailable(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME); // Requirement
            insertExtensionIfAvailable(VK_EXT_SHADER_ATOMIC_FLOAT_2_EXTENSION_NAME);
            shaderAtomicFloat2Features.shaderBufferFloat16Atomics       = enabledFeatures.shaderBufferFloat16Atomics;
            shaderAtomicFloat2Features.shaderBufferFloat16AtomicAdd     = enabledFeatures.shaderBufferFloat16AtomicAdd;
            shaderAtomicFloat2Features.shaderBufferFloat16AtomicMinMax  = enabledFeatures.shaderBufferFloat16AtomicMinMax;
            shaderAtomicFloat2Features.shaderBufferFloat32AtomicMinMax  = enabledFeatures.shaderBufferFloat32AtomicMinMax;
            shaderAtomicFloat2Features.shaderBufferFloat64AtomicMinMax  = enabledFeatures.shaderBufferFloat64AtomicMinMax;
            shaderAtomicFloat2Features.shaderSharedFloat16Atomics       = enabledFeatures.shaderSharedFloat16Atomics;
            shaderAtomicFloat2Features.shaderSharedFloat16AtomicAdd     = enabledFeatures.shaderSharedFloat16AtomicAdd;
            shaderAtomicFloat2Features.shaderSharedFloat16AtomicMinMax  = enabledFeatures.shaderSharedFloat16AtomicMinMax;
            shaderAtomicFloat2Features.shaderSharedFloat32AtomicMinMax  = enabledFeatures.shaderSharedFloat32AtomicMinMax;
            shaderAtomicFloat2Features.shaderSharedFloat64AtomicMinMax  = enabledFeatures.shaderSharedFloat64AtomicMinMax;
            shaderAtomicFloat2Features.shaderImageFloat32AtomicMinMax   = enabledFeatures.shaderImageFloat32AtomicMinMax;
            shaderAtomicFloat2Features.sparseImageFloat32AtomicMinMax   = enabledFeatures.sparseImageFloat32AtomicMinMax;
            addFeatureToChain(&shaderAtomicFloat2Features);
        }
            
        if (enabledFeatures.shaderImageInt64Atomics ||
            enabledFeatures.sparseImageInt64Atomics)
        {
            // All Requirements Exist in Vulkan 1.1
            insertExtensionIfAvailable(VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME);
            shaderImageAtomicInt64Features.shaderImageInt64Atomics = enabledFeatures.shaderImageInt64Atomics;
            shaderImageAtomicInt64Features.sparseImageInt64Atomics = enabledFeatures.sparseImageInt64Atomics;
            addFeatureToChain(&shaderImageAtomicInt64Features);
        }
            
        if (enabledFeatures.accelerationStructure ||
            enabledFeatures.accelerationStructureIndirectBuild ||
            enabledFeatures.accelerationStructureHostCommands ||
            enabledFeatures.descriptorBindingAccelerationStructureUpdateAfterBind)
        {
            // IMPLICIT ENABLE: descriptorIndexing -> Already handled because of resolveFeatureDependencies(featuresToEnable);
            // IMPLICIT ENABLE: bufferDeviceAddress -> Already handled because of resolveFeatureDependencies(featuresToEnable);
            // IMPLICIT ENABLE: VK_KHR_DEFERRED_HOST_OPERATIONS -> Already handled because of resolveFeatureDependencies(featuresToEnable);

            insertExtensionIfAvailable(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
            accelerationStructureFeatures.accelerationStructure = enabledFeatures.accelerationStructure;
            accelerationStructureFeatures.accelerationStructureIndirectBuild = enabledFeatures.accelerationStructureIndirectBuild;
            accelerationStructureFeatures.accelerationStructureHostCommands = enabledFeatures.accelerationStructureHostCommands;
            accelerationStructureFeatures.descriptorBindingAccelerationStructureUpdateAfterBind = enabledFeatures.descriptorBindingAccelerationStructureUpdateAfterBind;
            addFeatureToChain(&accelerationStructureFeatures);
        }
            
        if (enabledFeatures.rayQuery)
        {
            // IMPLICIT ENABLE: accelerationStructure -> Already handled because of resolveFeatureDependencies(featuresToEnable);
            insertExtensionIfAvailable(VK_KHR_SPIRV_1_4_EXTENSION_NAME); // Requires VK_KHR_spirv_1_4
            insertExtensionIfAvailable(VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME); // VK_KHR_spirv_1_4 requires VK_KHR_shader_float_controls

            insertExtensionIfAvailable(VK_KHR_RAY_QUERY_EXTENSION_NAME);
            rayQueryFeatures.rayQuery = enabledFeatures.rayQuery;
            addFeatureToChain(&rayQueryFeatures);
        }
            
        if (enabledFeatures.rayTracingPipeline ||
            enabledFeatures.rayTracingPipelineTraceRaysIndirect ||
            enabledFeatures.rayTraversalPrimitiveCulling)
        {
            // IMPLICIT ENABLE: accelerationStructure -> Already handled because of resolveFeatureDependencies(featuresToEnable);
            insertExtensionIfAvailable(VK_KHR_SPIRV_1_4_EXTENSION_NAME); // Requires VK_KHR_spirv_1_4
            insertExtensionIfAvailable(VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME); // VK_KHR_spirv_1_4 requires VK_KHR_shader_float_controls
                
            insertExtensionIfAvailable(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
            rayTracingPipelineFeatures.rayTracingPipeline = enabledFeatures.rayTracingPipeline;
            rayTracingPipelineFeatures.rayTracingPipelineTraceRaysIndirect = enabledFeatures.rayTracingPipelineTraceRaysIndirect;
            rayTracingPipelineFeatures.rayTraversalPrimitiveCulling = enabledFeatures.rayTraversalPrimitiveCulling;
            addFeatureToChain(&rayTracingPipelineFeatures);
        }

        if (enabledFeatures.rayTracingMotionBlur ||
            enabledFeatures.rayTracingMotionBlurPipelineTraceRaysIndirect)
        {
            // IMPLICIT ENABLE: rayTracingPipeline -> Already handled because of resolveFeatureDependencies(featuresToEnable);
            insertExtensionIfAvailable(VK_NV_RAY_TRACING_MOTION_BLUR_EXTENSION_NAME);
            rayTracingMotionBlurFeatures.rayTracingMotionBlur = enabledFeatures.rayTracingMotionBlur;
            rayTracingMotionBlurFeatures.rayTracingMotionBlurPipelineTraceRaysIndirect = enabledFeatures.rayTracingMotionBlurPipelineTraceRaysIndirect;
            addFeatureToChain(&rayTracingMotionBlurFeatures);
        }
            
        if(enabledFeatures.shaderDeviceClock)
        {
            // shaderClockFeatures.shaderSubgroupClock will be enabled by defaul and the extension name and feature struct should've been added by now, but the functions implicitly protect against duplication
            insertExtensionIfAvailable(VK_KHR_SHADER_CLOCK_EXTENSION_NAME);
            shaderClockFeatures.shaderDeviceClock = enabledFeatures.shaderDeviceClock;
            addFeatureToChain(&shaderClockFeatures);
        }

        CHECK_VULKAN_EXTENTION_FOR_SINGLE_VAR_FEATURE(shaderSubgroupUniformControlFlow, VK_KHR_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_EXTENSION_NAME, subgroupUniformControlFlowFeatures);
            
        if (enabledFeatures.workgroupMemoryExplicitLayout ||
            enabledFeatures.workgroupMemoryExplicitLayoutScalarBlockLayout ||
            enabledFeatures.workgroupMemoryExplicitLayout8BitAccess ||
            enabledFeatures.workgroupMemoryExplicitLayout16BitAccess)
        {
            insertExtensionIfAvailable(VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME);
            workgroupMemoryExplicitLayout.workgroupMemoryExplicitLayout = enabledFeatures.workgroupMemoryExplicitLayout;
            workgroupMemoryExplicitLayout.workgroupMemoryExplicitLayoutScalarBlockLayout = enabledFeatures.workgroupMemoryExplicitLayoutScalarBlockLayout;
            workgroupMemoryExplicitLayout.workgroupMemoryExplicitLayout8BitAccess = enabledFeatures.workgroupMemoryExplicitLayout8BitAccess;
            workgroupMemoryExplicitLayout.workgroupMemoryExplicitLayout16BitAccess = enabledFeatures.workgroupMemoryExplicitLayout16BitAccess;
            addFeatureToChain(&workgroupMemoryExplicitLayout);
        }
            
        if (enabledFeatures.computeDerivativeGroupQuads ||
            enabledFeatures.computeDerivativeGroupLinear)
        {
            // All Requirements Exist in Vulkan 1.1
            insertExtensionIfAvailable(VK_NV_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME);
            computeShaderDerivativesFeatures.computeDerivativeGroupQuads = enabledFeatures.computeDerivativeGroupQuads;
            computeShaderDerivativesFeatures.computeDerivativeGroupLinear = enabledFeatures.computeDerivativeGroupLinear;
            addFeatureToChain(&computeShaderDerivativesFeatures);
        }
            
        if (enabledFeatures.cooperativeMatrix ||
            enabledFeatures.cooperativeMatrixRobustBufferAccess)
        {
            // All Requirements Exist in Vulkan 1.1
            insertExtensionIfAvailable(VK_NV_COOPERATIVE_MATRIX_EXTENSION_NAME);
            cooperativeMatrixFeatures.cooperativeMatrix = enabledFeatures.cooperativeMatrix;
            cooperativeMatrixFeatures.cooperativeMatrixRobustBufferAccess = enabledFeatures.cooperativeMatrixRobustBufferAccess;
            addFeatureToChain(&cooperativeMatrixFeatures);
        }
            
        CHECK_VULKAN_EXTENTION_FOR_SINGLE_VAR_FEATURE(coverageReductionMode, VK_NV_COVERAGE_REDUCTION_MODE_EXTENSION_NAME, coverageReductionModeFeatures);
            
        // IMPLICIT ENABLE: deviceGeneratedCommands requires bufferDeviceAddress -> Already handled because of resolveFeatureDependencies(featuresToEnable); 
        CHECK_VULKAN_EXTENTION_FOR_SINGLE_VAR_FEATURE(deviceGeneratedCommands, VK_NV_DEVICE_GENERATED_COMMANDS_EXTENSION_NAME, deviceGeneratedCommandsFeatures);
            
        if (enabledFeatures.taskShader ||
            enabledFeatures.meshShader)
        {
            // All Requirements Exist in Vulkan 1.1
            insertExtensionIfAvailable(VK_NV_MESH_SHADER_EXTENSION_NAME);
            meshShaderFeatures.taskShader = enabledFeatures.taskShader;
            meshShaderFeatures.meshShader = enabledFeatures.meshShader;
            addFeatureToChain(&meshShaderFeatures);
        }

        if (enabledFeatures.mixedAttachmentSamples)
        {
            // All Requirements Exist in Vulkan 1.1
            insertExtensionIfAvailable(VK_AMD_MIXED_ATTACHMENT_SAMPLES_EXTENSION_NAME);
            insertExtensionIfAvailable(VK_NV_FRAMEBUFFER_MIXED_SAMPLES_EXTENSION_NAME);
        }

        if (enabledFeatures.hdrMetadata)
        {
            // IMPLICIT ENABLE: VK_KHR_swapchain -> Already handled because of resolveFeatureDependencies(featuresToEnable); 
            insertExtensionIfAvailable(VK_EXT_HDR_METADATA_EXTENSION_NAME);
        }
            
        if (enabledFeatures.displayTiming)
        {
            // IMPLICIT ENABLE: VK_KHR_swapchain -> Already handled because of resolveFeatureDependencies(featuresToEnable); 
            insertExtensionIfAvailable(VK_GOOGLE_DISPLAY_TIMING_EXTENSION_NAME);
        }
            
        if (enabledFeatures.rasterizationOrder)
            insertExtensionIfAvailable(VK_AMD_RASTERIZATION_ORDER_EXTENSION_NAME);
            
        if (enabledFeatures.shaderExplicitVertexParameter)
            insertExtensionIfAvailable(VK_AMD_SHADER_EXPLICIT_VERTEX_PARAMETER_EXTENSION_NAME);
            
        if (enabledFeatures.shaderInfoAMD)
            insertExtensionIfAvailable(VK_AMD_SHADER_INFO_EXTENSION_NAME);
            
        if (enabledFeatures.pipelineCreationCacheControl)
            insertExtensionIfAvailable(VK_EXT_PIPELINE_CREATION_CACHE_CONTROL_EXTENSION_NAME);

        CHECK_VULKAN_EXTENTION_FOR_SINGLE_VAR_FEATURE(colorWriteEnable, VK_EXT_COLOR_WRITE_ENABLE_EXTENSION_NAME, colorWriteEnableFeatures);
            
        if (enabledFeatures.conditionalRendering ||
            enabledFeatures.inheritedConditionalRendering)
        {
            // All Requirements Exist in Vulkan 1.1
            insertExtensionIfAvailable(VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME);
            conditionalRenderingFeatures.conditionalRendering = enabledFeatures.conditionalRendering;
            conditionalRenderingFeatures.inheritedConditionalRendering = enabledFeatures.inheritedConditionalRendering;
            addFeatureToChain(&conditionalRenderingFeatures);
        }

        CHECK_VULKAN_EXTENTION_FOR_SINGLE_VAR_FEATURE(deviceMemoryReport, VK_EXT_DEVICE_MEMORY_REPORT_EXTENSION_NAME, deviceMemoryReportFeatures);
            
        if (enabledFeatures.fragmentDensityMap ||
            enabledFeatures.fragmentDensityMapDynamic ||
            enabledFeatures.fragmentDensityMapNonSubsampledImages)
        {
            insertExtensionIfAvailable(VK_EXT_FRAGMENT_DENSITY_MAP_EXTENSION_NAME);
            fragmentDensityMapFeatures.fragmentDensityMap = enabledFeatures.fragmentDensityMap;
            fragmentDensityMapFeatures.fragmentDensityMapDynamic = enabledFeatures.fragmentDensityMapDynamic;
            fragmentDensityMapFeatures.fragmentDensityMapNonSubsampledImages = enabledFeatures.fragmentDensityMapNonSubsampledImages;
            addFeatureToChain(&fragmentDensityMapFeatures);
        }

        if (enabledFeatures.fragmentDensityMapDeferred)
        {
            // IMPLICIT ENABLE: fragmentDensityMap -> Already handled because of resolveFeatureDependencies(featuresToEnable);
            insertExtensionIfAvailable(VK_EXT_FRAGMENT_DENSITY_MAP_2_EXTENSION_NAME);
            fragmentDensityMap2Features.fragmentDensityMapDeferred = enabledFeatures.fragmentDensityMapDeferred;
            addFeatureToChain(&fragmentDensityMap2Features);
        }
            
        if (enabledFeatures.robustImageAccess)
            insertExtensionIfAvailable(VK_EXT_IMAGE_ROBUSTNESS_EXTENSION_NAME);
            
        if (enabledFeatures.rectangularLines ||
            enabledFeatures.bresenhamLines ||
            enabledFeatures.smoothLines ||
            enabledFeatures.stippledRectangularLines ||
            enabledFeatures.stippledBresenhamLines ||
            enabledFeatures.stippledSmoothLines)
        {
            insertExtensionIfAvailable(VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME);
            lineRasterizationFeatures.rectangularLines = enabledFeatures.rectangularLines;
            lineRasterizationFeatures.bresenhamLines = enabledFeatures.bresenhamLines;
            lineRasterizationFeatures.smoothLines = enabledFeatures.smoothLines;
            lineRasterizationFeatures.stippledRectangularLines = enabledFeatures.stippledRectangularLines;
            lineRasterizationFeatures.stippledBresenhamLines = enabledFeatures.stippledBresenhamLines;
            lineRasterizationFeatures.stippledSmoothLines = enabledFeatures.stippledSmoothLines;
            addFeatureToChain(&lineRasterizationFeatures);
        }

        CHECK_VULKAN_EXTENTION_FOR_SINGLE_VAR_FEATURE(memoryPriority, VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME, memoryPriorityFeatures);
            
        if (enabledFeatures.robustBufferAccess2 ||
            enabledFeatures.robustImageAccess2 ||
            enabledFeatures.nullDescriptor)
        {
            insertExtensionIfAvailable(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME);
            robustness2Features.robustBufferAccess2 = enabledFeatures.robustBufferAccess2;
            robustness2Features.robustImageAccess2 = enabledFeatures.robustImageAccess2;
            robustness2Features.nullDescriptor = enabledFeatures.nullDescriptor;
            addFeatureToChain(&robustness2Features);
        }
            
        if (enabledFeatures.performanceCounterQueryPools ||
            enabledFeatures.performanceCounterMultipleQueryPools)
        {
            insertExtensionIfAvailable(VK_INTEL_PERFORMANCE_QUERY_EXTENSION_NAME);
            performanceQueryFeatures.performanceCounterQueryPools = enabledFeatures.performanceCounterQueryPools;
            performanceQueryFeatures.performanceCounterMultipleQueryPools = enabledFeatures.performanceCounterMultipleQueryPools;
            addFeatureToChain(&performanceQueryFeatures);
        }

        CHECK_VULKAN_EXTENTION_FOR_SINGLE_VAR_FEATURE(pipelineExecutableInfo, VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME, pipelineExecutablePropertiesFeatures);
        
        CHECK_VULKAN_EXTENTION_FOR_SINGLE_VAR_FEATURE(deviceCoherentMemory, VK_AMD_DEVICE_COHERENT_MEMORY_EXTENSION_NAME, coherentMemoryFeatures);
            
        if (enabledFeatures.bufferMarkerAMD)
            insertExtensionIfAvailable(VK_AMD_BUFFER_MARKER_EXTENSION_NAME);

        if (enabledFeatures.swapchainMode.hasFlags(E_SWAPCHAIN_MODE::ESM_SURFACE))
        {
            // If we reach here then the instance extension VK_KHR_Surface was definitely enabled otherwise the extension wouldn't be reported by physical device
            insertExtensionIfAvailable(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
            insertExtensionIfAvailable(VK_KHR_SWAPCHAIN_MUTABLE_FORMAT_EXTENSION_NAME);
            // TODO: https://github.com/Devsh-Graphics-Programming/Nabla/issues/508
        }
        
        if (enabledFeatures.deferredHostOperations)
            insertExtensionIfAvailable(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);

#undef CHECK_VULKAN_1_2_FEATURE_FOR_SINGLE_VAR
#undef CHECK_VULKAN_1_2_FEATURE_FOR_EXT_ALIAS
#undef CHECK_VULKAN_EXTENTION_FOR_SINGLE_VAR_FEATURE
        
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

        vk_createInfo.queueCreateInfoCount = params.queueParamsCount;
        core::vector<VkDeviceQueueCreateInfo> queueCreateInfos(vk_createInfo.queueCreateInfoCount);
        for (uint32_t i = 0u; i < queueCreateInfos.size(); ++i)
        {
            const auto& qparams = params.queueParams[i];
            auto& qci = queueCreateInfos[i];
                    
            qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            qci.pNext = nullptr;
            qci.queueCount = qparams.count;
            qci.queueFamilyIndex = qparams.familyIndex;
            qci.flags = static_cast<VkDeviceQueueCreateFlags>(qparams.flags);
            qci.pQueuePriorities = qparams.priorities;
        }
        vk_createInfo.pQueueCreateInfos = queueCreateInfos.data();
#endif
    vk_createInfo.enabledExtensionCount = static_cast<uint32_t>(extensionStrings.size());
    vk_createInfo.ppEnabledExtensionNames = extensionStrings.data();

    if (!params.compilerSet)
        params.compilerSet = core::make_smart_refctd_ptr<asset::CCompilerSet>(core::smart_refctd_ptr(m_system));

    VkDevice vk_device = VK_NULL_HANDLE;
    if (vkCreateDevice(m_vkPhysicalDevice,&vk_createInfo,nullptr,&vk_device)!=VK_SUCCESS)
        return nullptr;

    return core::make_smart_refctd_ptr<CVulkanLogicalDevice>(core::smart_refctd_ptr<const IAPIConnection>(m_api),m_rdoc_api,this,vk_device,params);
}

}