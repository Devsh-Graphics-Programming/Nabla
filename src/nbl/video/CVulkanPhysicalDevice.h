#ifndef __NBL_C_VULKAN_PHYSICAL_DEVICE_H_INCLUDED__
#define __NBL_C_VULKAN_PHYSICAL_DEVICE_H_INCLUDED__

#include "nbl/video/IPhysicalDevice.h"
#include "nbl/video/CVulkanLogicalDevice.h"
#include "nbl/video/surface/CSurfaceVulkan.h"
#include "nbl/video/CVulkanCommon.h"

namespace nbl::video
{
        
class CVulkanPhysicalDevice final : public IPhysicalDevice
{
public:
    CVulkanPhysicalDevice(core::smart_refctd_ptr<system::ISystem>&& sys, core::smart_refctd_ptr<asset::IGLSLCompiler>&& glslc, IAPIConnection* api, renderdoc_api_t* rdoc, VkPhysicalDevice vk_physicalDevice, VkInstance vk_instance, uint32_t instanceApiVersion)
        : IPhysicalDevice(std::move(sys),std::move(glslc)), m_api(api), m_rdoc_api(rdoc), m_vkPhysicalDevice(vk_physicalDevice), m_vkInstance(vk_instance)
    {
        // Get Supported Extensions
        {
            uint32_t count;
            VkResult res = vkEnumerateDeviceExtensionProperties(m_vkPhysicalDevice, nullptr, &count, nullptr);
            assert(VK_SUCCESS == res);
            core::vector<VkExtensionProperties> vk_extensions(count);
            res = vkEnumerateDeviceExtensionProperties(m_vkPhysicalDevice, nullptr, &count, vk_extensions.data());
            assert(VK_SUCCESS == res);

            for (const auto& vk_extension : vk_extensions)
                m_availableFeatureSet.insert(vk_extension.extensionName);
        }
        
        // Get physical device's limits/properties
        
        // !! Always check the API version is >= 1.3 before using `vulkan13Properties`
        VkPhysicalDeviceVulkan13Properties              vulkan13Properties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES, nullptr };
        VkPhysicalDeviceMaintenance4Properties          maintanance4Properties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES, &vulkan13Properties};
        VkPhysicalDeviceInlineUniformBlockProperties    inlineUniformBlockProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_PROPERTIES, &maintanance4Properties };
        VkPhysicalDeviceSubgroupSizeControlProperties   subgroupSizeControlProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES, &inlineUniformBlockProperties };
        
        // !! Always check the API version is >= 1.2 before using `vulkan12Properties`
        VkPhysicalDeviceVulkan12Properties              vulkan12Properties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES, &subgroupSizeControlProperties };
        VkPhysicalDeviceDriverProperties                driverProperties              = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES, &vulkan12Properties };
        VkPhysicalDeviceFloatControlsProperties         floatControlsProperties       = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES, &driverProperties };
        VkPhysicalDeviceDescriptorIndexingProperties    descriptorIndexingProperties  = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES, &floatControlsProperties };
        VkPhysicalDeviceDepthStencilResolveProperties   depthStencilResolveProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_STENCIL_RESOLVE_PROPERTIES, &descriptorIndexingProperties };
        VkPhysicalDeviceSamplerFilterMinmaxProperties   samplerFilterMinmaxProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_FILTER_MINMAX_PROPERTIES, &depthStencilResolveProperties };
        VkPhysicalDeviceTimelineSemaphoreProperties     timelineSemaphoreProperties   = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_PROPERTIES, &samplerFilterMinmaxProperties };

        // !! Our minimum supported Vulkan version is 1.1, no need to check anything before using `vulkan11Properties`
        VkPhysicalDeviceVulkan11Properties vulkan11Properties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES, &timelineSemaphoreProperties };

        // Extensions
        VkPhysicalDeviceConservativeRasterizationPropertiesEXT conservativeRasterizationProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONSERVATIVE_RASTERIZATION_PROPERTIES_EXT, &vulkan11Properties };
        VkPhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingPipelineProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR, &conservativeRasterizationProperties };
        VkPhysicalDeviceAccelerationStructurePropertiesKHR accelerationStructureProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR, &rayTracingPipelineProperties };
        {
            VkPhysicalDeviceProperties2 deviceProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
            deviceProperties.pNext = &accelerationStructureProperties;
            vkGetPhysicalDeviceProperties2(m_vkPhysicalDevice, &deviceProperties);

            /* Vulkan 1.0 Core  */
            
            uint32_t apiVersion = std::min(instanceApiVersion, deviceProperties.properties.apiVersion);
            assert(apiVersion >= MinimumVulkanApiVersion);
            m_properties.apiVersion.major = VK_API_VERSION_MAJOR(apiVersion);
            m_properties.apiVersion.minor = VK_API_VERSION_MINOR(apiVersion);
            m_properties.apiVersion.patch = VK_API_VERSION_PATCH(apiVersion);

            m_properties.driverVersion = deviceProperties.properties.driverVersion;
            m_properties.vendorID = deviceProperties.properties.vendorID;
            m_properties.deviceID = deviceProperties.properties.deviceID;
            m_properties.deviceType = static_cast<E_TYPE>(deviceProperties.properties.deviceType);
            memcpy(m_properties.deviceName, deviceProperties.properties.deviceName, VK_MAX_PHYSICAL_DEVICE_NAME_SIZE);
            memcpy(m_properties.pipelineCacheUUID, deviceProperties.properties.pipelineCacheUUID, VK_UUID_SIZE);
            
            m_properties.limits.maxImageDimension1D = deviceProperties.properties.limits.maxImageDimension1D;
            m_properties.limits.maxImageDimension2D = deviceProperties.properties.limits.maxImageDimension2D;
            m_properties.limits.maxImageDimension3D = deviceProperties.properties.limits.maxImageDimension3D;
            m_properties.limits.maxImageDimensionCube = deviceProperties.properties.limits.maxImageDimensionCube;
            m_properties.limits.maxImageArrayLayers = deviceProperties.properties.limits.maxImageArrayLayers;
            m_properties.limits.maxBufferViewTexels = deviceProperties.properties.limits.maxTexelBufferElements;
            m_properties.limits.maxUBOSize = deviceProperties.properties.limits.maxUniformBufferRange;
            m_properties.limits.maxSSBOSize = deviceProperties.properties.limits.maxStorageBufferRange;
            m_properties.limits.maxPushConstantsSize = deviceProperties.properties.limits.maxPushConstantsSize;
            m_properties.limits.maxMemoryAllocationCount = deviceProperties.properties.limits.maxMemoryAllocationCount;
            m_properties.limits.maxSamplerAllocationCount = deviceProperties.properties.limits.maxSamplerAllocationCount;
            m_properties.limits.bufferImageGranularity = deviceProperties.properties.limits.bufferImageGranularity;

            // Max PerStage Descriptors
            m_properties.limits.maxPerStageDescriptorSamplers = deviceProperties.properties.limits.maxPerStageDescriptorSamplers;
            m_properties.limits.maxPerStageDescriptorUBOs = deviceProperties.properties.limits.maxPerStageDescriptorUniformBuffers;
            m_properties.limits.maxPerStageDescriptorSSBOs = deviceProperties.properties.limits.maxPerStageDescriptorStorageBuffers;
            m_properties.limits.maxPerStageDescriptorImages = deviceProperties.properties.limits.maxPerStageDescriptorSampledImages;
            m_properties.limits.maxPerStageDescriptorStorageImages = deviceProperties.properties.limits.maxPerStageDescriptorStorageImages;
            m_properties.limits.maxPerStageDescriptorInputAttachments = deviceProperties.properties.limits.maxPerStageDescriptorInputAttachments;
            m_properties.limits.maxPerStageResources = deviceProperties.properties.limits.maxPerStageResources;
            
            // Max Descriptors
            m_properties.limits.maxDescriptorSetSamplers = deviceProperties.properties.limits.maxDescriptorSetSamplers;
            m_properties.limits.maxDescriptorSetUBOs = deviceProperties.properties.limits.maxDescriptorSetUniformBuffers;
            m_properties.limits.maxDescriptorSetDynamicOffsetUBOs = deviceProperties.properties.limits.maxDescriptorSetUniformBuffersDynamic;
            m_properties.limits.maxDescriptorSetSSBOs = deviceProperties.properties.limits.maxDescriptorSetStorageBuffers;
            m_properties.limits.maxDescriptorSetDynamicOffsetSSBOs = deviceProperties.properties.limits.maxDescriptorSetStorageBuffersDynamic;
            m_properties.limits.maxDescriptorSetImages = deviceProperties.properties.limits.maxDescriptorSetSampledImages;
            m_properties.limits.maxDescriptorSetStorageImages = deviceProperties.properties.limits.maxDescriptorSetStorageImages;
            m_properties.limits.maxDescriptorSetInputAttachments = deviceProperties.properties.limits.maxDescriptorSetInputAttachments;

            m_properties.limits.maxVertexOutputComponents = deviceProperties.properties.limits.maxVertexOutputComponents;
            m_properties.limits.maxTessellationGenerationLevel = deviceProperties.properties.limits.maxTessellationGenerationLevel;
            m_properties.limits.maxTessellationPatchSize = deviceProperties.properties.limits.maxTessellationPatchSize;
            m_properties.limits.maxTessellationControlPerVertexInputComponents = deviceProperties.properties.limits.maxTessellationControlPerVertexInputComponents;
            m_properties.limits.maxTessellationControlPerVertexOutputComponents = deviceProperties.properties.limits.maxTessellationControlPerVertexOutputComponents;
            m_properties.limits.maxTessellationControlPerPatchOutputComponents = deviceProperties.properties.limits.maxTessellationControlPerPatchOutputComponents;
            m_properties.limits.maxTessellationControlTotalOutputComponents = deviceProperties.properties.limits.maxTessellationControlTotalOutputComponents;
            m_properties.limits.maxTessellationEvaluationInputComponents = deviceProperties.properties.limits.maxTessellationEvaluationInputComponents;
            m_properties.limits.maxTessellationEvaluationOutputComponents = deviceProperties.properties.limits.maxTessellationEvaluationOutputComponents;
            m_properties.limits.maxGeometryShaderInvocations = deviceProperties.properties.limits.maxGeometryShaderInvocations;
            m_properties.limits.maxGeometryInputComponents = deviceProperties.properties.limits.maxGeometryInputComponents;
            m_properties.limits.maxGeometryOutputComponents = deviceProperties.properties.limits.maxGeometryOutputComponents;
            m_properties.limits.maxGeometryOutputVertices = deviceProperties.properties.limits.maxGeometryOutputVertices;
            m_properties.limits.maxGeometryTotalOutputComponents = deviceProperties.properties.limits.maxGeometryTotalOutputComponents;
            m_properties.limits.maxFragmentInputComponents = deviceProperties.properties.limits.maxFragmentInputComponents;
            m_properties.limits.maxFragmentOutputAttachments = deviceProperties.properties.limits.maxFragmentOutputAttachments;
            m_properties.limits.maxFragmentDualSrcAttachments = deviceProperties.properties.limits.maxFragmentDualSrcAttachments;
            m_properties.limits.maxFragmentCombinedOutputResources = deviceProperties.properties.limits.maxFragmentCombinedOutputResources;
            m_properties.limits.maxComputeSharedMemorySize = deviceProperties.properties.limits.maxComputeSharedMemorySize;
            m_properties.limits.maxComputeWorkGroupCount[0] = deviceProperties.properties.limits.maxComputeWorkGroupCount[0];
            m_properties.limits.maxComputeWorkGroupCount[1] = deviceProperties.properties.limits.maxComputeWorkGroupCount[1];
            m_properties.limits.maxComputeWorkGroupCount[2] = deviceProperties.properties.limits.maxComputeWorkGroupCount[2];
            m_properties.limits.maxComputeWorkGroupInvocations = deviceProperties.properties.limits.maxComputeWorkGroupInvocations;
            m_properties.limits.maxWorkgroupSize[0] = deviceProperties.properties.limits.maxComputeWorkGroupSize[0];
            m_properties.limits.maxWorkgroupSize[1] = deviceProperties.properties.limits.maxComputeWorkGroupSize[1];
            m_properties.limits.maxWorkgroupSize[2] = deviceProperties.properties.limits.maxComputeWorkGroupSize[2];
            m_properties.limits.subPixelPrecisionBits = deviceProperties.properties.limits.subPixelPrecisionBits;
            m_properties.limits.maxDrawIndirectCount = deviceProperties.properties.limits.maxDrawIndirectCount;
            m_properties.limits.maxSamplerLodBias = deviceProperties.properties.limits.maxSamplerLodBias;
            m_properties.limits.maxSamplerAnisotropyLog2 = static_cast<uint8_t>(std::log2(deviceProperties.properties.limits.maxSamplerAnisotropy));
            m_properties.limits.maxViewports = deviceProperties.properties.limits.maxViewports;
            m_properties.limits.maxViewportDims[0] = deviceProperties.properties.limits.maxViewportDimensions[0];
            m_properties.limits.maxViewportDims[1] = deviceProperties.properties.limits.maxViewportDimensions[1];
            m_properties.limits.viewportBoundsRange[0] = deviceProperties.properties.limits.viewportBoundsRange[0];
            m_properties.limits.viewportBoundsRange[1] = deviceProperties.properties.limits.viewportBoundsRange[1];
            m_properties.limits.viewportSubPixelBits = deviceProperties.properties.limits.viewportSubPixelBits;
            m_properties.limits.minMemoryMapAlignment = deviceProperties.properties.limits.minMemoryMapAlignment;
            m_properties.limits.bufferViewAlignment = deviceProperties.properties.limits.minTexelBufferOffsetAlignment;
            m_properties.limits.minUBOAlignment = deviceProperties.properties.limits.minUniformBufferOffsetAlignment;
            m_properties.limits.minSSBOAlignment = deviceProperties.properties.limits.minStorageBufferOffsetAlignment;
            m_properties.limits.minTexelOffset = deviceProperties.properties.limits.minTexelOffset;
            m_properties.limits.maxTexelOffset = deviceProperties.properties.limits.maxTexelOffset;
            m_properties.limits.minTexelGatherOffset = deviceProperties.properties.limits.minTexelGatherOffset;
            m_properties.limits.maxTexelGatherOffset = deviceProperties.properties.limits.maxTexelGatherOffset;
            m_properties.limits.minInterpolationOffset = deviceProperties.properties.limits.minInterpolationOffset;
            m_properties.limits.maxInterpolationOffset = deviceProperties.properties.limits.maxInterpolationOffset;
            m_properties.limits.maxFramebufferWidth = deviceProperties.properties.limits.maxFramebufferWidth;
            m_properties.limits.maxFramebufferHeight = deviceProperties.properties.limits.maxFramebufferHeight;
            m_properties.limits.maxFramebufferLayers = deviceProperties.properties.limits.maxFramebufferLayers;
            m_properties.limits.framebufferColorSampleCounts = core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS>(deviceProperties.properties.limits.framebufferColorSampleCounts);
            m_properties.limits.framebufferDepthSampleCounts = core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS>(deviceProperties.properties.limits.framebufferDepthSampleCounts);
            m_properties.limits.framebufferStencilSampleCounts = core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS>(deviceProperties.properties.limits.framebufferStencilSampleCounts);
            m_properties.limits.framebufferNoAttachmentsSampleCounts = core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS>(deviceProperties.properties.limits.framebufferNoAttachmentsSampleCounts);
            m_properties.limits.maxColorAttachments = deviceProperties.properties.limits.maxColorAttachments;
            m_properties.limits.sampledImageColorSampleCounts = core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS>(deviceProperties.properties.limits.sampledImageColorSampleCounts);
            m_properties.limits.sampledImageIntegerSampleCounts = core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS>(deviceProperties.properties.limits.sampledImageIntegerSampleCounts);
            m_properties.limits.sampledImageDepthSampleCounts = core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS>(deviceProperties.properties.limits.sampledImageDepthSampleCounts);
            m_properties.limits.sampledImageStencilSampleCounts = core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS>(deviceProperties.properties.limits.sampledImageStencilSampleCounts);
            m_properties.limits.storageImageSampleCounts = core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS>(deviceProperties.properties.limits.storageImageSampleCounts);
            m_properties.limits.maxSampleMaskWords = deviceProperties.properties.limits.maxSampleMaskWords;
            m_properties.limits.timestampComputeAndGraphics = deviceProperties.properties.limits.timestampComputeAndGraphics;
            m_properties.limits.timestampPeriodInNanoSeconds = deviceProperties.properties.limits.timestampPeriod;
            m_properties.limits.maxClipDistances = deviceProperties.properties.limits.maxClipDistances;
            m_properties.limits.maxCullDistances = deviceProperties.properties.limits.maxCullDistances;
            m_properties.limits.maxCombinedClipAndCullDistances = deviceProperties.properties.limits.maxCombinedClipAndCullDistances;
            m_properties.limits.discreteQueuePriorities = deviceProperties.properties.limits.discreteQueuePriorities;
            m_properties.limits.pointSizeRange[0] = deviceProperties.properties.limits.pointSizeRange[0];
            m_properties.limits.pointSizeRange[1] = deviceProperties.properties.limits.pointSizeRange[1];
            m_properties.limits.lineWidthRange[0] = deviceProperties.properties.limits.lineWidthRange[0];
            m_properties.limits.lineWidthRange[1] = deviceProperties.properties.limits.lineWidthRange[1];
            m_properties.limits.pointSizeGranularity = deviceProperties.properties.limits.pointSizeGranularity;
            m_properties.limits.lineWidthGranularity = deviceProperties.properties.limits.lineWidthGranularity;
            m_properties.limits.strictLines = deviceProperties.properties.limits.strictLines;
            m_properties.limits.standardSampleLocations = deviceProperties.properties.limits.standardSampleLocations;
            m_properties.limits.optimalBufferCopyOffsetAlignment = deviceProperties.properties.limits.optimalBufferCopyOffsetAlignment;
            m_properties.limits.optimalBufferCopyRowPitchAlignment = deviceProperties.properties.limits.optimalBufferCopyRowPitchAlignment;
            m_properties.limits.nonCoherentAtomSize = deviceProperties.properties.limits.nonCoherentAtomSize;
            
            /* Vulkan 1.1 Core  */
            memcpy(m_properties.deviceUUID, vulkan11Properties.deviceUUID, VK_UUID_SIZE);
            memcpy(m_properties.driverUUID, vulkan11Properties.driverUUID, VK_UUID_SIZE);
            memcpy(m_properties.deviceLUID, vulkan11Properties.deviceLUID, VK_LUID_SIZE);
            m_properties.deviceNodeMask = vulkan11Properties.deviceNodeMask;
            m_properties.deviceLUIDValid = vulkan11Properties.deviceLUIDValid;
            m_properties.limits.maxPerSetDescriptors = vulkan11Properties.maxPerSetDescriptors;
            m_properties.limits.maxMemoryAllocationSize = vulkan11Properties.maxMemoryAllocationSize;

            /* SubgroupProperties */
            m_properties.limits.subgroupSize = vulkan11Properties.subgroupSize;
            m_properties.limits.subgroupOpsShaderStages = core::bitflag<asset::IShader::E_SHADER_STAGE>(vulkan11Properties.subgroupSupportedStages);
            m_properties.limits.shaderSubgroupBasic = vulkan11Properties.subgroupSupportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT;
            m_properties.limits.shaderSubgroupVote = vulkan11Properties.subgroupSupportedOperations & VK_SUBGROUP_FEATURE_VOTE_BIT;
            m_properties.limits.shaderSubgroupArithmetic = vulkan11Properties.subgroupSupportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
            m_properties.limits.shaderSubgroupBallot = vulkan11Properties.subgroupSupportedOperations & VK_SUBGROUP_FEATURE_BALLOT_BIT;
            m_properties.limits.shaderSubgroupShuffle = vulkan11Properties.subgroupSupportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
            m_properties.limits.shaderSubgroupShuffleRelative = vulkan11Properties.subgroupSupportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT;
            m_properties.limits.shaderSubgroupClustered = vulkan11Properties.subgroupSupportedOperations & VK_SUBGROUP_FEATURE_CLUSTERED_BIT;
            m_properties.limits.shaderSubgroupQuad = vulkan11Properties.subgroupSupportedOperations & VK_SUBGROUP_FEATURE_QUAD_BIT;
            m_properties.limits.shaderSubgroupQuadAllStages = vulkan11Properties.subgroupQuadOperationsInAllStages;
            m_properties.limits.pointClippingBehavior = static_cast<SLimits::E_POINT_CLIPPING_BEHAVIOR>(vulkan11Properties.pointClippingBehavior);

            /* Vulkan 1.2 Core  */
            m_properties.driverID = static_cast<E_DRIVER_ID>(driverProperties.driverID);
            memcpy(m_properties.driverName, driverProperties.driverName, VK_MAX_DRIVER_NAME_SIZE);
            memcpy(m_properties.driverInfo, driverProperties.driverInfo, VK_MAX_DRIVER_INFO_SIZE);
            m_properties.conformanceVersion = driverProperties.conformanceVersion;
            
            if(isExtensionSupported(VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME))
            {
                m_properties.limits.shaderSignedZeroInfNanPreserveFloat16   = floatControlsProperties.shaderSignedZeroInfNanPreserveFloat16;
                m_properties.limits.shaderSignedZeroInfNanPreserveFloat32   = floatControlsProperties.shaderSignedZeroInfNanPreserveFloat32;
                m_properties.limits.shaderSignedZeroInfNanPreserveFloat64   = floatControlsProperties.shaderSignedZeroInfNanPreserveFloat64;
                m_properties.limits.shaderDenormPreserveFloat16             = floatControlsProperties.shaderDenormPreserveFloat16;
                m_properties.limits.shaderDenormPreserveFloat32             = floatControlsProperties.shaderDenormPreserveFloat32;
                m_properties.limits.shaderDenormPreserveFloat64             = floatControlsProperties.shaderDenormPreserveFloat64;
                m_properties.limits.shaderDenormFlushToZeroFloat16          = floatControlsProperties.shaderDenormFlushToZeroFloat16;
                m_properties.limits.shaderDenormFlushToZeroFloat32          = floatControlsProperties.shaderDenormFlushToZeroFloat32;
                m_properties.limits.shaderDenormFlushToZeroFloat64          = floatControlsProperties.shaderDenormFlushToZeroFloat64;
                m_properties.limits.shaderRoundingModeRTEFloat16            = floatControlsProperties.shaderRoundingModeRTEFloat16;
                m_properties.limits.shaderRoundingModeRTEFloat32            = floatControlsProperties.shaderRoundingModeRTEFloat32;
                m_properties.limits.shaderRoundingModeRTEFloat64            = floatControlsProperties.shaderRoundingModeRTEFloat64;
                m_properties.limits.shaderRoundingModeRTZFloat16            = floatControlsProperties.shaderRoundingModeRTZFloat16;
                m_properties.limits.shaderRoundingModeRTZFloat32            = floatControlsProperties.shaderRoundingModeRTZFloat32;
                m_properties.limits.shaderRoundingModeRTZFloat64            = floatControlsProperties.shaderRoundingModeRTZFloat64;
            }
            
            if(isExtensionSupported(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME))
            {
                m_properties.limits.maxUpdateAfterBindDescriptorsInAllPools                 = descriptorIndexingProperties.maxUpdateAfterBindDescriptorsInAllPools;
                m_properties.limits.shaderUniformBufferArrayNonUniformIndexingNative        = descriptorIndexingProperties.shaderUniformBufferArrayNonUniformIndexingNative;
                m_properties.limits.shaderSampledImageArrayNonUniformIndexingNative         = descriptorIndexingProperties.shaderSampledImageArrayNonUniformIndexingNative;
                m_properties.limits.shaderStorageBufferArrayNonUniformIndexingNative        = descriptorIndexingProperties.shaderStorageBufferArrayNonUniformIndexingNative;
                m_properties.limits.shaderStorageImageArrayNonUniformIndexingNative         = descriptorIndexingProperties.shaderStorageImageArrayNonUniformIndexingNative;
                m_properties.limits.shaderInputAttachmentArrayNonUniformIndexingNative      = descriptorIndexingProperties.shaderInputAttachmentArrayNonUniformIndexingNative;
                m_properties.limits.robustBufferAccessUpdateAfterBind                       = descriptorIndexingProperties.robustBufferAccessUpdateAfterBind;
                m_properties.limits.quadDivergentImplicitLod                                = descriptorIndexingProperties.quadDivergentImplicitLod;
                m_properties.limits.maxPerStageDescriptorUpdateAfterBindSamplers            = descriptorIndexingProperties.maxPerStageDescriptorUpdateAfterBindSamplers;
                m_properties.limits.maxPerStageDescriptorUpdateAfterBindUBOs                = descriptorIndexingProperties.maxPerStageDescriptorUpdateAfterBindUniformBuffers;
                m_properties.limits.maxPerStageDescriptorUpdateAfterBindSSBOs               = descriptorIndexingProperties.maxPerStageDescriptorUpdateAfterBindStorageBuffers;
                m_properties.limits.maxPerStageDescriptorUpdateAfterBindImages              = descriptorIndexingProperties.maxPerStageDescriptorUpdateAfterBindSampledImages;
                m_properties.limits.maxPerStageDescriptorUpdateAfterBindStorageImages       = descriptorIndexingProperties.maxPerStageDescriptorUpdateAfterBindStorageImages;
                m_properties.limits.maxPerStageDescriptorUpdateAfterBindInputAttachments    = descriptorIndexingProperties.maxPerStageDescriptorUpdateAfterBindInputAttachments;
                m_properties.limits.maxPerStageUpdateAfterBindResources                     = descriptorIndexingProperties.maxPerStageUpdateAfterBindResources;
                m_properties.limits.maxDescriptorSetUpdateAfterBindSamplers                 = descriptorIndexingProperties.maxDescriptorSetUpdateAfterBindSamplers;
                m_properties.limits.maxDescriptorSetUpdateAfterBindUBOs                     = descriptorIndexingProperties.maxDescriptorSetUpdateAfterBindUniformBuffers;
                m_properties.limits.maxDescriptorSetUpdateAfterBindDynamicOffsetUBOs        = descriptorIndexingProperties.maxDescriptorSetUpdateAfterBindUniformBuffersDynamic;
                m_properties.limits.maxDescriptorSetUpdateAfterBindSSBOs                    = descriptorIndexingProperties.maxDescriptorSetUpdateAfterBindStorageBuffers;
                m_properties.limits.maxDescriptorSetUpdateAfterBindDynamicOffsetSSBOs       = descriptorIndexingProperties.maxDescriptorSetUpdateAfterBindStorageBuffersDynamic;
                m_properties.limits.maxDescriptorSetUpdateAfterBindImages                   = descriptorIndexingProperties.maxDescriptorSetUpdateAfterBindSampledImages;
                m_properties.limits.maxDescriptorSetUpdateAfterBindStorageImages            = descriptorIndexingProperties.maxDescriptorSetUpdateAfterBindStorageImages;
                m_properties.limits.maxDescriptorSetUpdateAfterBindInputAttachments         = descriptorIndexingProperties.maxDescriptorSetUpdateAfterBindInputAttachments;
            }
            
            if(isExtensionSupported(VK_EXT_SAMPLER_FILTER_MINMAX_EXTENSION_NAME))
            {
                m_properties.limits.filterMinmaxSingleComponentFormats = samplerFilterMinmaxProperties.filterMinmaxSingleComponentFormats;
                m_properties.limits.filterMinmaxImageComponentMapping = samplerFilterMinmaxProperties.filterMinmaxImageComponentMapping;
            }

            if(apiVersion >= VK_MAKE_API_VERSION(0, 1, 2, 0))
            {
                m_properties.limits.framebufferIntegerColorSampleCounts =  core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS>(vulkan12Properties.framebufferIntegerColorSampleCounts);
            }

            /* Vulkan 1.3 Core  */
            if(isExtensionSupported(VK_KHR_MAINTENANCE_4_EXTENSION_NAME))
                m_properties.limits.maxBufferSize = maintanance4Properties.maxBufferSize;
            else
                m_properties.limits.maxBufferSize = vulkan11Properties.maxMemoryAllocationSize;

            if(isExtensionSupported(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME))
            {
                m_properties.limits.minSubgroupSize = subgroupSizeControlProperties.minSubgroupSize;
                m_properties.limits.maxSubgroupSize = subgroupSizeControlProperties.maxSubgroupSize;
                m_properties.limits.maxComputeWorkgroupSubgroups = subgroupSizeControlProperties.maxComputeWorkgroupSubgroups;
                m_properties.limits.requiredSubgroupSizeStages = core::bitflag<asset::IShader::E_SHADER_STAGE>(subgroupSizeControlProperties.requiredSubgroupSizeStages);
            }
            else
            {
                getMinMaxSubgroupSizeFromDriverID(m_properties.driverID, m_properties.limits.minSubgroupSize, m_properties.limits.maxSubgroupSize);
            }

            /* Extensions */
            
            /* ConservativeRasterizationPropertiesEXT */
            if (isExtensionSupported(VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME))
            {
                m_properties.limits.primitiveOverestimationSize = conservativeRasterizationProperties.primitiveOverestimationSize;
                m_properties.limits.maxExtraPrimitiveOverestimationSize = conservativeRasterizationProperties.maxExtraPrimitiveOverestimationSize;
                m_properties.limits.extraPrimitiveOverestimationSizeGranularity = conservativeRasterizationProperties.extraPrimitiveOverestimationSizeGranularity;
                m_properties.limits.primitiveUnderestimation = conservativeRasterizationProperties.primitiveUnderestimation;
                m_properties.limits.conservativePointAndLineRasterization = conservativeRasterizationProperties.conservativePointAndLineRasterization;
                m_properties.limits.degenerateTrianglesRasterized = conservativeRasterizationProperties.degenerateTrianglesRasterized;
                m_properties.limits.degenerateLinesRasterized = conservativeRasterizationProperties.degenerateLinesRasterized;
                m_properties.limits.fullyCoveredFragmentShaderInputVariable = conservativeRasterizationProperties.fullyCoveredFragmentShaderInputVariable;
                m_properties.limits.conservativeRasterizationPostDepthCoverage = conservativeRasterizationProperties.conservativeRasterizationPostDepthCoverage;
            }
            
            /* AccelerationStructurePropertiesKHR */
            if (isExtensionSupported(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME))
            {
                m_properties.limits.maxGeometryCount = accelerationStructureProperties.maxGeometryCount;
                m_properties.limits.maxInstanceCount = accelerationStructureProperties.maxInstanceCount;
                m_properties.limits.maxPrimitiveCount = accelerationStructureProperties.maxPrimitiveCount;
                m_properties.limits.maxPerStageDescriptorAccelerationStructures = accelerationStructureProperties.maxPerStageDescriptorAccelerationStructures;
                m_properties.limits.maxPerStageDescriptorUpdateAfterBindAccelerationStructures = accelerationStructureProperties.maxPerStageDescriptorUpdateAfterBindAccelerationStructures;
                m_properties.limits.maxDescriptorSetAccelerationStructures = accelerationStructureProperties.maxDescriptorSetAccelerationStructures;
                m_properties.limits.maxDescriptorSetUpdateAfterBindAccelerationStructures = accelerationStructureProperties.maxDescriptorSetUpdateAfterBindAccelerationStructures;
                m_properties.limits.minAccelerationStructureScratchOffsetAlignment = accelerationStructureProperties.minAccelerationStructureScratchOffsetAlignment;
            }
            
            /* RayTracingPipelinePropertiesKHR */
            if (isExtensionSupported(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME))
            {
                m_properties.limits.shaderGroupHandleSize = rayTracingPipelineProperties.shaderGroupHandleSize;
                m_properties.limits.maxRayRecursionDepth = rayTracingPipelineProperties.maxRayRecursionDepth;
                m_properties.limits.maxShaderGroupStride = rayTracingPipelineProperties.maxShaderGroupStride;
                m_properties.limits.shaderGroupBaseAlignment = rayTracingPipelineProperties.shaderGroupBaseAlignment;
                m_properties.limits.shaderGroupHandleCaptureReplaySize = rayTracingPipelineProperties.shaderGroupHandleCaptureReplaySize;
                m_properties.limits.maxRayDispatchInvocationCount = rayTracingPipelineProperties.maxRayDispatchInvocationCount;
                m_properties.limits.shaderGroupHandleAlignment = rayTracingPipelineProperties.shaderGroupHandleAlignment;
                m_properties.limits.maxRayHitAttributeSize = rayTracingPipelineProperties.maxRayHitAttributeSize;
            }

            /* Nabla */
            m_properties.limits.allowCommandBufferQueryCopies = true; // always true in vk for all query types instead of PerformanceQuery which we don't support at the moment (have VkPhysicalDevicePerformanceQueryPropertiesKHR::allowCommandBufferQueryCopies in mind)
            m_properties.limits.maxOptimallyResidentWorkgroupInvocations = core::min(core::roundDownToPoT(deviceProperties.properties.limits.maxComputeWorkGroupInvocations),512u);
            constexpr auto beefyGPUWorkgroupMaxOccupancy = 256u; // TODO: find a way to query and report this somehow, persistent threads are very useful!
            m_properties.limits.maxResidentInvocations = beefyGPUWorkgroupMaxOccupancy*m_properties.limits.maxOptimallyResidentWorkgroupInvocations;

            /*
                [NO NABALA SUPPORT] Vulkan 1.0 implementation must support the 1.0 version of SPIR-V and the 1.0 version of the SPIR-V Extended Instructions for GLSL. If the VK_KHR_spirv_1_4 extension is enabled, the implementation must additionally support the 1.4 version of SPIR-V.
                A Vulkan 1.1 implementation must support the 1.0, 1.1, 1.2, and 1.3 versions of SPIR-V and the 1.0 version of the SPIR-V Extended Instructions for GLSL.
                A Vulkan 1.2 implementation must support the 1.0, 1.1, 1.2, 1.3, 1.4, and 1.5 versions of SPIR-V and the 1.0 version of the SPIR-V Extended Instructions for GLSL.
                A Vulkan 1.3 implementation must support the 1.0, 1.1, 1.2, 1.3, 1.4, and 1.5 versions of SPIR-V and the 1.0 version of the SPIR-V Extended Instructions for GLSL.
            */

            m_properties.limits.spirvVersion = asset::IGLSLCompiler::ESV_1_3;

            switch (VK_API_VERSION_MINOR(apiVersion))
            {
            case 0:
                m_properties.limits.spirvVersion = asset::IGLSLCompiler::ESV_1_0; 
                assert(false);
                break;
            case 1:
                m_properties.limits.spirvVersion = asset::IGLSLCompiler::ESV_1_3;
                break;
            case 2:
                m_properties.limits.spirvVersion = asset::IGLSLCompiler::ESV_1_5;
                break;
            case 3:
                m_properties.limits.spirvVersion = asset::IGLSLCompiler::ESV_1_5; //TODO(Erfan): Change to ESV_1_6 when we updated our glsl compiler submodules
                break;
            default:
                _NBL_DEBUG_BREAK_IF("Invalid Vulkan minor version!");
                break;
            }
        }
        
        // Get physical device's features
        // ! In Vulkan: These will be reported based on availability of an extension and will be enabled by enabling an extension
        // Table 51. Extension Feature Aliases (vkspec 1.3.211)
        // Extension                               Feature(s)
        // VK_KHR_shader_draw_parameters           shaderDrawParameters
        // VK_KHR_draw_indirect_count              drawIndirectCount
        // VK_KHR_sampler_mirror_clamp_to_edge     samplerMirrorClampToEdge
        // VK_EXT_descriptor_indexing              descriptorIndexing
        // VK_EXT_sampler_filter_minmax            samplerFilterMinmax
        // VK_EXT_shader_viewport_index_layer      shaderOutputViewportIndex, shaderOutputLayer
    

        // !! Our minimum supported Vulkan version is 1.1, no need to check anything before using `vulkan11Features`
        VkPhysicalDeviceVulkan11Features vulkan11Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, nullptr };

        // Extensions
        VkPhysicalDeviceSubgroupSizeControlFeaturesEXT subgroupSizeControlFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT, &vulkan11Features };
        VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipelineFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR, &subgroupSizeControlFeatures };
        VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR, &rayTracingPipelineFeatures };
        VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR, &accelerationFeatures };
        VkPhysicalDeviceBufferDeviceAddressFeaturesKHR bufferDeviceAddressFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_KHR, &rayQueryFeatures };
        VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT fragmentShaderInterlockFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT, &bufferDeviceAddressFeatures };
        {
            VkPhysicalDeviceFeatures2 deviceFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
            deviceFeatures.pNext = &fragmentShaderInterlockFeatures;
            vkGetPhysicalDeviceFeatures2(m_vkPhysicalDevice, &deviceFeatures);
            const auto& features = deviceFeatures.features;

            /* Vulkan 1.0 Core  */
            m_features.robustBufferAccess = features.robustBufferAccess;
            m_features.fullDrawIndexUint32 = features.fullDrawIndexUint32;
            m_features.imageCubeArray = features.imageCubeArray;
            m_features.independentBlend = features.independentBlend;
            m_features.geometryShader = features.geometryShader;
            m_features.tessellationShader = features.tessellationShader;
            m_features.logicOp = features.logicOp;
            m_features.multiDrawIndirect = features.multiDrawIndirect;
            m_features.drawIndirectFirstInstance = features.drawIndirectFirstInstance;
            m_features.depthClamp = features.depthClamp;
            m_features.depthBiasClamp = features.depthBiasClamp;
            m_features.fillModeNonSolid = features.fillModeNonSolid;
            m_features.depthBounds = features.depthBounds;
            m_features.wideLines = features.wideLines;
            m_features.largePoints = features.largePoints;
            m_features.alphaToOne = features.alphaToOne;
            m_features.multiViewport = features.multiViewport;
            m_features.samplerAnisotropy = features.samplerAnisotropy;
            m_features.shaderClipDistance = features.shaderClipDistance;
            m_features.shaderCullDistance = features.shaderCullDistance;
            m_features.vertexAttributeDouble = features.shaderFloat64;
            m_features.inheritedQueries = features.inheritedQueries;
            
            /* Vulkan 1.1 Core  */
            m_features.shaderDrawParameters = vulkan11Features.shaderDrawParameters;
            
            /* Vulkan 1.2 Core  */
            m_features.samplerMirrorClampToEdge = isExtensionSupported(VK_KHR_SAMPLER_MIRROR_CLAMP_TO_EDGE_EXTENSION_NAME);
            m_features.drawIndirectCount = isExtensionSupported(VK_KHR_DRAW_INDIRECT_COUNT_EXTENSION_NAME);
            m_features.samplerFilterMinmax = isExtensionSupported(VK_EXT_SAMPLER_FILTER_MINMAX_EXTENSION_NAME);
            
            /* Vulkan 1.3 Core  */
            if(isExtensionSupported(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME))
            {
                m_features.subgroupSizeControl  = subgroupSizeControlFeatures.subgroupSizeControl;
                m_features.computeFullSubgroups = subgroupSizeControlFeatures.computeFullSubgroups;
            }

            /* RayQueryFeaturesKHR */
            if (m_availableFeatureSet.find(VK_KHR_RAY_QUERY_EXTENSION_NAME) != m_availableFeatureSet.end())
                m_features.rayQuery = rayQueryFeatures.rayQuery;
            
            /* AccelerationStructureFeaturesKHR */                          // AccelerationStructure
            if (m_availableFeatureSet.find(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) != m_availableFeatureSet.end())
            {
                m_features.accelerationStructure = accelerationFeatures.accelerationStructure;
                m_features.accelerationStructureCaptureReplay = accelerationFeatures.accelerationStructureCaptureReplay;
                m_features.accelerationStructureIndirectBuild = accelerationFeatures.accelerationStructureIndirectBuild;
                m_features.accelerationStructureHostCommands = accelerationFeatures.accelerationStructureHostCommands;
                m_features.descriptorBindingAccelerationStructureUpdateAfterBind = accelerationFeatures.descriptorBindingAccelerationStructureUpdateAfterBind;
            }
            
            /* RayTracingPipelineFeaturesKHR */
            if (m_availableFeatureSet.find(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) != m_availableFeatureSet.end())
            {
                m_features.rayTracingPipeline = rayTracingPipelineFeatures.rayTracingPipeline;
                m_features.rayTracingPipelineShaderGroupHandleCaptureReplay = rayTracingPipelineFeatures.rayTracingPipelineShaderGroupHandleCaptureReplay;
                m_features.rayTracingPipelineShaderGroupHandleCaptureReplayMixed = rayTracingPipelineFeatures.rayTracingPipelineShaderGroupHandleCaptureReplayMixed;
                m_features.rayTracingPipelineTraceRaysIndirect = rayTracingPipelineFeatures.rayTracingPipelineTraceRaysIndirect;
                m_features.rayTraversalPrimitiveCulling = rayTracingPipelineFeatures.rayTraversalPrimitiveCulling;
            }
            
            /* FragmentShaderInterlockFeaturesEXT */
            if (m_availableFeatureSet.find(VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME) != m_availableFeatureSet.end())
            {
                m_features.fragmentShaderPixelInterlock = fragmentShaderInterlockFeatures.fragmentShaderPixelInterlock;
                m_features.fragmentShaderSampleInterlock = fragmentShaderInterlockFeatures.fragmentShaderSampleInterlock;
                m_features.fragmentShaderShadingRateInterlock = fragmentShaderInterlockFeatures.fragmentShaderShadingRateInterlock;
            }
            
            /* BufferDeviceAddressFeaturesKHR */
            if (m_availableFeatureSet.find(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) != m_availableFeatureSet.end())
            {
                m_features.bufferDeviceAddress = bufferDeviceAddressFeatures.bufferDeviceAddress;
            }
            
            m_features.dispatchBase = true;
        }

        // Get physical device's memory properties
        {
            m_memoryProperties = {};
            VkPhysicalDeviceMemoryProperties vk_physicalDeviceMemoryProperties;
            vkGetPhysicalDeviceMemoryProperties(vk_physicalDevice, &vk_physicalDeviceMemoryProperties);
            m_memoryProperties.memoryTypeCount = vk_physicalDeviceMemoryProperties.memoryTypeCount;
            for(uint32_t i = 0; i < m_memoryProperties.memoryTypeCount; ++i)
            {
                m_memoryProperties.memoryTypes[i].heapIndex = vk_physicalDeviceMemoryProperties.memoryTypes[i].heapIndex;
                m_memoryProperties.memoryTypes[i].propertyFlags = getMemoryPropertyFlagsFromVkMemoryPropertyFlags(vk_physicalDeviceMemoryProperties.memoryTypes[i].propertyFlags);
            }
            m_memoryProperties.memoryHeapCount = vk_physicalDeviceMemoryProperties.memoryHeapCount;
            for(uint32_t i = 0; i < m_memoryProperties.memoryHeapCount; ++i)
            {
                m_memoryProperties.memoryHeaps[i].size = vk_physicalDeviceMemoryProperties.memoryHeaps[i].size;
                m_memoryProperties.memoryHeaps[i].flags = core::bitflag<IDeviceMemoryAllocation::E_MEMORY_HEAP_FLAGS>(vk_physicalDeviceMemoryProperties.memoryHeaps[i].flags);
            }
        }
                
        uint32_t qfamCount = 0u;
        vkGetPhysicalDeviceQueueFamilyProperties(m_vkPhysicalDevice, &qfamCount, nullptr);
        core::vector<VkQueueFamilyProperties> qfamprops(qfamCount);
        vkGetPhysicalDeviceQueueFamilyProperties(m_vkPhysicalDevice, &qfamCount, qfamprops.data());

        m_qfamProperties = core::make_refctd_dynamic_array<qfam_props_array_t>(qfamCount);
        for (uint32_t i = 0u; i < qfamCount; ++i)
        {
            const auto& vkqf = qfamprops[i];
            auto& qf = (*m_qfamProperties)[i];
                    
            qf.queueCount = vkqf.queueCount;
            qf.queueFlags = static_cast<E_QUEUE_FLAGS>(vkqf.queueFlags);
            qf.timestampValidBits = vkqf.timestampValidBits;
            qf.minImageTransferGranularity = { vkqf.minImageTransferGranularity.width, vkqf.minImageTransferGranularity.height, vkqf.minImageTransferGranularity.depth };
        }

        std::ostringstream pool;
        bool runningRDoc = (m_rdoc_api != nullptr);
        addCommonGLSLDefines(pool,runningRDoc);
        finalizeGLSLDefinePool(std::move(pool));
    }
            
    inline VkPhysicalDevice getInternalObject() const { return m_vkPhysicalDevice; }
            
    E_API_TYPE getAPIType() const override { return EAT_VULKAN; }

    IDebugCallback* getDebugCallback() override { return m_api->getDebugCallback(); }

    bool isSwapchainSupported() const override
    {
        return isExtensionSupported(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }

    const SFormatImageUsage& getImageFormatUsagesLinear(const asset::E_FORMAT format) override
    {
        if (m_linearTilingUsages[format].isInitialized)
            return m_linearTilingUsages[format];

        VkFormatProperties vk_formatProps;
        vkGetPhysicalDeviceFormatProperties(m_vkPhysicalDevice, getVkFormatFromFormat(format),
            &vk_formatProps);

        const VkFormatFeatureFlags vk_formatFeatures = vk_formatProps.linearTilingFeatures;

        m_linearTilingUsages[format].sampledImage = (vk_formatFeatures & (VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) ? 1 : 0;
        m_linearTilingUsages[format].storageImage = (vk_formatFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) ? 1 : 0;
        m_linearTilingUsages[format].storageImageAtomic = (vk_formatFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT) ? 1 : 0;
        m_linearTilingUsages[format].attachment = (vk_formatFeatures & (VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) ? 1 : 0;
        m_linearTilingUsages[format].attachmentBlend = (vk_formatFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT) ? 1 : 0;
        m_linearTilingUsages[format].blitSrc = (vk_formatFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) ? 1 : 0;
        m_linearTilingUsages[format].blitDst = (vk_formatFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT) ? 1 : 0;
        m_linearTilingUsages[format].transferSrc = (vk_formatFeatures & VK_FORMAT_FEATURE_TRANSFER_SRC_BIT) ? 1 : 0;
        m_linearTilingUsages[format].transferDst = (vk_formatFeatures & VK_FORMAT_FEATURE_TRANSFER_DST_BIT) ? 1 : 0;
        // m_linearTilingUsages[format].log2MaxSmples = ; // Todo(achal)

        m_linearTilingUsages[format].isInitialized = 1;

        return m_linearTilingUsages[format];
    }

    const SFormatImageUsage& getImageFormatUsagesOptimal(const asset::E_FORMAT format) override
    {
        if (m_optimalTilingUsages[format].isInitialized)
            return m_optimalTilingUsages[format];

        VkFormatProperties vk_formatProps;
        vkGetPhysicalDeviceFormatProperties(m_vkPhysicalDevice, getVkFormatFromFormat(format),
            &vk_formatProps);

        const VkFormatFeatureFlags vk_formatFeatures = vk_formatProps.optimalTilingFeatures;

        m_optimalTilingUsages[format].sampledImage = vk_formatFeatures & (VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) ? 1 : 0;
        m_optimalTilingUsages[format].storageImage = vk_formatFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT ? 1 : 0;
        m_optimalTilingUsages[format].storageImageAtomic = vk_formatFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT ? 1 : 0;
        m_optimalTilingUsages[format].attachment = vk_formatFeatures & (VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) ? 1 : 0;
        m_optimalTilingUsages[format].attachmentBlend = vk_formatFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT ? 1 : 0;
        m_optimalTilingUsages[format].blitSrc = vk_formatFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT ? 1 : 0;
        m_optimalTilingUsages[format].blitDst = vk_formatFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT ? 1 : 0;
        m_optimalTilingUsages[format].transferSrc = vk_formatFeatures & VK_FORMAT_FEATURE_TRANSFER_SRC_BIT ? 1 : 0;
        m_optimalTilingUsages[format].transferDst = vk_formatFeatures & VK_FORMAT_FEATURE_TRANSFER_DST_BIT ? 1 : 0;
        // m_optimalTilingUsages[format].log2MaxSmples = ; // Todo(achal)

        m_optimalTilingUsages[format].isInitialized = 1;

        return m_optimalTilingUsages[format];
    }

    const SFormatBufferUsage& getBufferFormatUsages(const asset::E_FORMAT format) override
    {
        if (m_bufferUsages[format].isInitialized)
            return m_bufferUsages[format];

        VkFormatProperties vk_formatProps;
        vkGetPhysicalDeviceFormatProperties(m_vkPhysicalDevice, getVkFormatFromFormat(format),
            &vk_formatProps);

        const VkFormatFeatureFlags vk_formatFeatures = vk_formatProps.bufferFeatures;

        m_bufferUsages[format].vertexAttribute = (vk_formatProps.bufferFeatures & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT) ? 1 : 0;
        m_bufferUsages[format].bufferView = (vk_formatProps.bufferFeatures & VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT) ? 1 : 0;
        m_bufferUsages[format].storageBufferView = (vk_formatProps.bufferFeatures & VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT) ? 1 : 0;
        m_bufferUsages[format].storageBufferViewAtomic = (vk_formatProps.bufferFeatures & VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_ATOMIC_BIT) ? 1 : 0;
        m_bufferUsages[format].accelerationStructureVertex = (vk_formatProps.bufferFeatures & VK_FORMAT_FEATURE_ACCELERATION_STRUCTURE_VERTEX_BUFFER_BIT_KHR) ? 1 : 0;

        m_bufferUsages[format].isInitialized = 1;

        return m_bufferUsages[format];
    }
            
protected:
    core::smart_refctd_ptr<ILogicalDevice> createLogicalDevice_impl(const ILogicalDevice::SCreationParams& params) override
    {
        auto insertFeatureIfAvailable = [this](const ILogicalDevice::E_FEATURE feature, auto& featureSet) -> bool
        {
            constexpr uint32_t MAX_COUNT = 1 << 12u;
            ILogicalDevice::E_FEATURE requiredFeatures[MAX_COUNT];
            uint32_t requiredFeatureCount = 0u;
            CVulkanLogicalDevice::getRequiredFeatures(feature, requiredFeatureCount, requiredFeatures);
            assert(requiredFeatureCount <= MAX_COUNT);

            const char* vulkanNames[MAX_COUNT] = {};
            for (uint32_t i = 0u; i < requiredFeatureCount; ++i)
            {
                vulkanNames[i] = CVulkanLogicalDevice::getVulkanExtensionName(requiredFeatures[i]);
                if (m_availableFeatureSet.find(vulkanNames[i]) == m_availableFeatureSet.end())
                    return false;
            }

            featureSet.insert(vulkanNames, vulkanNames + requiredFeatureCount);
            return true;
        };

        core::unordered_set<core::string> selectedFeatureSet;
        for (uint32_t i = 0u; i < params.requiredFeatureCount; ++i)
        {
            if (!insertFeatureIfAvailable(params.requiredFeatures[i], selectedFeatureSet))
                return nullptr;
        }

        for (uint32_t i = 0u; i < params.optionalFeatureCount; ++i)
        {
            if (!insertFeatureIfAvailable(params.optionalFeatures[i], selectedFeatureSet))
                continue;
        }
                    
        if (m_availableFeatureSet.find(VK_KHR_SPIRV_1_4_EXTENSION_NAME) != m_availableFeatureSet.end() && (m_properties.limits.spirvVersion < asset::IGLSLCompiler::ESV_1_4))
        {
            m_properties.limits.spirvVersion = asset::IGLSLCompiler::ESV_1_4;
            selectedFeatureSet.insert(VK_KHR_SPIRV_1_4_EXTENSION_NAME);
        }

        core::vector<const char*> selectedFeatures(selectedFeatureSet.size());
        {
            uint32_t i = 0u;
            for (const auto& feature : selectedFeatureSet)
                selectedFeatures[i++] = feature.c_str();
        }

        void * firstFeatureInChain = nullptr;

        // Vulkan has problems with having features in the feature chain that have values set to false.
        // For example having an empty "RayTracingPipelineFeaturesKHR" in the chain will lead to validation errors for RayQueryONLY applications.
        auto addFeatureToChain = [&firstFeatureInChain](void* feature) -> void
        {
            struct VulkanStructHeader
            {
                VkStructureType    sType;
                void*              pNext;
            };
            VulkanStructHeader* enabledExtStructPtr = reinterpret_cast<VulkanStructHeader*>(feature);
            enabledExtStructPtr->pNext = firstFeatureInChain;
            firstFeatureInChain = feature;
        };

        VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
        if (selectedFeatureSet.find(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) != selectedFeatureSet.end())
        {
            accelerationStructureFeatures.accelerationStructure = m_features.accelerationStructure;
            accelerationStructureFeatures.accelerationStructureCaptureReplay = m_features.accelerationStructureCaptureReplay;
            accelerationStructureFeatures.accelerationStructureIndirectBuild = m_features.accelerationStructureIndirectBuild;
            accelerationStructureFeatures.accelerationStructureHostCommands = m_features.accelerationStructureHostCommands;
            accelerationStructureFeatures.descriptorBindingAccelerationStructureUpdateAfterBind = m_features.descriptorBindingAccelerationStructureUpdateAfterBind;
            addFeatureToChain(&accelerationStructureFeatures);
        }

        VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipelineFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
        if (selectedFeatureSet.find(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) != selectedFeatureSet.end())
        {
            rayTracingPipelineFeatures.rayTracingPipeline = m_features.rayTracingPipeline;
            rayTracingPipelineFeatures.rayTracingPipelineShaderGroupHandleCaptureReplay = m_features.rayTracingPipelineShaderGroupHandleCaptureReplay;
            rayTracingPipelineFeatures.rayTracingPipelineShaderGroupHandleCaptureReplayMixed = m_features.rayTracingPipelineShaderGroupHandleCaptureReplayMixed;
            rayTracingPipelineFeatures.rayTracingPipelineTraceRaysIndirect = m_features.rayTracingPipelineTraceRaysIndirect;
            rayTracingPipelineFeatures.rayTraversalPrimitiveCulling = m_features.rayTraversalPrimitiveCulling;
            addFeatureToChain(&rayTracingPipelineFeatures);
        }

        VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
        if (selectedFeatureSet.find(VK_KHR_RAY_QUERY_EXTENSION_NAME) != selectedFeatureSet.end())
        {
            rayQueryFeatures.rayQuery = m_features.rayQuery;
            addFeatureToChain(&rayQueryFeatures);
        }
        
        VkPhysicalDeviceBufferDeviceAddressFeaturesKHR bufferDeviceAddressFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_KHR};
        if (selectedFeatureSet.find(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) != selectedFeatureSet.end())
        {
            bufferDeviceAddressFeatures.bufferDeviceAddress = m_features.bufferDeviceAddress;
            addFeatureToChain(&bufferDeviceAddressFeatures);
        }

        VkPhysicalDeviceFeatures2 vk_deviceFeatures2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
        vk_deviceFeatures2.pNext = firstFeatureInChain;
        vk_deviceFeatures2.features = {};
        // Enabled Logical Device Features By Default:
        // TODO: seperate logical and physical device features.
        vk_deviceFeatures2.features.samplerAnisotropy = m_features.samplerAnisotropy;
        vk_deviceFeatures2.features.inheritedQueries = m_features.inheritedQueries;
        vk_deviceFeatures2.features.geometryShader = m_features.geometryShader;

        // Create Device
        VkDeviceCreateInfo vk_createInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
        vk_createInfo.pNext = &vk_deviceFeatures2; // Vulkan >= 1.1 Device uses createInfo.pNext to use features
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

        vk_createInfo.enabledExtensionCount = static_cast<uint32_t>(selectedFeatures.size());
        vk_createInfo.ppEnabledExtensionNames = selectedFeatures.data();
        
        VkDevice vk_device = VK_NULL_HANDLE;
        if (vkCreateDevice(m_vkPhysicalDevice, &vk_createInfo, nullptr, &vk_device) == VK_SUCCESS)
        {
            return core::make_smart_refctd_ptr<CVulkanLogicalDevice>(core::smart_refctd_ptr<IAPIConnection>(m_api),m_rdoc_api,this,vk_device,m_vkInstance,params);
        }
        else
        {
            return nullptr;
        }    
    }

    inline bool isExtensionSupported(const char* name) const
    {
        return m_availableFeatureSet.find(name) != m_availableFeatureSet.end();
    }

private:
    IAPIConnection* m_api; // purposefully not refcounted to avoid circular ref
    renderdoc_api_t* m_rdoc_api;
    VkInstance m_vkInstance;
    VkPhysicalDevice m_vkPhysicalDevice;
    core::unordered_set<std::string> m_availableFeatureSet;
};
        
}

#endif