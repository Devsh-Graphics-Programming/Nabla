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
        VkPhysicalDeviceDriverProperties driverProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES };
        VkPhysicalDeviceIDProperties deviceIDProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES, &driverProperties };
        VkPhysicalDeviceSubgroupProperties subgroupProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES, &deviceIDProperties };
        VkPhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingPipelineProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR, &subgroupProperties };
        VkPhysicalDeviceAccelerationStructurePropertiesKHR accelerationStructureProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR, &rayTracingPipelineProperties };
        {
            VkPhysicalDeviceProperties2 deviceProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
            deviceProperties.pNext = &accelerationStructureProperties;
            vkGetPhysicalDeviceProperties2(m_vkPhysicalDevice, &deviceProperties);
            
            memcpy(m_properties.deviceUUID, deviceIDProperties.deviceUUID, VK_UUID_SIZE);
            m_properties.deviceType = static_cast<E_TYPE>(deviceProperties.properties.deviceType);
            m_properties.driverID = static_cast<E_DRIVER_ID>(driverProperties.driverID);
            // TODO fill m_properties
                    
            m_properties.limits.bufferViewAlignment = deviceProperties.properties.limits.minTexelBufferOffsetAlignment;
            m_properties.limits.UBOAlignment = deviceProperties.properties.limits.minUniformBufferOffsetAlignment;
            m_properties.limits.SSBOAlignment = deviceProperties.properties.limits.minStorageBufferOffsetAlignment;
            m_properties.limits.maxSamplerAnisotropyLog2 = std::log2(deviceProperties.properties.limits.maxSamplerAnisotropy);
            
            m_properties.limits.timestampPeriodInNanoSeconds = deviceProperties.properties.limits.timestampPeriod;
            
            m_properties.limits.maxBufferViewSizeTexels = deviceProperties.properties.limits.maxTexelBufferElements;
            m_properties.limits.maxUBOSize = deviceProperties.properties.limits.maxUniformBufferRange;
            m_properties.limits.maxSSBOSize = deviceProperties.properties.limits.maxStorageBufferRange;
            m_properties.limits.maxBufferSize = core::max(m_properties.limits.maxUBOSize, m_properties.limits.maxSSBOSize);
                    
            m_properties.limits.maxPerStageDescriptorSSBOs = deviceProperties.properties.limits.maxPerStageDescriptorStorageBuffers;
                    
            m_properties.limits.maxDescriptorSetSSBOs = deviceProperties.properties.limits.maxDescriptorSetStorageBuffers;
            m_properties.limits.maxDescriptorSetUBOs = deviceProperties.properties.limits.maxDescriptorSetUniformBuffers;
            m_properties.limits.maxDescriptorSetDynamicOffsetSSBOs = deviceProperties.properties.limits.maxDescriptorSetStorageBuffersDynamic;
            m_properties.limits.maxDescriptorSetDynamicOffsetUBOs = deviceProperties.properties.limits.maxDescriptorSetUniformBuffersDynamic;
            m_properties.limits.maxDescriptorSetImages = deviceProperties.properties.limits.maxDescriptorSetSampledImages;
            m_properties.limits.maxDescriptorSetStorageImages = deviceProperties.properties.limits.maxDescriptorSetStorageImages;
                    
            m_properties.limits.pointSizeRange[0] = deviceProperties.properties.limits.pointSizeRange[0];
            m_properties.limits.pointSizeRange[1] = deviceProperties.properties.limits.pointSizeRange[1];
            m_properties.limits.lineWidthRange[0] = deviceProperties.properties.limits.lineWidthRange[0];
            m_properties.limits.lineWidthRange[1] = deviceProperties.properties.limits.lineWidthRange[1];

            m_properties.limits.maxViewports = deviceProperties.properties.limits.maxViewports;
            m_properties.limits.maxViewportDims[0] = deviceProperties.properties.limits.maxViewportDimensions[0];
            m_properties.limits.maxViewportDims[1] = deviceProperties.properties.limits.maxViewportDimensions[1];
            
            m_properties.limits.maxComputeSharedMemorySize = deviceProperties.properties.limits.maxComputeSharedMemorySize;
                    
            m_properties.limits.maxWorkgroupSize[0] = deviceProperties.properties.limits.maxComputeWorkGroupSize[0];
            m_properties.limits.maxWorkgroupSize[1] = deviceProperties.properties.limits.maxComputeWorkGroupSize[1];
            m_properties.limits.maxWorkgroupSize[2] = deviceProperties.properties.limits.maxComputeWorkGroupSize[2];
                    
            m_properties.limits.subgroupSize = subgroupProperties.subgroupSize;
            m_properties.limits.subgroupOpsShaderStages = static_cast<asset::IShader::E_SHADER_STAGE>(subgroupProperties.supportedStages);

            m_properties.limits.nonCoherentAtomSize = deviceProperties.properties.limits.nonCoherentAtomSize;
            
            m_properties.limits.maxOptimallyResidentWorkgroupInvocations = core::min(core::roundDownToPoT(deviceProperties.properties.limits.maxComputeWorkGroupInvocations),512u);
            constexpr auto beefyGPUWorkgroupMaxOccupancy = 256u; // TODO: find a way to query and report this somehow, persistent threads are very useful!
            m_properties.limits.maxResidentInvocations = beefyGPUWorkgroupMaxOccupancy*m_properties.limits.maxOptimallyResidentWorkgroupInvocations;

            /*
                [NO NABALA SUPPORT] Vulkan 1.0 implementation must support the 1.0 version of SPIR-V and the 1.0 version of the SPIR-V Extended Instructions for GLSL. If the VK_KHR_spirv_1_4 extension is enabled, the implementation must additionally support the 1.4 version of SPIR-V.
                A Vulkan 1.1 implementation must support the 1.0, 1.1, 1.2, and 1.3 versions of SPIR-V and the 1.0 version of the SPIR-V Extended Instructions for GLSL.
                A Vulkan 1.2 implementation must support the 1.0, 1.1, 1.2, 1.3, 1.4, and 1.5 versions of SPIR-V and the 1.0 version of the SPIR-V Extended Instructions for GLSL.
                A Vulkan 1.3 implementation must support the 1.0, 1.1, 1.2, 1.3, 1.4, and 1.5 versions of SPIR-V and the 1.0 version of the SPIR-V Extended Instructions for GLSL.
            */
            
            uint32_t apiVersion = std::min(instanceApiVersion, deviceProperties.properties.apiVersion);
            assert(apiVersion >= MinimumVulkanApiVersion);
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

            m_properties.apiVersion.major = VK_API_VERSION_MAJOR(apiVersion);
            m_properties.apiVersion.minor = VK_API_VERSION_MINOR(apiVersion);
            m_properties.apiVersion.patch = VK_API_VERSION_PATCH(apiVersion);

            // AccelerationStructure
            if (m_availableFeatureSet.find(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) != m_availableFeatureSet.end())
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

            // RayTracingPipeline
            if (m_availableFeatureSet.find(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) != m_availableFeatureSet.end())
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
        }
        
        // Get physical device's features
        VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipelineFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR };
        VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR, &rayTracingPipelineFeatures };
        VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR, &accelerationFeatures };
        VkPhysicalDeviceBufferDeviceAddressFeaturesKHR bufferDeviceAddressFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_KHR, &rayQueryFeatures };
        VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT fragmentShaderInterlockFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT, &bufferDeviceAddressFeatures };
        {
            VkPhysicalDeviceFeatures2 deviceFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
            deviceFeatures.pNext = &fragmentShaderInterlockFeatures;
            vkGetPhysicalDeviceFeatures2(m_vkPhysicalDevice, &deviceFeatures);
            const auto& features = deviceFeatures.features;
                    
            m_features.robustBufferAccess = features.robustBufferAccess;
            m_features.imageCubeArray = features.imageCubeArray;
            m_features.logicOp = features.logicOp;
            m_features.multiDrawIndirect = features.multiDrawIndirect;
            m_features.samplerAnisotropy = features.samplerAnisotropy;
            m_features.multiViewport = features.multiViewport;
            m_features.vertexAttributeDouble = features.shaderFloat64;
            m_features.dispatchBase = false; // Todo(achal): Umm.. what is this? Whether you can call VkCmdDispatchBase with non zero base args
            m_features.shaderSubgroupBasic = subgroupProperties.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT;
            m_features.shaderSubgroupVote = subgroupProperties.supportedOperations & VK_SUBGROUP_FEATURE_VOTE_BIT;
            m_features.shaderSubgroupArithmetic = subgroupProperties.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
            m_features.shaderSubgroupBallot = subgroupProperties.supportedOperations & VK_SUBGROUP_FEATURE_BALLOT_BIT;
            m_features.shaderSubgroupShuffle = subgroupProperties.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
            m_features.shaderSubgroupShuffleRelative = subgroupProperties.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT;
            m_features.shaderSubgroupClustered = subgroupProperties.supportedOperations & VK_SUBGROUP_FEATURE_CLUSTERED_BIT;
            m_features.shaderSubgroupQuad = subgroupProperties.supportedOperations & VK_SUBGROUP_FEATURE_QUAD_BIT;
            m_features.shaderSubgroupQuadAllStages = ((subgroupProperties.supportedStages & asset::IShader::E_SHADER_STAGE::ESS_ALL)
                                                        == asset::IShader::E_SHADER_STAGE::ESS_ALL);
            m_features.inheritedQueries = features.inheritedQueries;
            m_features.geometryShader = features.geometryShader;
            // RayQuery
            if (m_availableFeatureSet.find(VK_KHR_RAY_QUERY_EXTENSION_NAME) != m_availableFeatureSet.end())
                m_features.rayQuery = rayQueryFeatures.rayQuery;
            m_features.allowCommandBufferQueryCopies = true; // always true in vk for all query types instead of PerformanceQuery which we don't support at the moment (have VkPhysicalDevicePerformanceQueryPropertiesKHR::allowCommandBufferQueryCopies in mind)
            
                                                             // AccelerationStructure
            if (m_availableFeatureSet.find(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) != m_availableFeatureSet.end())
            {
                m_features.accelerationStructure = accelerationFeatures.accelerationStructure;
                m_features.accelerationStructureCaptureReplay = accelerationFeatures.accelerationStructureCaptureReplay;
                m_features.accelerationStructureIndirectBuild = accelerationFeatures.accelerationStructureIndirectBuild;
                m_features.accelerationStructureHostCommands = accelerationFeatures.accelerationStructureHostCommands;
                m_features.descriptorBindingAccelerationStructureUpdateAfterBind = accelerationFeatures.descriptorBindingAccelerationStructureUpdateAfterBind;
            }

            // RayTracingPipeline
            if (m_availableFeatureSet.find(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) != m_availableFeatureSet.end())
            {
                m_features.rayTracingPipeline = rayTracingPipelineFeatures.rayTracingPipeline;
                m_features.rayTracingPipelineShaderGroupHandleCaptureReplay = rayTracingPipelineFeatures.rayTracingPipelineShaderGroupHandleCaptureReplay;
                m_features.rayTracingPipelineShaderGroupHandleCaptureReplayMixed = rayTracingPipelineFeatures.rayTracingPipelineShaderGroupHandleCaptureReplayMixed;
                m_features.rayTracingPipelineTraceRaysIndirect = rayTracingPipelineFeatures.rayTracingPipelineTraceRaysIndirect;
                m_features.rayTraversalPrimitiveCulling = rayTracingPipelineFeatures.rayTraversalPrimitiveCulling;
            }

            // Buffer Device Address
            if (m_availableFeatureSet.find(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) != m_availableFeatureSet.end())
            {
                m_features.bufferDeviceAddress = bufferDeviceAddressFeatures.bufferDeviceAddress;
                // bufferDeviceAddressFeatures.bufferDeviceAddress;
                // bufferDeviceAddressFeatures.bufferDeviceAddress;
            }
            
            // FragmentShaderInterlock
            if (m_availableFeatureSet.find(VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME) != m_availableFeatureSet.end())
            {
                m_features.fragmentShaderPixelInterlock = fragmentShaderInterlockFeatures.fragmentShaderPixelInterlock;
                m_features.fragmentShaderSampleInterlock = fragmentShaderInterlockFeatures.fragmentShaderSampleInterlock;
                m_features.fragmentShaderShadingRateInterlock = fragmentShaderInterlockFeatures.fragmentShaderShadingRateInterlock;
            }
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