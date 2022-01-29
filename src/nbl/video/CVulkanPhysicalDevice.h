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
    CVulkanPhysicalDevice(core::smart_refctd_ptr<system::ISystem>&& sys, core::smart_refctd_ptr<asset::IGLSLCompiler>&& glslc, IAPIConnection* api, renderdoc_api_t* rdoc, VkPhysicalDevice vk_physicalDevice, VkInstance vk_instance)
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
        VkPhysicalDeviceSubgroupProperties subgroupProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES };
        VkPhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingPipelineProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR, &subgroupProperties };
        VkPhysicalDeviceAccelerationStructurePropertiesKHR accelerationStructureProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR, &rayTracingPipelineProperties };
        {
            VkPhysicalDeviceProperties2 deviceProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
            deviceProperties.pNext = &accelerationStructureProperties;
            vkGetPhysicalDeviceProperties2(m_vkPhysicalDevice, &deviceProperties);
                    
            // TODO fill m_properties
                    
            m_limits.UBOAlignment = deviceProperties.properties.limits.minUniformBufferOffsetAlignment;
            m_limits.SSBOAlignment = deviceProperties.properties.limits.minStorageBufferOffsetAlignment;
            m_limits.bufferViewAlignment = deviceProperties.properties.limits.minTexelBufferOffsetAlignment;
            m_limits.maxSamplerAnisotropyLog2 = std::log2(deviceProperties.properties.limits.maxSamplerAnisotropy);

            m_limits.maxUBOSize = deviceProperties.properties.limits.maxUniformBufferRange;
            m_limits.maxSSBOSize = deviceProperties.properties.limits.maxStorageBufferRange;
            m_limits.maxBufferViewSizeTexels = deviceProperties.properties.limits.maxTexelBufferElements;
            m_limits.maxBufferSize = core::max(m_limits.maxUBOSize, m_limits.maxSSBOSize);
                    
            m_limits.maxPerStageSSBOs = deviceProperties.properties.limits.maxPerStageDescriptorStorageBuffers;
                    
            m_limits.maxSSBOs = deviceProperties.properties.limits.maxDescriptorSetStorageBuffers;
            m_limits.maxUBOs = deviceProperties.properties.limits.maxDescriptorSetUniformBuffers;
            m_limits.maxDynamicOffsetSSBOs = deviceProperties.properties.limits.maxDescriptorSetStorageBuffersDynamic;
            m_limits.maxDynamicOffsetUBOs = deviceProperties.properties.limits.maxDescriptorSetUniformBuffersDynamic;
            m_limits.maxTextures = deviceProperties.properties.limits.maxDescriptorSetSamplers;
            m_limits.maxStorageImages = deviceProperties.properties.limits.maxDescriptorSetStorageImages;
                    
            m_limits.pointSizeRange[0] = deviceProperties.properties.limits.pointSizeRange[0];
            m_limits.pointSizeRange[1] = deviceProperties.properties.limits.pointSizeRange[1];
            m_limits.lineWidthRange[0] = deviceProperties.properties.limits.lineWidthRange[0];
            m_limits.lineWidthRange[1] = deviceProperties.properties.limits.lineWidthRange[1];
                    
            m_limits.maxViewports = deviceProperties.properties.limits.maxViewports;
            m_limits.maxViewportDims[0] = deviceProperties.properties.limits.maxViewportDimensions[0];
            m_limits.maxViewportDims[1] = deviceProperties.properties.limits.maxViewportDimensions[1];
                    
            m_limits.maxWorkgroupSize[0] = deviceProperties.properties.limits.maxComputeWorkGroupSize[0];
            m_limits.maxWorkgroupSize[1] = deviceProperties.properties.limits.maxComputeWorkGroupSize[1];
            m_limits.maxWorkgroupSize[2] = deviceProperties.properties.limits.maxComputeWorkGroupSize[2];
                    
            m_limits.subgroupSize = subgroupProperties.subgroupSize;
            m_limits.subgroupOpsShaderStages = static_cast<asset::IShader::E_SHADER_STAGE>(subgroupProperties.supportedStages);

            m_limits.nonCoherentAtomSize = deviceProperties.properties.limits.nonCoherentAtomSize;
            
            m_limits.maxOptimallyResidentWorkgroupInvocations = core::min(core::roundDownToPoT(deviceProperties.properties.limits.maxComputeWorkGroupInvocations),512u);
            constexpr auto beefyGPUWorkgroupMaxOccupancy = 256u; // TODO: find a way to query and report this somehow, persistent threads are very useful!
            m_limits.maxResidentInvocations = beefyGPUWorkgroupMaxOccupancy*m_limits.maxOptimallyResidentWorkgroupInvocations;

            m_limits.spirvVersion = asset::IGLSLCompiler::ESV_1_3;
#if 0
            switch (VK_API_VERSION_MINOR(deviceProperties.properties.apiVersion))
            {
            case 0:
                m_limits.spirvVersion = asset::IGLSLCompiler::ESV_1_0; break;
            case 1:
                m_limits.spirvVersion = asset::IGLSLCompiler::ESV_1_3; break;
            case 2:
                m_limits.spirvVersion = asset::IGLSLCompiler::ESV_1_5; break;
            default:
                _NBL_DEBUG_BREAK_IF("Invalid Vulkan minor version!");
                break;
            }
#endif

            // AccelerationStructure
            if (m_availableFeatureSet.find(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) != m_availableFeatureSet.end())
            {
                m_limits.maxGeometryCount = accelerationStructureProperties.maxGeometryCount;
                m_limits.maxInstanceCount = accelerationStructureProperties.maxInstanceCount;
                m_limits.maxPrimitiveCount = accelerationStructureProperties.maxPrimitiveCount;
                m_limits.maxPerStageDescriptorAccelerationStructures = accelerationStructureProperties.maxPerStageDescriptorAccelerationStructures;
                m_limits.maxPerStageDescriptorUpdateAfterBindAccelerationStructures = accelerationStructureProperties.maxPerStageDescriptorUpdateAfterBindAccelerationStructures;
                m_limits.maxDescriptorSetAccelerationStructures = accelerationStructureProperties.maxDescriptorSetAccelerationStructures;
                m_limits.maxDescriptorSetUpdateAfterBindAccelerationStructures = accelerationStructureProperties.maxDescriptorSetUpdateAfterBindAccelerationStructures;
                m_limits.minAccelerationStructureScratchOffsetAlignment = accelerationStructureProperties.minAccelerationStructureScratchOffsetAlignment;
            }

            // RayTracingPipeline
            if (m_availableFeatureSet.find(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) != m_availableFeatureSet.end())
            {
                m_limits.shaderGroupHandleSize = rayTracingPipelineProperties.shaderGroupHandleSize;
                m_limits.maxRayRecursionDepth = rayTracingPipelineProperties.maxRayRecursionDepth;
                m_limits.maxShaderGroupStride = rayTracingPipelineProperties.maxShaderGroupStride;
                m_limits.shaderGroupBaseAlignment = rayTracingPipelineProperties.shaderGroupBaseAlignment;
                m_limits.shaderGroupHandleCaptureReplaySize = rayTracingPipelineProperties.shaderGroupHandleCaptureReplaySize;
                m_limits.maxRayDispatchInvocationCount = rayTracingPipelineProperties.maxRayDispatchInvocationCount;
                m_limits.shaderGroupHandleAlignment = rayTracingPipelineProperties.shaderGroupHandleAlignment;
                m_limits.maxRayHitAttributeSize = rayTracingPipelineProperties.maxRayHitAttributeSize;
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
            VkPhysicalDeviceMemoryProperties vk_physicalDeviceMemoryProperties;
            vkGetPhysicalDeviceMemoryProperties(vk_physicalDevice, &vk_physicalDeviceMemoryProperties);
            memcpy(&m_memoryProperties, &vk_physicalDeviceMemoryProperties, sizeof(VkPhysicalDeviceMemoryProperties));
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
        addCommonGLSLDefines(pool,false/*TODO: @achal detect if RenderDoc is running*/);
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

        if (selectedFeatureSet.find(VK_KHR_SPIRV_1_4_EXTENSION_NAME) != selectedFeatureSet.end()
            && (m_limits.spirvVersion < asset::IGLSLCompiler::ESV_1_4))
        {
            m_limits.spirvVersion = asset::IGLSLCompiler::ESV_1_4;
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