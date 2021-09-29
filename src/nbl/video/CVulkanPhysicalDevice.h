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
            uint32_t  count;
            // Get Count First and Resize
            VkResult res = vkEnumerateDeviceExtensionProperties(m_vkPhysicalDevice, nullptr, &count, nullptr);
            assert(VK_SUCCESS == res);
            supportedExtensions.resize(count); 
            // Now fill the Vector
            res = vkEnumerateDeviceExtensionProperties(m_vkPhysicalDevice, nullptr, &count, supportedExtensions.data());
            assert(VK_SUCCESS == res);
            supportedExtensions.resize(core::min(supportedExtensions.size(), size_t(count)));
        }

        // Get physical device's limits
        VkPhysicalDeviceSubgroupProperties subgroupProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES, nullptr };
        VkPhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingPipelineProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR, &subgroupProperties };
        VkPhysicalDeviceAccelerationStructurePropertiesKHR accelerationProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR, &rayTracingPipelineProperties };
        {
            VkPhysicalDeviceProperties2 deviceProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
            deviceProperties.pNext = &accelerationProperties;
            vkGetPhysicalDeviceProperties2(m_vkPhysicalDevice, &deviceProperties);
                    
            // TODO fill m_properties
                    
            m_limits.UBOAlignment = deviceProperties.properties.limits.minUniformBufferOffsetAlignment;
            m_limits.SSBOAlignment = deviceProperties.properties.limits.minStorageBufferOffsetAlignment;
            m_limits.bufferViewAlignment = deviceProperties.properties.limits.minTexelBufferOffsetAlignment;
                    
            m_limits.maxUBOSize = deviceProperties.properties.limits.maxUniformBufferRange;
            m_limits.maxSSBOSize = deviceProperties.properties.limits.maxStorageBufferRange;
            m_limits.maxBufferViewSizeTexels = deviceProperties.properties.limits.maxTexelBufferElements;
            m_limits.maxBufferSize = core::max(m_limits.maxUBOSize, m_limits.maxSSBOSize);
                    
            m_limits.maxPerStageSSBOs = deviceProperties.properties.limits.maxPerStageDescriptorStorageBuffers;
                    
            m_limits.maxSSBOs = deviceProperties.properties.limits.maxDescriptorSetStorageBuffers;
            m_limits.maxUBOs = deviceProperties.properties.limits.maxDescriptorSetUniformBuffers;
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
            m_limits.subgroupOpsShaderStages = static_cast<asset::ISpecializedShader::E_SHADER_STAGE>(subgroupProperties.supportedStages);

            m_limits.nonCoherentAtomSize = deviceProperties.properties.limits.nonCoherentAtomSize;
        }
        
        // Get physical device's features
        VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipelineFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR, nullptr };
        VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR, &rayTracingPipelineFeatures };
        VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR, &accelerationFeatures };
        {
            VkPhysicalDeviceFeatures2 deviceFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
            deviceFeatures.pNext = &rayQueryFeatures;
            vkGetPhysicalDeviceFeatures2(m_vkPhysicalDevice, &deviceFeatures);
            auto features = deviceFeatures.features;
                    
            m_features.robustBufferAccess = features.robustBufferAccess;
            m_features.imageCubeArray = features.imageCubeArray;
            m_features.logicOp = features.logicOp;
            m_features.multiDrawIndirect = features.multiDrawIndirect;
            m_features.multiViewport = features.multiViewport;
            m_features.vertexAttributeDouble = features.shaderFloat64;
            m_features.dispatchBase = false; // Todo(achal): Umm.. what is this?
            m_features.shaderSubgroupBasic = subgroupProperties.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT;
            m_features.shaderSubgroupVote = subgroupProperties.supportedOperations & VK_SUBGROUP_FEATURE_VOTE_BIT;
            m_features.shaderSubgroupArithmetic = subgroupProperties.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
            m_features.shaderSubgroupBallot = subgroupProperties.supportedOperations & VK_SUBGROUP_FEATURE_BALLOT_BIT;
            m_features.shaderSubgroupShuffle = subgroupProperties.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
            m_features.shaderSubgroupShuffleRelative = subgroupProperties.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT;
            m_features.shaderSubgroupClustered = subgroupProperties.supportedOperations & VK_SUBGROUP_FEATURE_CLUSTERED_BIT;
            m_features.shaderSubgroupQuad = subgroupProperties.supportedOperations & VK_SUBGROUP_FEATURE_QUAD_BIT;
            m_features.shaderSubgroupQuadAllStages = ((subgroupProperties.supportedStages & asset::ISpecializedShader::E_SHADER_STAGE::ESS_ALL)
                                                        == asset::ISpecializedShader::E_SHADER_STAGE::ESS_ALL);
            m_features.rayQuery = rayQueryFeatures.rayQuery;
            m_features.accelerationStructure = accelerationFeatures.accelerationStructure;
            m_features.accelerationStructureCaptureReplay = accelerationFeatures.accelerationStructureCaptureReplay;
            m_features.accelerationStructureIndirectBuild = accelerationFeatures.accelerationStructureIndirectBuild;
            m_features.accelerationStructureHostCommands = accelerationFeatures.accelerationStructureHostCommands;
            m_features.descriptorBindingAccelerationStructureUpdateAfterBind = accelerationFeatures.descriptorBindingAccelerationStructureUpdateAfterBind;
        }
                
        requestDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME, false);
        requestDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME, true); // requires vulkan 1.1
        requestDeviceExtension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME, true); // required by VK_KHR_acceleration_structure
        requestDeviceExtension(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME, true); // required by VK_KHR_acceleration_structure
        // requestDeviceExtension<VkPhysicalDeviceAccelerationStructureFeaturesKHR>(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, true, &accelerationFeatures);
        // requestDeviceExtension<VkPhysicalDeviceRayTracingPipelineFeaturesKHR>(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, true, &rayTracingPipelineFeatures);
        // requestDeviceExtension<VkPhysicalDeviceRayQueryFeaturesKHR>(VK_KHR_RAY_QUERY_EXTENSION_NAME, true, &rayQueryFeatures);

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
    }
            
    inline VkPhysicalDevice getInternalObject() const { return m_vkPhysicalDevice; }
            
    E_API_TYPE getAPIType() const override { return EAT_VULKAN; }

    IDebugCallback* getDebugCallback() override { return m_api->getDebugCallback(); }

    bool isSwapchainSupported() const override
    {
        return isExtensionSupported(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }

    SFormatProperties getFormatProperties(asset::E_FORMAT format) const override
    {
        SFormatProperties result;

        VkFormatProperties vk_formatProps;
        vkGetPhysicalDeviceFormatProperties(m_vkPhysicalDevice, getVkFormatFromFormat(format),
            &vk_formatProps);

        result.linearTilingFeatures = static_cast<asset::E_FORMAT_FEATURE>(vk_formatProps.linearTilingFeatures);
        result.optimalTilingFeatures = static_cast<asset::E_FORMAT_FEATURE>(vk_formatProps.optimalTilingFeatures);
        result.bufferFeatures = static_cast<asset::E_FORMAT_FEATURE>(vk_formatProps.bufferFeatures);

        return result;
    }
            
protected:
    core::smart_refctd_ptr<ILogicalDevice> createLogicalDevice_impl(const ILogicalDevice::SCreationParams& params) override
    {
        // Filter Requested Extensions based on Availability and constructs chain for features
        std::vector<std::string> filteredExtensions;
        void* firstFeatureInChain;
        getFilteredExtensions(filteredExtensions, firstFeatureInChain);
        
        // Get cstr's for vulkan input
        std::vector<const char*> filteredExtensionNames;
        for(const auto& it : filteredExtensions)
            filteredExtensionNames.push_back(it.c_str());

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

        vk_createInfo.enabledLayerCount = 1u; // deprecated and ignored param
        const char* validationLayerName[] = { "VK_LAYER_KHRONOS_validation" };
        vk_createInfo.ppEnabledLayerNames = validationLayerName; // deprecated and ignored param

        // Todo(achal): Need to get this from the user based on if its a headless compute or some presentation worthy workload
        vk_createInfo.enabledExtensionCount = filteredExtensionNames.size();
        vk_createInfo.ppEnabledExtensionNames = filteredExtensionNames.data();
        
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

    // Functions and Structs to work with extensions and add them 
    struct DeviceExtensionRequest 
    {
        struct FeatureHeader
        {
            VkStructureType           sType;
            void*                     pNext;
        };

        DeviceExtensionRequest(const char* extensionName, bool isOptional = false, std::vector<uint8_t>&& featureMemory = std::vector<uint8_t>())
            : name(extensionName), optional(isOptional), featureMem(std::move(featureMemory))
        { }

        using byte = unsigned char; 

        std::string             name;
        bool                    optional = false;
        std::vector<uint8_t>    featureMem = std::vector<uint8_t>(); // usefull when constructing features chain for vk_createInfo.pEnabledFeatures
        // TODO(Better Container for featureMem?)
    };

    std::vector<VkExtensionProperties> getSupportedExtensions(VkPhysicalDevice physicalDevice)
    {
        return supportedExtensions;
    }

    bool isExtensionSupported(const char* name) const {
        for (uint32_t i = 0u; i < supportedExtensions.size(); ++i)
        {
            if (strcmp(supportedExtensions[i].extensionName, name) == 0)
                return true;
        }
        return false;
    }

    void requestDeviceExtension(const char* name, bool optional = false)
    {
        deviceExtensionRequests.emplace_back(name, optional, std::vector<uint8_t>());
    }
    
    template<typename T>
    void requestDeviceExtension(const char* name, bool optional = false, const T* featureStruct = nullptr)
    {
        std::vector<uint8_t>    featureMem;
        auto ptr = reinterpret_cast<const uint8_t*>(featureStruct);
        featureMem.insert(featureMem.end(),ptr,ptr+sizeof(T));
        deviceExtensionRequests.emplace_back(name, optional, std::move(featureMem));
    }

    // Checks if Requested Extensions are Supported based on `is_optional` and `extensionName` and fills a chain for features
    void getFilteredExtensions(std::vector<std::string>& filteredExtensions, void*& firstFeatureInChain)
    {
        std::vector<void*> featureStructPtrs;

        for(uint32_t i = 0; i < deviceExtensionRequests.size(); ++i)
        {
            auto & request = deviceExtensionRequests[i];
            bool isSupported = isExtensionSupported(request.name.c_str());

            if(isSupported)
            {
                filteredExtensions.push_back(request.name);
                if(!request.featureMem.empty()) {
                    featureStructPtrs.push_back(request.featureMem.data());
                }
            }
            else if(request.optional == false)
            {
                assert(false && "Device Extension is not supported by PhysicalDevice");
            }
        }
        
               
        // Construct Chain for Device Features
        // ExtensionHeader is just something for reinterpret_cast
        struct ExtensionHeader 
        {
            VkStructureType sType;
            void*           pNext;
        };

        for(uint32_t i = 0; i < featureStructPtrs.size(); i++)
        {
            ExtensionHeader* header  = reinterpret_cast<ExtensionHeader*>(featureStructPtrs[i]);
            header->pNext = (i < featureStructPtrs.size() - 1) ? featureStructPtrs[i + 1] : nullptr;
        }
        if(featureStructPtrs.size() > 0)
            firstFeatureInChain = featureStructPtrs[0];
        else 
            firstFeatureInChain = nullptr;
    }

private:
    IAPIConnection* m_api; // purposefully not refcounted to avoid circular ref
    renderdoc_api_t* m_rdoc_api;
    VkPhysicalDevice m_vkPhysicalDevice;
    VkInstance m_vkInstance;

    std::vector<DeviceExtensionRequest> deviceExtensionRequests;
    std::vector<VkExtensionProperties> supportedExtensions;
};
        
}

#endif