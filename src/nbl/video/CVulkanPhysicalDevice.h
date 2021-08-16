#ifndef __NBL_C_VULKAN_PHYSICAL_DEVICE_H_INCLUDED__
#define __NBL_C_VULKAN_PHYSICAL_DEVICE_H_INCLUDED__

#include "nbl/video/IPhysicalDevice.h"
#include "nbl/video/CVKLogicalDevice.h"
#include "nbl/video/surface/CSurfaceVulkan.h"
#include "nbl/video/CVulkanCommon.h"

namespace nbl::video
{
        
class CVulkanPhysicalDevice final : public IPhysicalDevice
{
public:
    CVulkanPhysicalDevice(VkPhysicalDevice vk_physicalDevice, core::smart_refctd_ptr<system::ISystem>&& sys,
        core::smart_refctd_ptr<asset::IGLSLCompiler>&& glslc)
        : IPhysicalDevice(std::move(sys), std::move(glslc)), m_vkPhysicalDevice(vk_physicalDevice)
    {
        // Get physical device's limits
        VkPhysicalDeviceSubgroupProperties subgroupProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES };
        {
            VkPhysicalDeviceProperties2 deviceProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
            deviceProperties.pNext = &subgroupProperties;
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
            m_limits.subgroupOpsShaderStages = subgroupProperties.supportedStages;
        }
                
        // Get physical device's features
        {
            VkPhysicalDeviceFeatures features;
            vkGetPhysicalDeviceFeatures(m_vkPhysicalDevice, &features);
                    
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

    // Todo(achal)
    IDebugCallback* getDebugCallback() override { return nullptr; }

    void getAvailableFormatsForSurface(const ISurface* surface, uint32_t& formatCount,
        ISurface::SFormat* formats) const override
    {
        constexpr uint32_t MAX_SURFACE_FORMAT_COUNT = 1000u;

        if (surface->getAPIType() != EAT_VULKAN)
            return;

        // Todo(achal): not sure yet, how would I handle multiple platforms without making
        // this function templated
        VkSurfaceKHR vk_surface = static_cast<const CSurfaceVulkanWin32*>(surface)->getInternalObject();

        VkResult retval = vkGetPhysicalDeviceSurfaceFormatsKHR(m_vkPhysicalDevice, vk_surface,
            &formatCount, nullptr);

        // Todo(achal): Would there be a need to handle VK_INCOMPLETE separately?
        if ((retval != VK_SUCCESS) && (retval != VK_INCOMPLETE))
        {
            formatCount = 0u;
            return;
        }

        if (!formats)
            return;

        VkSurfaceFormatKHR vk_formats[MAX_SURFACE_FORMAT_COUNT];
        retval = vkGetPhysicalDeviceSurfaceFormatsKHR(m_vkPhysicalDevice, vk_surface,
            &formatCount, vk_formats);

        // Todo(achal): Would there be a need to handle VK_INCOMPLETE separately?
        if ((retval != VK_SUCCESS) && (retval != VK_INCOMPLETE))
        {
            formatCount = 0u;
            formats = nullptr;
            return;
        }

        for (uint32_t i = 0u; i < formatCount; ++i)
        {
            formats[i].format = getFormatFromVkFormat(vk_formats[i].format);
            formats[i].colorSpace = getColorSpaceFromVkColorSpaceKHR(vk_formats[i].colorSpace);
        }
    }
    
    ISurface::E_PRESENT_MODE getAvailablePresentModesForSurface(const ISurface* surface) const override
    {
        constexpr uint32_t MAX_PRESENT_MODE_COUNT = 4u;

        if (surface->getAPIType() != EAT_VULKAN)
            return ISurface::EPM_UNKNOWN;

        // Todo(achal): not sure yet, how would I handle multiple platforms without making
        // this function templated
        VkSurfaceKHR vk_surface = static_cast<const CSurfaceVulkanWin32*>(surface)->getInternalObject();

        uint32_t count = 0u;
        VkResult retval = vkGetPhysicalDeviceSurfacePresentModesKHR(m_vkPhysicalDevice, vk_surface,
            &count, nullptr);

        // Todo(achal): Would there be a need to handle VK_INCOMPLETE separately?
        if ((retval != VK_SUCCESS) && (retval != VK_INCOMPLETE))
            return ISurface::EPM_UNKNOWN;

        assert(count <= MAX_PRESENT_MODE_COUNT);

        VkPresentModeKHR vk_presentModes[MAX_PRESENT_MODE_COUNT];
        retval = vkGetPhysicalDeviceSurfacePresentModesKHR(m_vkPhysicalDevice, vk_surface,
            &count, vk_presentModes);

        // Todo(achal): Would there be a need to handle VK_INCOMPLETE separately?
        if ((retval != VK_SUCCESS) && (retval != VK_INCOMPLETE))
            return ISurface::EPM_UNKNOWN;

        ISurface::E_PRESENT_MODE result = static_cast<ISurface::E_PRESENT_MODE>(0);

        for (uint32_t i = 0u; i < count; ++i)
            result = static_cast<ISurface::E_PRESENT_MODE>(result | getPresentModeFromVkPresentModeKHR(vk_presentModes[i]));

        return result;
    }

    virtual bool getSurfaceCapabilities(const ISurface* surface, ISurface::SCapabilities& capabilities) const
    {
        if (surface->getAPIType() != EAT_VULKAN)
            return false;

        // Todo(achal): not sure yet, how would I handle multiple platforms without making
        // this function templated
        VkSurfaceKHR vk_surface = static_cast<const CSurfaceVulkanWin32*>(surface)->getInternalObject();

        VkSurfaceCapabilitiesKHR vk_surfaceCapabilities;
        if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_vkPhysicalDevice, vk_surface,
            &vk_surfaceCapabilities) != VK_SUCCESS)
        {
            return false;
        }

        capabilities.minImageCount = vk_surfaceCapabilities.minImageCount;
        capabilities.maxImageCount = vk_surfaceCapabilities.maxImageCount;
        capabilities.currentExtent = vk_surfaceCapabilities.currentExtent;
        capabilities.minImageExtent = vk_surfaceCapabilities.minImageExtent;
        capabilities.maxImageExtent = vk_surfaceCapabilities.maxImageExtent;
        capabilities.maxImageArrayLayers = vk_surfaceCapabilities.maxImageArrayLayers;
        // Todo(achal)
        // VkSurfaceTransformFlagsKHR       supportedTransforms;
        // VkSurfaceTransformFlagBitsKHR    currentTransform;
        // VkCompositeAlphaFlagsKHR         supportedCompositeAlpha;
        capabilities.supportedUsageFlags = static_cast<asset::IImage::E_USAGE_FLAGS>(vk_surfaceCapabilities.supportedUsageFlags);

        return true;
    }

    bool isSwapchainSupported() const override
    {
        constexpr uint32_t MAX_DEVICE_EXTENSIONS_COUNT = 250u;

        uint32_t availableExtensionCount;
        VkResult retval = vkEnumerateDeviceExtensionProperties(m_vkPhysicalDevice, NULL, &availableExtensionCount, NULL);

        // Todo(achal): Would there be a need to handle VK_INCOMPLETE separately?
        if ((retval != VK_SUCCESS) && (retval != VK_INCOMPLETE))
            return false;

        assert(availableExtensionCount <= MAX_DEVICE_EXTENSIONS_COUNT);

        VkExtensionProperties availableExtensions[MAX_DEVICE_EXTENSIONS_COUNT];
        retval = vkEnumerateDeviceExtensionProperties(m_vkPhysicalDevice, NULL, &availableExtensionCount, availableExtensions);

        // Todo(achal): Would there be a need to handle VK_INCOMPLETE separately?
        if ((retval != VK_SUCCESS) && (retval != VK_INCOMPLETE))
            return false;

        for (uint32_t i = 0u; i < availableExtensionCount; ++i)
        {
            if (strcmp(availableExtensions[i].extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0)
                return true;
        }

        return false;
    }
            
protected:
    core::smart_refctd_ptr<ILogicalDevice> createLogicalDevice_impl(const ILogicalDevice::SCreationParams& params) override
    {
        VkDeviceCreateInfo vk_createInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
        vk_createInfo.pNext = nullptr; // there is a super long list of available extensions to this structure
        vk_createInfo.flags = static_cast<VkDeviceCreateFlags>(0); // reserved for future use, by Vulkan

        vk_createInfo.queueCreateInfoCount = params.queueParamsCount;
        core::vector<VkDeviceQueueCreateInfo> queueCreateInfos(vk_createInfo.queueCreateInfoCount);
        for (uint32_t i = 0u; i < queueCreateInfos.size(); ++i)
        {
            const auto& qparams = params.queueCreateInfos[i];
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
        const uint32_t deviceExtensionCount = 1u;
        const char* deviceExtensions[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
                
        vk_createInfo.enabledExtensionCount = deviceExtensionCount;
        vk_createInfo.ppEnabledExtensionNames = deviceExtensions;
        
        // Todo(achal): Need to get this from the user, which features they want
        VkPhysicalDeviceFeatures vk_deviceFeatures = {};        
        vk_createInfo.pEnabledFeatures = &vk_deviceFeatures;        
                
        VkDevice vk_device = VK_NULL_HANDLE;
        if (vkCreateDevice(m_vkPhysicalDevice, &vk_createInfo, nullptr, &vk_device) == VK_SUCCESS)
        {
            return core::make_smart_refctd_ptr<CVKLogicalDevice>(this, vk_device, params,
                core::smart_refctd_ptr(m_system));
        }
        else
        {
            return nullptr;
        }    
    }
            
private:
    VkPhysicalDevice m_vkPhysicalDevice;
};
        
}

#endif