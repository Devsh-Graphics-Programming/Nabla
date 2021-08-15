#ifndef __NBL_C_VULKAN_PHYSICAL_DEVICE_H_INCLUDED__
#define __NBL_C_VULKAN_PHYSICAL_DEVICE_H_INCLUDED__

#include "nbl/video/IPhysicalDevice.h"

#include <volk.h>

#include "nbl/video/CVKLogicalDevice.h"
#include "nbl/video/surface/CSurfaceVulkan.h"

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

    IDebugCallback* getDebugCallback() override { return nullptr; }

    void getAvailableFormatsForSurface(const ISurface* surface, uint32_t& formatCount, ISurface::SFormat* formats) const override
    {
#if 0
        const ISurfaceVK* vk_surface = static_cast<const ISurfaceVK*>(surface); // Todo(achal): This is problematic, if passed `surface` isn't a vulkan surface

        vkGetPhysicalDeviceSurfaceFormatsKHR(m_vkPhysicalDevice, vk_surface->m_surface, &formatCount, nullptr);

        if (!formats)
            return;

        std::vector<VkSurfaceFormatKHR> vk_formats(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(m_vkPhysicalDevice, vk_surface->m_surface, &formatCount, vk_formats.data());

        for (uint32_t i = 0u; i < formatCount; ++i)
        {
            formats[i].format = ISurfaceVK::getFormat(vk_formats[i].format);
            formats[i].colorSpace = ISurfaceVK::getColorSpace(vk_formats[i].colorSpace);
        }
#endif
    }
    
    ISurface::E_PRESENT_MODE getAvailablePresentModesForSurface(const ISurface* surface) const override
    {
#if 0
        const ISurfaceVK* vk_surface = static_cast<const ISurfaceVK*>(surface); // Todo(achal): This is problematic, if passed `surface` isn't a vulkan surface

        uint32_t count = 0u;
        vkGetPhysicalDeviceSurfacePresentModesKHR(m_vkPhysicalDevice, vk_surface->m_surface, &count, NULL);
        std::vector<VkPresentModeKHR> vk_presentModes(count);
        vkGetPhysicalDeviceSurfacePresentModesKHR(m_vkPhysicalDevice, vk_surface->m_surface, &count, vk_presentModes.data());

        ISurface::E_PRESENT_MODE result = static_cast<ISurface::E_PRESENT_MODE>(0);

        for (uint32_t i = 0u; i < count; ++i)
            result = static_cast<ISurface::E_PRESENT_MODE>(result | ISurfaceVK::getPresentMode(vk_presentModes[i]));

        return result;
#endif
        return ISurface::E_PRESENT_MODE::EPM_FIFO;
    }
    
    uint32_t getMinImageCountForSurface(const ISurface* surface) const override
    {
#if 0
        // Todo(achal): This is problematic, if passed `surface` isn't a vulkan surface
        const ISurfaceVK* vk_surface = static_cast<const ISurfaceVK*>(surface);

        VkSurfaceCapabilitiesKHR surfaceCapabilities;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_vkPhysicalDevice, vk_surface->m_surface,
            &surfaceCapabilities);

        return surfaceCapabilities.minImageCount;
#endif
        return ~0u;
    }
            
protected:
    core::smart_refctd_ptr<ILogicalDevice> createLogicalDevice_impl(const ILogicalDevice::SCreationParams& params) override
    {
        VkDeviceCreateInfo createInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
        // Todo(achal): Need to get this from the user
        createInfo.enabledLayerCount = 1u;
        const char* validationLayerName[] = { "VK_LAYER_KHRONOS_validation" };
        createInfo.ppEnabledLayerNames = validationLayerName;

        createInfo.queueCreateInfoCount = params.queueParamsCount;
                
        core::vector<VkDeviceQueueCreateInfo> queueCreateInfos(createInfo.queueCreateInfoCount);
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
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
                
        // Todo(achal): Need to get this from the user based on if its a headless compute or some presentation worthy workload
        const uint32_t deviceExtensionCount = 1u;
        const char* deviceExtensions[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
                
        createInfo.enabledExtensionCount = deviceExtensionCount;
        createInfo.ppEnabledExtensionNames = deviceExtensions;
                
        // Todo(achal): Need to get this from the user, which features they want
        VkPhysicalDeviceFeatures deviceFeatures = {};
                
        createInfo.pEnabledFeatures = &deviceFeatures;
                
        VkDevice vkdev = VK_NULL_HANDLE;
        assert(vkCreateDevice(m_vkPhysicalDevice, &createInfo, nullptr, &vkdev) == VK_SUCCESS);
                
        return core::make_smart_refctd_ptr<CVKLogicalDevice>(this, vkdev, params,
            core::smart_refctd_ptr(m_system));
    }
            
private:
    VkPhysicalDevice m_vkPhysicalDevice;
};
        
}

#endif