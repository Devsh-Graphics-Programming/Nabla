#ifndef __NBL_C_VULKAN_PHYSICAL_DEVICE_H_INCLUDED__
#define __NBL_C_VULKAN_PHYSICAL_DEVICE_H_INCLUDED__

#include "nbl/video/IPhysicalDevice.h"

#include <volk.h>

#include "nbl/video/CVKLogicalDevice.h"
#include "nbl/video/surface/ISurfaceVK.h"

namespace nbl::video
{
        
class CVulkanPhysicalDevice final : public IPhysicalDevice
{
public:
    CVulkanPhysicalDevice(VkPhysicalDevice _vkphd, core::smart_refctd_ptr<system::ISystem>&& sys, core::smart_refctd_ptr<asset::IGLSLCompiler>&& glslc)
        : IPhysicalDevice(std::move(sys), std::move(glslc)), m_vkphysdev(_vkphd)
    {
        // Get physical device's limits
        VkPhysicalDeviceSubgroupProperties subgroupProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES };
        {
            VkPhysicalDeviceProperties2 deviceProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
            deviceProperties.pNext = &subgroupProperties;
            vkGetPhysicalDeviceProperties2(m_vkphysdev, &deviceProperties);
                    
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
            vkGetPhysicalDeviceFeatures(m_vkphysdev, &features);
                    
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
        vkGetPhysicalDeviceQueueFamilyProperties(m_vkphysdev, &qfamCount, nullptr);
        core::vector<VkQueueFamilyProperties> qfamprops(qfamCount);
        vkGetPhysicalDeviceQueueFamilyProperties(m_vkphysdev, &qfamCount, qfamprops.data());
                
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
            
    inline VkPhysicalDevice getInternalObject() const { return m_vkphysdev; }
            
    E_API_TYPE getAPIType() const override { return EAT_VULKAN; }

    std::vector<ISurface::SFormat> getAvailableFormatsForSurface(const ISurface* surface) const override
    {
        const ISurfaceVK* vk_surface = static_cast<const ISurfaceVK*>(surface); // Todo(achal): This is problematic, if passed `surface` isn't a vulkan surface

        uint32_t formatCount = 0u;
        vkGetPhysicalDeviceSurfaceFormatsKHR(m_vkphysdev, vk_surface->m_surface, &formatCount, nullptr);
        std::vector<VkSurfaceFormatKHR> formats(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(m_vkphysdev, vk_surface->m_surface, &formatCount, formats.data());

        std::vector<ISurface::SFormat> result(formatCount);

        for (uint32_t i = 0u; i < formatCount; ++i)
        {
            result[i].format = getFormat(formats[i].format);
            result[i].colorSpace = getColorSpace(formats[i].colorSpace);
        }

        return result;
    }

    std::vector<ISurface::E_PRESENT_MODE> getAvailablePresentModesForSurface(const ISurface* surface) const override
    {
        const ISurfaceVK* vk_surface = static_cast<const ISurfaceVK*>(surface); // Todo(achal): This is problematic, if passed `surface` isn't a vulkan surface

        uint32_t count = 0u;
        vkGetPhysicalDeviceSurfacePresentModesKHR(m_vkphysdev, vk_surface->m_surface, &count, NULL);
        std::vector<VkPresentModeKHR> vk_presentModes(count);
        vkGetPhysicalDeviceSurfacePresentModesKHR(m_vkphysdev, vk_surface->m_surface, &count, vk_presentModes.data());

        std::vector<ISurface::E_PRESENT_MODE> result(count);
        for (uint32_t i = 0u; i < count; ++i)
        {
            result[i] = getPresentMode(vk_presentModes[i]);
        }

        return result;
    }
    
    uint32_t getMinImageCountForSurface(const ISurface* surface) const override
    {
        // Todo(achal): This is problematic, if passed `surface` isn't a vulkan surface
        const ISurfaceVK* vk_surface = static_cast<const ISurfaceVK*>(surface);

        VkSurfaceCapabilitiesKHR surfaceCapabilities;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_vkphysdev, vk_surface->m_surface,
            &surfaceCapabilities);

        return surfaceCapabilities.minImageCount;
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
        assert(vkCreateDevice(m_vkphysdev, &createInfo, nullptr, &vkdev) == VK_SUCCESS);
                
        return core::make_smart_refctd_ptr<CVKLogicalDevice>(vkdev, params, core::smart_refctd_ptr(m_system), core::smart_refctd_ptr(m_GLSLCompiler));
    }
            
private:
    static inline asset::E_FORMAT getFormat(VkFormat in)
    {
        if (in <= VK_FORMAT_BC7_SRGB_BLOCK)
            return static_cast<asset::E_FORMAT>(in);

        // Note(achal): Some of this ugliness could be remedied if we put the range [EF_ETC2_R8G8B8_UNORM_BLOCK, EF_EAC_R11G11_SNORM_BLOCK] just
        // after EF_BC7_SRGB_BLOCK, not sure how rest of the code will react to it
        if (in >= VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK && in <= VK_FORMAT_EAC_R11G11_SNORM_BLOCK) // [147, 156] --> [175, 184]
            return static_cast<asset::E_FORMAT>(in + 28u);

        if (in >= VK_FORMAT_ASTC_4x4_UNORM_BLOCK && in <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK) // [157, 184]
            return static_cast<asset::E_FORMAT>(in - 10u);

        // Note(achal): This ugliness is not so easy to get rid of
        if (in >= VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG && in <= VK_FORMAT_PVRTC2_4BPP_SRGB_BLOCK_IMG) // [1000054000, 1000054007] --> [185, 192]
            return static_cast<asset::E_FORMAT>(in - 1000053815u);

        if (in >= VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM && in <= VK_FORMAT_G8_B8_R8_3PLANE_444_UNORM) // [1000156002, 1000156006] --> [193, 197]
            return static_cast<asset::E_FORMAT>(in - 1000155809);

        // Todo(achal): Log a warning that you got an unrecognized format
        return asset::EF_UNKNOWN;
    }

    // Todo(achal): Check it, a lot of stuff could be incorrect!
    static inline ISurface::SColorSpace getColorSpace(VkColorSpaceKHR in)
    {
        ISurface::SColorSpace result = {};

        switch (in)
        {
            case VK_COLOR_SPACE_SRGB_NONLINEAR_KHR:
            {
                result.primary = asset::ECP_SRGB;
                result.eotf = asset::EOTF_sRGB;
            } break;
                
            case VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT:
            {
                result.primary = asset::ECP_DISPLAY_P3;
                result.eotf = asset::EOTF_sRGB; // spec says "sRGB-like"
            } break;

            case VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT:
            {
                result.primary = asset::ECP_SRGB;
                result.eotf = asset::EOTF_IDENTITY;
            } break;

            case VK_COLOR_SPACE_DISPLAY_P3_LINEAR_EXT:
            {
                result.primary = asset::ECP_DISPLAY_P3;
                result.eotf = asset::EOTF_IDENTITY;
            } break;

            case VK_COLOR_SPACE_DCI_P3_NONLINEAR_EXT:
            {
                result.primary = asset::ECP_DCI_P3;
                result.eotf = asset::EOTF_DCI_P3_XYZ;
            } break;

            case VK_COLOR_SPACE_BT709_LINEAR_EXT:
            {
                result.primary = asset::ECP_SRGB;
                result.eotf = asset::EOTF_IDENTITY;
            } break;

            case VK_COLOR_SPACE_BT709_NONLINEAR_EXT:
            {
                result.primary = asset::ECP_SRGB;
                result.eotf = asset::EOTF_SMPTE_170M;
            } break;

            case VK_COLOR_SPACE_BT2020_LINEAR_EXT:
            {
                result.primary = asset::ECP_BT2020;
                result.eotf = asset::EOTF_IDENTITY;
            } break;

            case VK_COLOR_SPACE_HDR10_ST2084_EXT:
            {
                result.primary = asset::ECP_BT2020;
                result.eotf = asset::EOTF_SMPTE_ST2084;
            } break;

            case VK_COLOR_SPACE_DOLBYVISION_EXT:
            {
                result.primary = asset::ECP_BT2020;
                result.eotf = asset::EOTF_SMPTE_ST2084;
            } break;

            case VK_COLOR_SPACE_HDR10_HLG_EXT:
            {
                result.primary = asset::ECP_BT2020;
                result.eotf = asset::EOTF_HDR10_HLG;
            } break;

            case VK_COLOR_SPACE_ADOBERGB_LINEAR_EXT:
            {
                result.primary = asset::ECP_ADOBERGB;
                result.eotf = asset::EOTF_IDENTITY;
            } break;

            case VK_COLOR_SPACE_ADOBERGB_NONLINEAR_EXT:
            {
                result.primary = asset::ECP_ADOBERGB;
                result.eotf = asset::EOTF_GAMMA_2_2;
            } break;

            case VK_COLOR_SPACE_PASS_THROUGH_EXT:
            {
                result.primary = asset::ECP_PASS_THROUGH;
                result.eotf = asset::EOTF_IDENTITY;
            } break;
                
            case VK_COLOR_SPACE_EXTENDED_SRGB_NONLINEAR_EXT:
            {
                result.primary = asset::ECP_SRGB;
                result.eotf = asset::EOTF_sRGB;
            } break;

            case VK_COLOR_SPACE_DISPLAY_NATIVE_AMD: // this one is completely bogus, I don't understand it at all
            {
                result.primary = asset::ECP_SRGB;
                result.eotf = asset::EOTF_UNKNOWN;
            } break;

            default:
            {
                // Todo(achal): Log warning unknown color space
            } break;
        }

        return result;
    }

    static inline ISurface::E_PRESENT_MODE getPresentMode(VkPresentModeKHR in)
    {
        if (in <= VK_PRESENT_MODE_FIFO_RELAXED_KHR)
        {
            return static_cast<ISurface::E_PRESENT_MODE>(in);
        }
        else
        {
            // Todo(achal): Log warning unknown present modes
            return static_cast<ISurface::E_PRESENT_MODE>(0);
        }
    }

    VkPhysicalDevice m_vkphysdev;
};
        
}

#endif