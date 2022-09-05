#ifndef __NBL_VIDEO_S_DEFAULT_PHYSICAL_DEVICE_FILTER_H_INCLUDED__
#define __NBL_VIDEO_S_DEFAULT_PHYSICAL_DEVICE_FILTER_H_INCLUDED__

#include "nbl/video/IPhysicalDevice.h"

namespace nbl::video
{
    struct SDefaultPhysicalDeviceFilter
    {
        IPhysicalDevice::APIVersion                     minApiVersion = {0u, 0u, 0u};
        core::bitflag<IPhysicalDevice::E_TYPE>          deviceTypeMask = core::bitflag<IPhysicalDevice::E_TYPE>(0xffu);
        core::bitflag<IPhysicalDevice::E_DRIVER_ID>     driverIDMask = core::bitflag<IPhysicalDevice::E_DRIVER_ID>(0xffff'ffffu);
        VkConformanceVersion                            minConformanceVersion = {0u, 0u, 0u, 0u};
        IPhysicalDevice::SLimits                        minimumLimits = {}; // minimum required limits to be satisfied
        IPhysicalDevice::SFeatures                      requiredFeatures = {};
        IPhysicalDevice::SFormatBufferUsages            requiredBufferFormatUsages = {};
        IPhysicalDevice::SFormatImageUsages             requiredImageFormatUsagesLinearTiling = {};
        IPhysicalDevice::SFormatImageUsages             requiredImageFormatUsagesOptimalTiling = {};

        // TODO: memory requirements
        /*
        using RequiredMemoryType = core::bitflag<IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS>;
        core::set<RequiredMemoryType,size_t> requiredMemoryTypesWithMinimumSizes;
        */
        
        // TODO: queue requirements
        /*
        struct QueueSiblings
        {
            core::bitflag<E_QUEUE_FLAGS> requiredFlags = NONE;
            core::bitflag<E_QUEUE_FLAGS> disallowedFlags = NONE;
            uint32_t queueCount = 0u;
            // family's transfer granularity needs to be <=
            asset::VkExtent3D maxImageTransferGranularity = {0x80000000u,0x80000000u,0x80000000u};
        };
        */
        
        // To determine whether a queue family of a physical device supports presentation to a given surface
        //  See vkGetPhysicalDeviceSurfaceSupportKHR
        struct SurfaceCompatibility
        {
            ISurface* surface = nullptr;
            // Setting this to `EQF_NONE` means it sufffices to find any queue family that can present to this surface, regardless of flags it might have
            core::bitflag<IPhysicalDevice::E_QUEUE_FLAGS> presentationQueueFlags = IPhysicalDevice::E_QUEUE_FLAGS::EQF_NONE;
        };
        SurfaceCompatibility * requiredSurfaceCompatibilities = nullptr;
        uint32_t requiredSurfaceCompatibilitiesCount = 0u;

        bool meetsRequirements(const IPhysicalDevice * const physicalDevice) const
        {
            const auto& properties = physicalDevice->getProperties();
            const auto& physDevLimits = physicalDevice->getProperties().limits;
            const auto& physDevFeatures = physicalDevice->getFeatures();
            const auto& memoryProps = physicalDevice->getMemoryProperties();
            const auto& queueProps = physicalDevice->getQueueFamilyProperties();

            if (properties.apiVersion < minApiVersion)
                return false;
            if (!deviceTypeMask.hasFlags(properties.deviceType))
                return false;
            if (!driverIDMask.hasFlags(properties.driverID))
                return false;
            
            auto conformanceVersionValid = [&]() -> bool
            {
                if (properties.conformanceVersion.major != minConformanceVersion.major)
                    return properties.conformanceVersion.major > minConformanceVersion.major;
                else if (properties.conformanceVersion.minor != minConformanceVersion.minor)
                    return properties.conformanceVersion.minor > minConformanceVersion.minor;
                else if (properties.conformanceVersion.subminor != minConformanceVersion.subminor)
                    return properties.conformanceVersion.subminor > minConformanceVersion.subminor;
                else if (properties.conformanceVersion.patch != minConformanceVersion.patch)
                    return properties.conformanceVersion.patch > minConformanceVersion.patch;
                return true;
            }();
            
            if (!conformanceVersionValid)
                return false;

            if (!minimumLimits.isSubsetOf(physDevLimits))
                return false;

            if (!requiredFeatures.isSubsetOf(physDevFeatures))
                return false;

            if (!requiredBufferFormatUsages.isSubsetOf(physicalDevice->getBufferFormatUsages()))
                return false;
            if (!requiredImageFormatUsagesLinearTiling.isSubsetOf(physicalDevice->getImageFormatUsagesLinearTiling()))
                return false;
            if (!requiredImageFormatUsagesOptimalTiling.isSubsetOf(physicalDevice->getImageFormatUsagesOptimalTiling()))
                return false;

            if (requiredSurfaceCompatibilities != nullptr)
            {
                for (uint32_t i = 0u; i < requiredSurfaceCompatibilitiesCount; ++i)
                {
                    const auto& requiredSurfaceCompatibility = requiredSurfaceCompatibilities[i];
                    if (requiredSurfaceCompatibility.surface == nullptr)
                        continue; // we don't care about compatibility with a nullptr surface :)
                    
                    const auto& queueFamilyProperties = physicalDevice->getQueueFamilyProperties();

                    bool physicalDeviceSupportsSurfaceWithQueueFlags = false;
                    for (uint32_t qfam = 0u; qfam < queueFamilyProperties.size(); ++qfam)
                    {
                        const auto& familyProperty = queueFamilyProperties[qfam];
                        if(familyProperty.queueFlags.hasFlags(requiredSurfaceCompatibility.presentationQueueFlags))
                            if(requiredSurfaceCompatibility.surface->isSupportedForPhysicalDevice(physicalDevice, qfam))
                                physicalDeviceSupportsSurfaceWithQueueFlags = true;
                    }

                    if(!physicalDeviceSupportsSurfaceWithQueueFlags)
                        return false;
                }
            }

            return true;
        }
    };
}

#endif