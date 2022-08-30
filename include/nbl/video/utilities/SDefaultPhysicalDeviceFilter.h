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
        // TODO: BufferFormatUsages
        // TODO: OptimalImageFormatUsages
        // TODO: LinearImageFormatUsages
        // TODO: ISurface* obligatoryCompatibleSurfaces
        
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
                if(properties.conformanceVersion.major != minConformanceVersion.major)
                    return properties.conformanceVersion.major > minConformanceVersion.major;
                else if(properties.conformanceVersion.minor != minConformanceVersion.minor)
                    return properties.conformanceVersion.minor > minConformanceVersion.minor;
                else if(properties.conformanceVersion.subminor != minConformanceVersion.subminor)
                    return properties.conformanceVersion.subminor > minConformanceVersion.subminor
                else if(properties.conformanceVersion.patch != minConformanceVersion.patch)
                    return properties.conformanceVersion.patch > minConformanceVersion.patch;
                return true;
            }();
            
            if (!conformanceVersionValid)
                return false;

            // trust me, this is the correct way to do it, don't even think about using `limits < minimumLimits` to detect failure, TODO: figure out a more intuitive way to do this. maybe `operator <` is a bit misleading...
            if (!(minimumLimits < physDevLimits))
                return false;

            // again, based on how the operator is implemented, makes more sense to check failure like below, but it doesn't mean >= like normal arithmetic:
            if (!(requiredFeatures < physDevFeatures))
                return false;

            return true;
        }
    };
}

#endif