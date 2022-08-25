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
            if(physicalDevice->getProperties().apiVersion < minApiVersion)
                return false;
            return true;
        }
    };
}

#endif