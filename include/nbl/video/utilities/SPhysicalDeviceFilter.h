#ifndef __NBL_VIDEO_S_PHYSICAL_DEVICE_FILTER_H_INCLUDED__
#define __NBL_VIDEO_S_PHYSICAL_DEVICE_FILTER_H_INCLUDED__

#include "nbl/video/IPhysicalDevice.h"

namespace nbl::video
{
    struct SPhysicalDeviceFilter
    {
        IPhysicalDevice::APIVersion                     minApiVersion = {0u, 0u, 0u};
        core::bitflag<IPhysicalDevice::E_TYPE>          deviceTypeMask = core::bitflag<IPhysicalDevice::E_TYPE>(0xffu);
        core::bitflag<IPhysicalDevice::E_DRIVER_ID>     driverIDMask = core::bitflag<IPhysicalDevice::E_DRIVER_ID>(0xffff'ffffu);
        VkConformanceVersion                            minConformanceVersion = {0u, 0u, 0u, 0u};
        IPhysicalDevice::SLimits                        minimumLimits = {}; // minimum required limits to be satisfied
        IPhysicalDevice::SFeatures                      requiredFeatures = {}; // required features, will be also used to enable logical device features
        IPhysicalDevice::SFormatBufferUsages            requiredBufferFormatUsages = {};
        IPhysicalDevice::SFormatImageUsages             requiredImageFormatUsagesLinearTiling = {};
        IPhysicalDevice::SFormatImageUsages             requiredImageFormatUsagesOptimalTiling = {};

        struct MemoryRequirement
        {
            size_t size = 0ull;
            core::bitflag<IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS> memoryFlags = IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS::EMPF_NONE;
        };
        MemoryRequirement* memoryRequirements = nullptr;
        uint32_t memoryRequirementsCount = 0u;
        
        struct QueueRequirement
        {
            core::bitflag<IPhysicalDevice::E_QUEUE_FLAGS> requiredFlags = IPhysicalDevice::E_QUEUE_FLAGS::EQF_NONE;
            core::bitflag<IPhysicalDevice::E_QUEUE_FLAGS> disallowedFlags = IPhysicalDevice::E_QUEUE_FLAGS::EQF_NONE;
            uint32_t queueCount = 0u;
            // family's transfer granularity needs to be <=
            asset::VkExtent3D maxImageTransferGranularity = {0x80000000u,0x80000000u,0x80000000u};
        };
        QueueRequirement* queueRequirements = nullptr;
        uint32_t queueRequirementsCount = 0u;

        // To determine whether a queue family of a physical device supports presentation to a given surface
        //  See vkGetPhysicalDeviceSurfaceSupportKHR
        struct SurfaceCompatibility
        {
            ISurface* surface = nullptr;
            // Setting this to `EQF_NONE` means it sufffices to find any queue family that can present to this surface, regardless of flags it might have
            core::bitflag<IPhysicalDevice::E_QUEUE_FLAGS> presentationQueueFlags = IPhysicalDevice::E_QUEUE_FLAGS::EQF_NONE;
        };
        SurfaceCompatibility* requiredSurfaceCompatibilities = nullptr;
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

            // Surface Compatibility
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

            // Memory Requirements Checking:
            core::bitflag<IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS> heapFlags[VK_MAX_MEMORY_HEAPS] = {};
            for (uint32_t h = 0; h < memoryProps.memoryHeapCount; ++h)
            {
                heapFlags[h] = IDeviceMemoryAllocation::EMPF_NONE;
                for (uint32_t p = 0; p < memoryProps.memoryTypeCount; ++p)
                    if (memoryProps.memoryTypes[p].heapIndex == h)
                        heapFlags[h] |= memoryProps.memoryTypes[p].propertyFlags;
            }
            // over-estimation, Not exact 
            // TODO: Exact or Better Logic -> try find a feasible fitting of requirements into heaps.
            for (uint32_t m = 0; m < memoryRequirementsCount; ++m)
            {
                size_t memSize = memoryRequirements[m].size;
                for (uint32_t h = 0; h < memoryProps.memoryHeapCount; ++h)
                    if (heapFlags[h].hasFlags(memoryRequirements[m].memoryFlags))
                        memSize = (memoryProps.memoryHeaps[h].size > memSize) ? 0ull : memSize - memoryProps.memoryHeaps[h].size;
                if (memSize > 0)
                    return false;
            }
            
            // Queue Requirements Checking:
            // over-estimation, Not exact 
            // TODO: Exact or Better Logic -> try find a feasible fitting of requirements into queue families.
            for (uint32_t q = 0; q < queueRequirementsCount; ++q)
            {
                const auto& queueReqs = queueRequirements[q];
                uint32_t queueCount = queueReqs.queueCount;
                
                for (uint32_t qfam = 0; qfam < queueProps.size(); ++qfam)
                {
                    const auto& queueFamilyProps = queueProps[qfam];

                    // has requiredFlags
                    if (queueFamilyProps.queueFlags.hasFlags(queueReqs.requiredFlags))
                    {
                        // doesn't have disallowed flags
                        if ((queueFamilyProps.queueFlags & queueReqs.disallowedFlags).value == 0)
                        {
                            // imageTransferGranularity
                            if (queueReqs.maxImageTransferGranularity.width > queueFamilyProps.minImageTransferGranularity.width &&
                                queueReqs.maxImageTransferGranularity.height > queueFamilyProps.minImageTransferGranularity.height &&
                                queueReqs.maxImageTransferGranularity.depth > queueFamilyProps.minImageTransferGranularity.depth)
                            {
                                queueCount = (queueFamilyProps.queueCount > queueCount) ? 0ull : queueCount - queueFamilyProps.queueCount;
                            }
                        }
                    }
                }

                if (queueCount > 0)
                    return false;
            }

            return true;
        }
    };
}

#endif