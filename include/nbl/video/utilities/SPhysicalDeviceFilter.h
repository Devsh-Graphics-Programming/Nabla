#ifndef _NBL_VIDEO_S_PHYSICAL_DEVICE_FILTER_H_INCLUDED_
#define _NBL_VIDEO_S_PHYSICAL_DEVICE_FILTER_H_INCLUDED_

#include "nbl/video/IPhysicalDevice.h"
#include "nbl/video/IGPUCommandBuffer.h"

namespace nbl::video
{
    struct SPhysicalDeviceFilter
    {
        IPhysicalDevice::APIVersion                     minApiVersion = {1u, 3u, 0u, 0u};
        core::bitflag<IPhysicalDevice::E_TYPE>          deviceTypeMask = core::bitflag<IPhysicalDevice::E_TYPE>(0xffu);
        core::bitflag<IPhysicalDevice::E_DRIVER_ID>     driverIDMask = core::bitflag<IPhysicalDevice::E_DRIVER_ID>(0xffff'ffffu);
        IPhysicalDevice::APIVersion                     minConformanceVersion = {0u, 0u, 0u, 0u};
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
        std::span<const MemoryRequirement> memoryRequirements = {};
        
        struct QueueRequirement
        {
            inline bool familyMatches(const IPhysicalDevice::SQueueFamilyProperties& props) const
            {
                const auto& queueFlags = props.queueFlags;
                if (!queueFlags.hasFlags(requiredFlags))
                    return false;

                // doesn't have disallowed flags
                if (queueFlags&disallowedFlags)
                    return false;

                return maxImageTransferGranularity.width >= props.minImageTransferGranularity.width &&
                        maxImageTransferGranularity.height >= props.minImageTransferGranularity.height &&
                        maxImageTransferGranularity.depth >= props.minImageTransferGranularity.depth;
            }

            core::bitflag<IQueue::FAMILY_FLAGS> requiredFlags = IQueue::FAMILY_FLAGS::NONE;
            core::bitflag<IQueue::FAMILY_FLAGS> disallowedFlags = IQueue::FAMILY_FLAGS::NONE;
            uint32_t queueCount = 0u;
            // family's transfer granularity needs to be <=
            asset::VkExtent3D maxImageTransferGranularity = {0x80000000u,0x80000000u,0x80000000u};
        };
        std::span<const QueueRequirement> queueRequirements = {};

        // To determine whether a queue family of a physical device supports presentation to a given surface
        //  See vkGetPhysicalDeviceSurfaceSupportKHR
        struct SurfaceCompatibility
        {
            const ISurface* surface = nullptr;
            // Setting this to `EQF_NONE` means it sufffices to find any queue family that can present to this surface, regardless of flags it might have
            core::bitflag<IQueue::FAMILY_FLAGS> presentationQueueFlags = IQueue::FAMILY_FLAGS::NONE;
        };
        std::span<const SurfaceCompatibility> requiredSurfaceCompatibilities = {};


        // sift through multiple devices
        template<typename PhysicalDevice> requires std::is_same_v<std::remove_cv_t<PhysicalDevice>,video::IPhysicalDevice>
        void operator()(core::set<PhysicalDevice*>& physicalDevices) const
        {
            std::erase_if(physicalDevices,[&](const video::IPhysicalDevice* device)->bool{return !meetsRequirements(device);});
        }

        // check one device
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
            for (const auto& requiredSurfaceCompatibility : requiredSurfaceCompatibilities)
            {
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
            for (const auto& req : memoryRequirements)
            {
                size_t memSize = req.size;
                for (uint32_t h=0; h<memoryProps.memoryHeapCount; ++h)
                    if (heapFlags[h].hasFlags(req.memoryFlags))
                        memSize = memoryProps.memoryHeaps[h].size>memSize ? 0ull:(memSize-memoryProps.memoryHeaps[h].size);
                if (memSize>0)
                    return false;
            }
            
            // Queue Requirements Checking:
            // over-estimation, Not exact 
            // TODO: Exact or Better Logic -> try find a feasible fitting of requirements into queue families.
            for (const auto& queueReqs : queueRequirements)
            {
                uint32_t queueCount = queueReqs.queueCount;
                for (uint32_t qfam=0; qfam<queueProps.size(); ++qfam)
                {
                    const auto& queueFamilyProps = queueProps[qfam];
                    if (queueReqs.familyMatches(queueFamilyProps))
                        queueCount -= core::min(queueFamilyProps.queueCount,queueCount);
                }

                if (queueCount>0)
                    return false;
            }

            return true;
        }
    };
}

#endif