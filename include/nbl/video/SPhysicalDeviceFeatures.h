#ifndef _NBL_VIDEO_S_PHYSICAL_DEVICE_FEATURES_H_INCLUDED_
#define _NBL_VIDEO_S_PHYSICAL_DEVICE_FEATURES_H_INCLUDED_

#include "nbl/video/ECommonEnums.h"

namespace nbl::video
{
//! Usage of feature API
//! ## LogicalDevice creation enabled features shouldn't necessarily equal the ones it reports as enabled (superset)
//! **RARE: Creating a physical device with all advertised features/extensions:**
//! auto features = physicalDevice->getFeatures();
//! 
//! ILogicalDevice::SCreationParams params = {};
//! params.queueParamsCount = ; // set queue stuff
//! params.queueParams = ; // set queue stuff
//! params.enabledFeatures = features;
//! auto device = physicalDevice->createLogicalDevice(params);
//! **FREQUENT: Choosing a physical device with the features**
//! IPhysicalDevice::SRequiredProperties props = {}; // default initializes to apiVersion=1.1, deviceType = ET_UNKNOWN, pipelineCacheUUID = '\0', device UUID=`\0`, driverUUID=`\0`, deviceLUID=`\0`, deviceNodeMask= ~0u, driverID=UNKNOWN
//! // example of particular config
//! props.apiVersion = 1.2;
//! props.deviceTypeMask = ~IPhysicalDevice::ET_CPU;
//! props.driverIDMask = ~(EDI_AMD_PROPRIETARY|EDI_INTEL_PROPRIETARY_WINDOWS); // would be goot to turn the enum into a mask
//! props.conformanceVersion = 1.2;
//! 
//! SDeviceFeatures requiredFeatures = {};
//! requiredFeatures.rayQuery = true;
//! 
//! SDeviceLimits minimumLimits = {}; // would default initialize to worst possible values (small values for maximum sizes, large values for alignments, etc.)

struct SPhysicalDeviceFeatures
{
    #include "nbl/video/SPhysicalDeviceFeatures_members.h"

    inline bool operator==(const SPhysicalDeviceFeatures& _rhs) const
    {
        return memcmp(this, &_rhs, sizeof(SPhysicalDeviceFeatures)) == 0u;
    }

    inline bool isSubsetOf(const SPhysicalDeviceFeatures& _rhs) const
    {
        const auto& intersection = intersectWith(_rhs);
        return intersection == *this;
    }

    inline SPhysicalDeviceFeatures unionWith(const SPhysicalDeviceFeatures& _rhs) const
    {
        SPhysicalDeviceFeatures res = *this;

        #include "nbl/video/SPhysicalDeviceFeatures_union.h"

        return res;
    }

    inline SPhysicalDeviceFeatures intersectWith(const SPhysicalDeviceFeatures& _rhs) const
    {
        SPhysicalDeviceFeatures res = *this;

        #include "nbl/video/SPhysicalDeviceFeatures_intersect.h"

        return res;
    }
};

template<typename T>
concept DeviceFeatureDependantClass = requires(const SPhysicalDeviceFeatures& availableFeatures, SPhysicalDeviceFeatures& features) { 
    T::enableRequiredFeautres(features);
    T::enablePreferredFeatures(availableFeatures, features);
};

} // nbl::video
#endif
