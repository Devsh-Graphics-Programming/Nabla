#ifndef __NBL_I_PHYSICAL_DEVICE_H_INCLUDED__
#define __NBL_I_PHYSICAL_DEVICE_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"

#include "nbl/video/ILogicalDevice.h"
#include "nbl/asset/IImage.h" //for VkExtent3D only
#include <type_traits>

namespace nbl {
namespace video
{

class IPhysicalDevice : public core::IReferenceCounted
{
public:
    struct SProperties
    {
        // TODO
    };

    struct SFeatures
    {
        // TODO
    };

    enum E_QUEUE_FLAGS : uint32_t
    {
        EQF_GRAPHICS_BIT = 0x01,
        EQF_COMPUTE_BIT = 0x02,
        EQF_TRANSFER_BIT = 0x04,
        EQF_SPARSE_BINDING_BIT = 0x08,
        EQF_PROTECTED_BIT = 0x10
    };
    struct SQueueFamilyProperties
    {
        std::underlying_type_t<E_QUEUE_FLAGS> queueFlags;
        uint32_t queueCount;
        uint32_t timestampValidBits;
        asset::VkExtent3D minImageTransferGranularity;
    };

    IPhysicalDevice() = default;

    const SProperties& getProperties() const { return m_properties; }
    const SFeatures& getFeatures() const { return m_features; }

    auto getQueueFamilyProperties() const 
    {
        using citer_t = qfam_props_array_t::pointee::const_iterator;
        return core::SRange<const SQueueFamilyProperties, citer_t, citer_t>(
            m_qfamProperties->cbegin(),
            m_qfamProperties->cend()
        );
    }

    core::smart_refctd_ptr<ILogicalDevice> createLogicalDevice(const ILogicalDevice::SCreationParams& params)
    {
        if (!validateLogicalDeviceCreation(params))
            return nullptr;

        return createLogicalDevice_impl(params);
    }

protected:
    virtual core::smart_refctd_ptr<ILogicalDevice> createLogicalDevice_impl(const ILogicalDevice::SCreationParams& params) = 0;

    bool validateLogicalDeviceCreation(const ILogicalDevice::SCreationParams& params) const
    {
        using range_t = core::SRange<const ILogicalDevice::SQueueCreationParams>;
        range_t qcis(params.queueCreateInfos, params.queueCreateInfos+params.queueParamsCount);

        for (const auto& qci : qcis)
        {
            if (qci.familyIndex >= m_qfamProperties->size())
                return false;

            const auto& qfam = (*m_qfamProperties)[qci.familyIndex];
            if (qci.count == 0u)
                return false;
            if (qci.count > qfam.queueCount)
                return false;

            for (uint32_t i = 0u; i < qci.count; ++i)
            {
                const float priority = qci.priorities[i];
                if (priority < 0.f)
                    return false;
                if (priority > 1.f)
                    return false;
            }
        }

        return true;
    }

    virtual ~IPhysicalDevice() = default;

    SProperties m_properties;
    SFeatures m_features;
    using qfam_props_array_t = core::smart_refctd_dynamic_array<SQueueFamilyProperties>;
    qfam_props_array_t m_qfamProperties;
};

}
}

#endif
