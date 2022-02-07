#include "nbl/video/IAPIConnection.h"

#include "nbl/video/IPhysicalDevice.h"

namespace nbl::video
{
core::SRange<IPhysicalDevice* const> IAPIConnection::getPhysicalDevices() const
{
    static_assert(sizeof(std::unique_ptr<IPhysicalDevice>) == sizeof(void*));

    return core::SRange<IPhysicalDevice* const>(
        reinterpret_cast<IPhysicalDevice* const*>(m_physicalDevices.data()),
        reinterpret_cast<IPhysicalDevice* const*>(m_physicalDevices.data()) + m_physicalDevices.size());
}

core::SRange<const IAPIConnection::E_FEATURE> IAPIConnection::getDependentFeatures(const E_FEATURE feature)
{
    switch(feature)
    {
        case EF_SURFACE: {
            static E_FEATURE depFeatures[] = {EF_SURFACE};
            return {depFeatures, depFeatures + sizeof(depFeatures) / sizeof(E_FEATURE)};
        }
        default:
            return {nullptr, nullptr};
    }
}

}