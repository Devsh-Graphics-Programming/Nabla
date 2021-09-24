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

core::SRange<IAPIConnection::E_FEATURE> IAPIConnection::getDependentFeatures(const E_FEATURE feature)
{
#if 0
    constexpr uint32_t MAX_COUNT = (1 << 13) / sizeof(E_FEATURE);

    E_FEATURE depFeatures[MAX_COUNT];
    core::SRange<E_FEATURE> result = { E_SURFACE };
    uint32_t totalDepFeatureCount = 0u;
    if (feature == E_SURFACE)
    {
        depFeatures[totalDepFeatureCount++] = E_SURFACE;
        depFeatures[totalDepFeatureCount++] = E_SURFACE;
    }

    return result;
#endif
    return { nullptr, nullptr };
}

}