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

}