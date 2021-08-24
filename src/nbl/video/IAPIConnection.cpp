#include "nbl/video/IAPIConnection.h"

#include "nbl/video/IPhysicalDevice.h"

namespace nbl::video
{

core::SRange<IPhysicalDevice* const> IAPIConnection::getPhysicalDevices() const
{
    IPhysicalDevice* const begin = m_physicalDevices[0].get();
    IPhysicalDevice* const end = begin + m_physicalDevices.size();

    return core::SRange<IPhysicalDevice* const>(&begin, &end);
}

}