#include "nbl/video/IDeviceMemoryAllocation.h"

namespace nbl::video
{

E_API_TYPE IDeviceMemoryAllocation::getAPIType() const
{
    assert(m_originDevice); // any device memory shouldn't be allocated without creating a logical device

    return m_originDevice->getAPIType();
}

IDeviceMemoryAllocation::MemoryRange IDeviceMemoryAllocation::alignNonCoherentRange(MemoryRange range) const
{
    const auto alignment = m_originDevice->getPhysicalDevice()->getLimits().nonCoherentAtomSize;
    range.offset = core::alignDown(range.offset,alignment);
    range.length = core::min(core::alignUp(range.length,alignment),m_allocationSize);
    return range;
}

}