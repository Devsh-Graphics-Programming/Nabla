#include "nbl/video/IDeviceMemoryAllocation.h"

namespace nbl::video
{

E_API_TYPE IDeviceMemoryAllocation::getAPIType() const
{
    assert(m_originDevice); // any device memory shouldn't be allocated without creating a logical device

    return m_originDevice->getAPIType();
}

}