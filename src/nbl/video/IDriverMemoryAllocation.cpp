#include "nbl/video/IDriverMemoryAllocation.h"

namespace nbl::video
{
E_API_TYPE IDriverMemoryAllocation::getAPIType() const
{
    assert(m_originDevice);  // any device memory shouldn't be allocated without creating a logical device

    return m_originDevice->getAPIType();
}

}