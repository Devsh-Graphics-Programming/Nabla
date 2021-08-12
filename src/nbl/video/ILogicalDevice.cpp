#include "nbl/video/IPhysicalDevice.h"

namespace nbl::video
{

E_API_TYPE ILogicalDevice::getAPIType() const { return m_physicalDevice->getAPIType(); }

}