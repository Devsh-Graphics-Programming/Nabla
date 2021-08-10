#include "nbl/video/IBackendObject.h"

#include "nbl/video/ILogicalDevice.h"

namespace nbl::video
{

E_API_TYPE IBackendObject::getAPIType() const
{
    return m_originDevice->getAPIType();
}

}