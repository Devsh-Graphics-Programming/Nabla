#ifndef __NBL_I_BACKEND_OBJECT_H_INCLUDED__
#define __NBL_I_BACKEND_OBJECT_H_INCLUDED__

#include "nbl/video/EApiType.h"
#include <type_traits>

namespace nbl
{
namespace video
{

class ILogicalDevice;

class IBackendObject
{
public:
    IBackendObject(const ILogicalDevice* device) : m_originDevice(device) {}
    virtual ~IBackendObject() = default;

    E_API_TYPE getAPIType() const;

    bool isCompatibleDevicewise(const IBackendObject* other) const
    {
        return (m_originDevice == other->m_originDevice);
    }

    bool wasCreatedBy(const ILogicalDevice* device) const
    {
        return m_originDevice == device;
    }

protected:
    const ILogicalDevice* getOriginDevice() const { return m_originDevice; }

private:
    const ILogicalDevice* const m_originDevice;
};

}
}

#endif
