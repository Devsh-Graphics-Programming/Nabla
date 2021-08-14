#ifndef __NBL_VIDEO_DEFINITION_I_BACKEND_OBJECT_H_INCLUDED__
#define __NBL_VIDEO_DEFINITION_I_BACKEND_OBJECT_H_INCLUDED__

namespace nbl::video
{

// public
inline IBackendObject::IBackendObject(core::smart_refctd_ptr<const ILogicalDevice>&& device) : m_originDevice(std::move(device)) {}
inline IBackendObject::~IBackendObject() {}


inline E_API_TYPE IBackendObject::getAPIType() const
{
    return m_originDevice->getAPIType();
}

inline bool IBackendObject::isCompatibleDevicewise(const IBackendObject* other) const
{
    return m_originDevice==other->m_originDevice;
}

inline bool IBackendObject::wasCreatedBy(const ILogicalDevice* device) const
{
    return m_originDevice.get()==device;
}

// protected
inline const ILogicalDevice* IBackendObject::getOriginDevice() const { return m_originDevice.get(); }


}

#endif
