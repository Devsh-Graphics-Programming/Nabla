#ifndef __NBL_C_OPENGL_EVENT_H_INCLUDED__
#define __NBL_C_OPENGL_EVENT_H_INCLUDED__

#include "nbl/video/IGPUEvent.h"
#include "COpenGLExtensionHandler.h"
#include "nbl/video/COpenGLSync.h"

namespace nbl {
namespace video
{

class COpenGLEvent : public IGPUEvent
{
public:
    using IGPUEvent::IGPUEvent;

private:
    core::smart_refctd_ptr<COpenGLSync> m_syncSignal, m_syncUnsignal;
};

}}

#endif