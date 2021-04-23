#ifndef __NBL_C_OPENGL_EVENT_H_INCLUDED__
#define __NBL_C_OPENGL_EVENT_H_INCLUDED__

#include "nbl/video/IGPUEvent.h"

namespace nbl {
namespace video
{

// for now impl of COpenGLEvent doesnt include any COpenGLSync since 
// we dont support no DEVICE_ONLY events yet
// and so no waiting GL calls are required
class COpenGLEvent : public IGPUEvent
{
public:
    using IGPUEvent::IGPUEvent;
};

}}

#endif