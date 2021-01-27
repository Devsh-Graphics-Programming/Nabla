#ifndef __NBL_C_OPENGL_CONNECTION_H_INCLUDED__
#define __NBL_C_OPENGL_CONNECTION_H_INCLUDED__

#include "nbl/video/IAPIConnection.h"
#include "nbl/video/CEGLCaller.h"

namespace nbl {
namespace video
{

class COpenGLConnection final : public IAPIConnection
{
public:
    COpenGLConnection()
    {
        
    }

protected:
    ~COpenGLConnection()
    {

    }

private:
    // Note: EGL is not initialized here, each thread willing to use EGL will need to initialize it separately
    // (thats at least how our spoof EGL behaves, not sure if it's how EGL spec wants it)
    egl::CEGLCaller m_egl;
};

}
}

#endif