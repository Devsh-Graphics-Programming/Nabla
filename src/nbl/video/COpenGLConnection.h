#ifndef __NBL_C_OPENGL_CONNECTION_H_INCLUDED__
#define __NBL_C_OPENGL_CONNECTION_H_INCLUDED__

#include "nbl/video/IAPIConnection.h"

#include <SDL_video.h>

namespace nbl {
namespace video
{

class COpenGLConnection final : public IAPIConnection
{
public:
    COpenGLConnection()
    {
        // "offscreen" for windowless app
        //actually fuck SDL2
        // global state for whole process
        SDL_VideoInit("windows");
    }

protected:
    ~COpenGLConnection()
    {

    }
};

}
}

#endif