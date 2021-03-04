#include "nbl/video/COpenGLConnection.h"

namespace nbl {
namespace video
{

core::smart_refctd_ptr<IAPIConnection> createOpenGLConnection()
{
    return core::make_smart_refctd_ptr<COpenGLConnection>();
}

}
}