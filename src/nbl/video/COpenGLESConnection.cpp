#include "nbl/video/COpenGLESConnection.h"

namespace nbl {
namespace video
{

core::smart_refctd_ptr<IAPIConnection> createOpenGLESConnection()
{
    return core::make_smart_refctd_ptr<COpenGLESConnection>();
}

}
}