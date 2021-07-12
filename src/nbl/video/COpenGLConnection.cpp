#include "nbl/video/COpenGLConnection.h"

namespace nbl {
namespace video
{

core::smart_refctd_ptr<IAPIConnection> createOpenGLConnection(core::smart_refctd_ptr<system::ISystem>&& sys, SDebugCallback* dbgCb)
{
    return core::make_smart_refctd_ptr<COpenGLConnection>(std::move(sys), dbgCb);
}

}
}