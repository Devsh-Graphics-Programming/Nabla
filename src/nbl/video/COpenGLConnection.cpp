#include "nbl/video/COpenGLConnection.h"

namespace nbl::video
{

core::smart_refctd_ptr<IAPIConnection> createOpenGLConnection(core::smart_refctd_ptr<system::ISystem>&& sys, SDebugCallback* dbgCb, system::logger_opt_smart_ptr&& logger)
{
    return core::make_smart_refctd_ptr<COpenGLConnection>(std::move(sys), dbgCb, std::move(logger));
}


}