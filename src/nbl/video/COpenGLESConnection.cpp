#include "nbl/video/COpenGLESConnection.h"

namespace nbl {
namespace video
{

core::smart_refctd_ptr<IAPIConnection> createOpenGLESConnection(core::smart_refctd_ptr<system::ISystem>&& sys, SDebugCallback* dbgCb, system::logger_opt_smart_ptr&& logger)
{
    return core::make_smart_refctd_ptr<COpenGLESConnection>(std::move(sys), dbgCb, std::move(logger));
}

}
}