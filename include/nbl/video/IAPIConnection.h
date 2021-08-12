#ifndef __NBL_I_API_CONNECTION_H_INCLUDED__
#define __NBL_I_API_CONNECTION_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"
#include "nbl/video/IPhysicalDevice.h"
#include "nbl/video/EApiType.h"
#include "nbl/video/surface/ISurface.h"
#include "nbl/ui/IWindow.h"
#include "nbl/asset/utils/IGLSLCompiler.h"
#include "nbl/video/debug/debug.h"

namespace nbl {
namespace video
{

class IAPIConnection : public core::IReferenceCounted
{
public:
    static core::smart_refctd_ptr<IAPIConnection> create(core::smart_refctd_ptr<system::ISystem>&& sys, E_API_TYPE apiType, uint32_t appVer, const char* appName, SDebugCallback* dbgCb = nullptr, system::logger_opt_smart_ptr&& logger = nullptr);

    virtual E_API_TYPE getAPIType() const = 0;

    virtual core::SRange<const core::smart_refctd_ptr<IPhysicalDevice>> getPhysicalDevices() const = 0;

    virtual core::smart_refctd_ptr<ISurface> createSurface(ui::IWindow* window) const = 0;

protected:
    IAPIConnection(core::smart_refctd_ptr<system::ISystem>&& sys, system::logger_opt_smart_ptr&& logger);
    virtual ~IAPIConnection() = default;

    // idk where to put those, so here they are for now
    core::smart_refctd_ptr<system::ISystem> m_system;
    core::smart_refctd_ptr<asset::IGLSLCompiler> m_GLSLCompiler;
    system::logger_opt_smart_ptr m_logger;
};

}
}


#endif