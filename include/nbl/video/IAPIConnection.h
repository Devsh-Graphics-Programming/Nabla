#ifndef __NBL_I_API_CONNECTION_H_INCLUDED__
#define __NBL_I_API_CONNECTION_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"
#include "nbl/video/IPhysicalDevice.h"
#include "nbl/video/EApiType.h"
#include "nbl/video/surface/ISurface.h"
#include "nbl/ui/IWindow.h"
#include "IFileSystem.h"
#include "nbl/asset/utils/IGLSLCompiler.h"
#include "nbl/video/debug/debug.h"

namespace nbl {
namespace video
{

class IAPIConnection : public core::IReferenceCounted
{
public:
    static core::smart_refctd_ptr<IAPIConnection> create(E_API_TYPE apiType, uint32_t appVer, const char* appName, const SDebugCallback& dbgCb);

    virtual E_API_TYPE getAPIType() const = 0;

    virtual core::SRange<const core::smart_refctd_ptr<IPhysicalDevice>> getPhysicalDevices() const = 0;

    virtual core::smart_refctd_ptr<ISurface> createSurface(ui::IWindow* window) const = 0;

protected:
    IAPIConnection(const SDebugCallback& dbgCb);
    virtual ~IAPIConnection() = default;

    // idk where to put those, so here they are for now
    core::smart_refctd_ptr<io::IFileSystem> m_fs;
    core::smart_refctd_ptr<asset::IGLSLCompiler> m_GLSLCompiler;

    SDebugCallback m_debugCallback;
};

}
}


#endif