#ifndef __NBL_I_API_CONNECTION_H_INCLUDED__
#define __NBL_I_API_CONNECTION_H_INCLUDED__

#include "nbl/core/declarations.h"

#include "nbl/ui/IWindow.h"
#include "nbl/asset/utils/IGLSLCompiler.h"

#include "nbl/video/EApiType.h"
#include "nbl/video/debug/IDebugCallback.h"
#include "nbl/video/IPhysicalDevice.h"
#include "nbl/video/surface/ISurface.h"

namespace nbl::video
{

class IAPIConnection : public core::IReferenceCounted
{
    public:
        virtual E_API_TYPE getAPIType() const = 0;

        virtual core::SRange<const core::smart_refctd_ptr<IPhysicalDevice>> getPhysicalDevices() const = 0;

        virtual core::smart_refctd_ptr<ISurface> createSurface(ui::IWindow* window) const = 0;

        virtual IDebugCallback* getDebugCallback() const = 0;
};

}


#endif