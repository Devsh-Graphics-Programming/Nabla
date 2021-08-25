#include "nbl/system/ISystem.h"
#include "nbl/system/CArchiveLoaderZip.h"

namespace nbl::system
{
    ISystem::ISystem(core::smart_refctd_ptr<ISystemCaller>&& caller) : m_dispatcher(this, std::move(caller))
    {
        addArchiveLoader(core::make_smart_refctd_ptr<CArchiveLoaderZip>());
    }
}
