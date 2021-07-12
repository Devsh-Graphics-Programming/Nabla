#include "nbl/system/ISystem.h"
#include "nbl/system/CSystemWin32.h"
namespace nbl::system
{
core::smart_refctd_ptr<ISystem> nbl::system::ISystem::create()
{
	core::smart_refctd_ptr<ISystem::ISystemCaller> caller = nullptr;
#ifdef _NBL_PLATFORM_WINDOWS_
	caller = core::make_smart_refctd_ptr<CSystemCallerWin32>();
#endif
	return core::make_smart_refctd_ptr<ISystem>(std::move(caller));
}
}
