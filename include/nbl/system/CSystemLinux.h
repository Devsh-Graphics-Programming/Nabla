#ifndef _NBL_SYSTEM_C_SYSTEM_LINUX_H_INCLUDED_
#define _NBL_SYSTEM_C_SYSTEM_LINUX_H_INCLUDED_


#include "nbl/system/ISystem.h"

namespace nbl::system
{
#ifdef _NBL_PLATFORM_LINUX_
#include "nbl/system/ISystemPOSIX.h"

class NBL_API CSystemLinux final : public ISystemPOSIX
{
	public:
		CSystemLinux() : ISystemPOSIX() {}

		SystemInfo getSystemInfo() const override;
};
#endif
}

#endif