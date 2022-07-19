#ifndef _NBL_SYSTEM_C_SYSTEM_LINUX_H_INCLUDED_
#define _NBL_SYSTEM_C_SYSTEM_LINUX_H_INCLUDED_


#include "nbl/system/ISystem.h"
#include "nbl/system/ISystemPOSIX.h"

namespace nbl::system
{
#ifdef _NBL_PLATFORM_LINUX_

class CSystemLinux final : public ISystemPOSIX
{
	public:
		inline CSystemLinux() : ISystemPOSIX() {}

		NBL_API2 SystemInfo getSystemInfo() const override;
};
#endif
}

#endif