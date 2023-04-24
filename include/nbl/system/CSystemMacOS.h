#ifndef _NBL_SYSTEM_C_SYSTEM_MACOS_H_INCLUDED_
#define _NBL_SYSTEM_C_SYSTEM_MACOS_H_INCLUDED_

#include "nbl/system/ISystem.h"
#include "nbl/system/ISystemPOSIX.h"

namespace nbl::system
{
#ifdef _NBL_PLATFORM_MACOS_

class CSystemMacOS final : public ISystemPOSIX
{
	public:
		inline CSystemMacOS() : ISystemPOSIX() {}

		NBL_API2 SystemInfo getSystemInfo() const override;
};
#endif // _NBL_PLATFORM_MACOS_
}

#endif // _NBL_SYSTEM_C_SYSTEM_MACOS_H_INCLUDED_
