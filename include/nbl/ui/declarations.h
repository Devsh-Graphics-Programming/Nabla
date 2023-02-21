// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_UI_DECLARATIONS_H_INCLUDED__
#define __NBL_UI_DECLARATIONS_H_INCLUDED__

// dependencies
#include "nbl/system/declarations.h"

// windows
#if defined(_NBL_PLATFORM_WINDOWS_)
#	include "nbl/ui/CWindowManagerWin32.h"
#elif defined(_NBL_BUILD_WITH_WAYLAND) && defined(_NBL_TEST_WAYLAND)
#	include "nbl/ui/CWindowManagerWayland.h"
#elif defined(_NBL_PLATFORM_LINUX_)
#	include "nbl/ui/CWindowManagerXcb.h"
#endif // TODO more platforms (android)

// clipboards
#include "nbl/ui/IClipboardManager.h"

// events
#include "nbl/ui/IInputEventChannel.h"


#endif