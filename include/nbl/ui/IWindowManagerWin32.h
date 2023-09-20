#ifndef _NBL_UI_I_WINDOWMANAGER_WIN32_INCLUDED_
#define _NBL_UI_I_WINDOWMANAGER_WIN32_INCLUDED_

#include "nbl/ui/IWindowManager.h"

#ifdef _NBL_PLATFORM_WINDOWS_
namespace nbl::ui
{

class IWindowManagerWin32 : public IWindowManager
{
	public:
		NBL_API2 static core::smart_refctd_ptr<IWindowManagerWin32> create();
};

}
#endif
#endif