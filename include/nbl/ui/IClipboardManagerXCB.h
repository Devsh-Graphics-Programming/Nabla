#ifndef _NBL_UI_I_CLIPBOARD_MANAGER_XCB_INCLUDED_
#define _NBL_UI_I_CLIPBOARD_MANAGER_XCB_INCLUDED_

#ifdef _NBL_PLATFORM_LINUX_

#include "nbl/ui/IClipboardManager.h"

namespace nbl::ui
{
class XCBConnection;

// details on XCB clipboard protocol: https://tronche.com/gui/x/icccm/sec-2.html#s-2
class NBL_API2 IClipboardManagerXCB : public IClipboardManager
{
    public:
        IClipboardManagerXCB() : IClipboardManager() {}
        virtual ~IClipboardManagerXCB() = default;

        virtual std::string getClipboardText() = 0;
        virtual bool setClipboardText(const std::string_view& data) = 0;
        virtual void process(const IWindowXCB* window, xcb_generic_event_t* event) = 0;
};

}

#endif

#endif