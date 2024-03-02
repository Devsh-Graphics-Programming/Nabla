#ifndef _NBL_UI_C_CLIPBOARD_MANAGER_XCB_INCLUDED_
#define _NBL_UI_C_CLIPBOARD_MANAGER_XCB_INCLUDED_

#ifdef _NBL_PLATFORM_LINUX_

#include "nbl/core/decl/Types.h"
#include "nbl/ui/IClipboardManagerXCB.h"
namespace nbl::ui
{

class IWindowXCB;
class XCBConnection;

// details on XCB clipboard protocol: https://tronche.com/gui/x/icccm/sec-2.html#s-2
// class NBL_API2 CClipboardManagerXCB final : public IClipboardManagerXCB
// {
//     public:
//         inline CClipboardManagerXCB(core::smart_refctd_ptr<XCBConnection>&& connect): 
//             IClipboardManagerXCB(),
//             m_connection(std::move(connect)) {}

//         virtual std::string getClipboardText() override;
//         virtual bool setClipboardText(const std::string_view& data) override;

//         void process(const IWindowXCB* window, xcb_generic_event_t* event) override;
//     private:
//         core::smart_refctd_ptr<XCBConnection> m_connection;
//         std::mutex m_clipboardMutex;
//         std::condition_variable m_clipboardResponseCV;
//         std::string m_clipboardResponse; // data sent to the clipboard by another application

//         std::string m_savedClipboard; // data saved to the clipboard for another application to read

// };

}

#endif

#endif