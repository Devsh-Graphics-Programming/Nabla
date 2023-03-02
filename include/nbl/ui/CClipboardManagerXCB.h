#ifndef _NBL_UI_C_CLIPBOARD_MANAGER_XCB_INCLUDED_
#define _NBL_UI_C_CLIPBOARD_MANAGER_XCB_INCLUDED_

#ifdef _NBL_PLATFORM_LINUX_

#include "nbl/core/decl/Types.h"
#include "nbl/ui/IClipboardManagerXCB.h"
#include "nbl/ui/XCBConnection.h"
namespace nbl::ui
{

class IWindowXCB;
class XCBConnection;

// details on XCB clipboard protocol: https://tronche.com/gui/x/icccm/sec-2.html#s-2
class NBL_API2 CClipboardManagerXCB final : public IClipboardManagerXCB
{
    public:
        inline CClipboardManagerXCB(core::smart_refctd_ptr<XCBConnection>&& connect): 
            IClipboardManagerXCB(),
            m_connection(std::move(connect)) {}

        virtual std::string getClipboardText() override;
        virtual bool setClipboardText(const std::string_view& data) override;

        void process(const IWindowXCB* window, xcb_generic_event_t* event) override;
    private:
        core::smart_refctd_ptr<XCBConnection> m_connection;
        std::mutex m_clipboardMutex;
        std::condition_variable m_clipboardResponseCV;
        std::string m_clipboardResponse; // data sent to the clipboard by another application

        std::string m_savedClipboard; // data saved to the clipboard for another application to read

        XCBConnection::XCBAtomToken<core::StringLiteral("CLIPBOARD")> m_CLIPBOARD;
        XCBConnection::XCBAtomToken<core::StringLiteral("TARGETS")> m_TARGETS;
        XCBConnection::XCBAtomToken<core::StringLiteral("INCR")> m_INCR;

        XCBConnection::XCBAtomToken<core::StringLiteral("UTF8_STRING")> m_formatUTF8_0;
        XCBConnection::XCBAtomToken<core::StringLiteral("text/plain;charset=utf-8")> m_formatUTF8_1;
        XCBConnection::XCBAtomToken<core::StringLiteral("text/plain;charset=UTF-8")> m_formatUTF8_2;
        XCBConnection::XCBAtomToken<core::StringLiteral("GTK_TEXT_BUFFER_CONTENTS")> m_formatGTK;
        XCBConnection::XCBAtomToken<core::StringLiteral("STRING")> m_formatString;
        XCBConnection::XCBAtomToken<core::StringLiteral("TEXT")> m_formatText;
        XCBConnection::XCBAtomToken<core::StringLiteral("text/plain")> m_formatTextPlain;
};

}

#endif

#endif