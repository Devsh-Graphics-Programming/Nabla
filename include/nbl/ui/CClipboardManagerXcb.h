#ifndef _NBL_UI_C_CLIPBOARD_MANAGER_XCB_INCLUDED_
#define _NBL_UI_C_CLIPBOARD_MANAGER_XCB_INCLUDED_

#include <string>
#include <vector>
#include <xcb/xproto.h>
#ifdef _NBL_PLATFORM_LINUX_

#include "nbl/core/decl/Types.h"
#include "nbl/ui/IClipboardManager.h"
#include "nbl/ui/XcbConnection.h"

namespace nbl::ui
{

// details on XCB clipboard protocol: https://tronche.com/gui/x/icccm/sec-2.html#s-2
class NBL_API2 CClipboardManagerXcb final : public IClipboardManager
{
		using base_t = IClipboardManager;
	public:
		inline CClipboardManagerXcb(core::smart_refctd_ptr<XcbConnection>&& connect): 
			m_xcbConnection(std::move(connect)) {}

		virtual std::string getClipboardText() override;
		virtual bool setClipboardText(const std::string_view& data) override;

		void process(const IWindowXcb* window, xcb_generic_event_t* event);
	private:
		core::smart_refctd_ptr<XcbConnection> m_xcbConnection;

		xcb_window_t getClipboardWindow();

		struct {
			std::string m_data;
			std::vector<xcb_atom_t> m_formats;
		} m_stagedClipboard;
		
		std::mutex m_clipboardMutex;
		std::condition_variable m_clipboardResponseCV;
		std::string m_clipboardResponse;
		// bool ready = false;

		XcbConnection::XCBAtomToken<core::StringLiteral("CLIPBOARD")> m_CLIPBOARD;
		XcbConnection::XCBAtomToken<core::StringLiteral("TARGETS")> m_TARGETS;
		XcbConnection::XCBAtomToken<core::StringLiteral("INCR")> m_INCR;
		

		XcbConnection::XCBAtomToken<core::StringLiteral("UTF8_STRING")> m_formatUTF8_0;
		XcbConnection::XCBAtomToken<core::StringLiteral("text/plain;charset=utf-8")> m_formatUTF8_1;
		XcbConnection::XCBAtomToken<core::StringLiteral("text/plain;charset=UTF-8")> m_formatUTF8_2;
		XcbConnection::XCBAtomToken<core::StringLiteral("GTK_TEXT_BUFFER_CONTENTS")> m_formatGTK;
		XcbConnection::XCBAtomToken<core::StringLiteral("STRING")> m_formatString;
		XcbConnection::XCBAtomToken<core::StringLiteral("TEXT")> m_formatText;
		XcbConnection::XCBAtomToken<core::StringLiteral("text/plain")> m_formatTextPlain;
		

};

}

#endif

#endif