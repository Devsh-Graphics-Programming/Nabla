#ifndef __C_WINDOW_XCB_H_INCLUDED__
#define __C_WINDOW_XCB_H_INCLUDED__

#include <cstdlib>

#include "nbl/core/decl/smart_refctd_ptr.h"
#include "nbl/ui/CClipboardManagerXcb.h"
#include "nbl/ui/IWindowXcb.h"
#include "nbl/ui/XcbConnection.h"

namespace nbl::ui
{

class CCursorControlXcb;
class CWindowManagerXcb;
class CClipboardManagerXcb;

class NBL_API2 CWindowXcb final : public IWindowXcb
{
	
public:
	CWindowXcb(core::smart_refctd_ptr<CWindowManagerXcb>&& winManager, SCreationParams&& params);
	~CWindowXcb();

	// Display* getDisplay() const override { return m_dpy; }
	xcb_window_t getXcbWindow() const override { return m_xcbWindow; }
	xcb_connection_t* getXcbConnection() const override {
		return m_xcbConnection->getRawConnection();
	}

	virtual IClipboardManager* getClipboardManager() override;
	virtual ICursorControl* getCursorControl() override;
	virtual IWindowManager* getManager() override;

	virtual bool setWindowSize_impl(uint32_t width, uint32_t height) override;
	virtual bool setWindowPosition_impl(int32_t x, int32_t y) override;
	virtual bool setWindowRotation_impl(bool landscape) override;
	virtual bool setWindowVisible_impl(bool visible) override;
	virtual bool setWindowMaximized_impl(bool maximized) override;

	virtual void setCaption(const std::string_view& caption) override;
	
private:
    CWindowXcb(core::smart_refctd_ptr<system::ISystem>&& sys, uint32_t _w, uint32_t _h, E_CREATE_FLAGS _flags);
	
	core::smart_refctd_ptr<CWindowManagerXcb> m_windowManager;
	core::smart_refctd_ptr<XcbConnection> m_xcbConnection;
	core::smart_refctd_ptr<CCursorControlXcb> m_cursorControl;
	core::smart_refctd_ptr<CClipboardManagerXcb> m_clipboardManager;

	class CDispatchThread final : public system::IThreadHandler<CDispatchThread>
	{
	public:

		inline CDispatchThread(CWindowXcb& window);
		inline ~CDispatchThread() {
		}

		void init();
		void exit();
		void work(lock_t& lock);

		inline bool wakeupPredicate() const { return true; }
		inline bool continuePredicate() const { return true; }
	private:
		CWindowXcb& m_window;
		// xcb_connection_t* m_connection = nullptr;
		friend class CWindowXcb;
	} m_dispatcher;

	xcb_window_t m_xcbWindow = 0;

	XcbConnection::XCBAtomToken<core::StringLiteral("WM_DELETE_WINDOW")> m_WM_DELETE_WINDOW;
	XcbConnection::XCBAtomToken<core::StringLiteral("WM_PROTOCOLS")> m_WM_PROTOCOLS;
	XcbConnection::XCBAtomToken<core::StringLiteral("_NET_WM_PING")> m_NET_WM_PING;
	
	XcbConnection::XCBAtomToken<core::StringLiteral("_NET_WM_STATE_MAXIMIZED_VERT")> m_NET_WM_STATE_MAXIMIZED_VERT;
	XcbConnection::XCBAtomToken<core::StringLiteral("_NET_WM_STATE_MAXIMIZED_HORZ")> m_NET_WM_STATE_MAXIMIZED_HORZ;
	XcbConnection::XCBAtomToken<core::StringLiteral("_NET_WM_STATE_FULLSCREEN")> m_NET_WM_STATE_FULLSCREEN;

};

}

#endif
