#ifndef _NBL_UI_C_WINDOW_XCB_H_INCLUDED_
#define _NBL_UI_C_WINDOW_XCB_H_INCLUDED_

#include "nbl/core/decl/smart_refctd_ptr.h"
#include "nbl/ui/IClipboardManagerXCB.h"
#include "nbl/ui/IWindowXCB.h"
#include "nbl/ui/XCBConnection.h"

#include <cstdlib>

namespace nbl::ui
{

class CWindowManagerXCB;
class XCBConnection;
class CCursorControlXCB;
class IClipboardManagerXCB;

class NBL_API2 CWindowXCB final : public IWindowXCB
{
	
public:
	CWindowXCB(core::smart_refctd_ptr<CWindowManagerXCB>&& winManager, SCreationParams&& params);
	~CWindowXCB();

	const native_handle_t* getNativeHandle() const override  {
		return &m_handle;
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
    CWindowXCB(core::smart_refctd_ptr<system::ISystem>&& sys, uint32_t _w, uint32_t _h, E_CREATE_FLAGS _flags);
	
	core::smart_refctd_ptr<CWindowManagerXCB> m_windowManager;
	core::smart_refctd_ptr<XCBConnection> m_connection;
	core::smart_refctd_ptr<CCursorControlXCB> m_cursorControl;
	core::smart_refctd_ptr<IClipboardManagerXCB> m_clipboardManager;

	class CDispatchThread final : public system::IThreadHandler<CDispatchThread>
	{
	public:
		using base_t = system::IThreadHandler<CDispatchThread>;

		inline CDispatchThread(CWindowXCB& window) : 
			base_t(base_t::start_on_construction_t {}),
			m_window(window) {

		}
		inline ~CDispatchThread() {
		}

		inline void init() {}
		inline void exit() {}
		void work(lock_t& lock);

		inline bool wakeupPredicate() const { return true; }
		inline bool continuePredicate() const { return true; }
	private:
		CWindowXCB& m_window;
		friend class CWindowXCB;
	} m_dispatcher;
	
	native_handle_t m_handle = {{0}};

	XCBConnection::XCBAtomToken<core::StringLiteral("WM_DELETE_WINDOW")> m_WM_DELETE_WINDOW;
	XCBConnection::XCBAtomToken<core::StringLiteral("WM_PROTOCOLS")> m_WM_PROTOCOLS;
	XCBConnection::XCBAtomToken<core::StringLiteral("_NET_WM_PING")> m_NET_WM_PING;
	
	XCBConnection::XCBAtomToken<core::StringLiteral("_NET_WM_STATE_MAXIMIZED_VERT")> m_NET_WM_STATE_MAXIMIZED_VERT;
	XCBConnection::XCBAtomToken<core::StringLiteral("_NET_WM_STATE_MAXIMIZED_HORZ")> m_NET_WM_STATE_MAXIMIZED_HORZ;
	XCBConnection::XCBAtomToken<core::StringLiteral("_NET_WM_STATE_FULLSCREEN")> m_NET_WM_STATE_FULLSCREEN;

};

}

#endif
