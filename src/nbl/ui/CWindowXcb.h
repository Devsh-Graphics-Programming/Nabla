#ifndef _NBL_UI_C_WINDOW_XCB_H_INCLUDED_
#define _NBL_UI_C_WINDOW_XCB_H_INCLUDED_

#include "nbl/ui/CWindowManagerXcb.h"

#ifdef _NBL_PLATFORM_LINUX_
namespace nbl::ui
{

class NBL_API2 CWindowXcb final : public IWindowXcb
{
	public:
		CWindowXcb(SCreationParams&& params, core::smart_refctd_ptr<CWindowManagerXcb>&& winManager, xcb_window_t window);

		inline const native_handle_t& getNativeHandle() const override {return m_native;}

		void setCaption(const std::string_view& caption) override;

		// TODO: clipboard, cursor and input channels are the follow-up to NAB-9
		inline IClipboardManager* getClipboardManager() override {return nullptr;}
		inline ICursorControl* getCursorControl() const override {return nullptr;}

		inline IWindowManager* getManager() const override {return m_windowManager.get();}

	protected:
		inline ~CWindowXcb() override
		{
			m_windowManager->destroyWindow(this);
		}

	private:
		friend class CWindowManagerXcb;

		// all of these run on the event thread of the manager
		void onDeleteRequested();
		void onConfigured(const xcb_configure_notify_event_t* ev);
		void onMapped();
		void onUnmapped();
		void onFocusChanged(const bool keyboard, const bool gained);
		// re-reads `WM_STATE` and `_NET_WM_STATE` to detect minimize, maximize and restore
		void updateState();

		core::smart_refctd_ptr<CWindowManagerXcb> m_windowManager;
		native_handle_t m_native;
		// what the X server and window manager last told us, flags in `IWindow` can disagree when a callback vetoed a change
		bool m_minimizedState = false;
		bool m_maximizedState = false;
};

}
#endif

#endif
