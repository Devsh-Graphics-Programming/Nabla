#ifndef _NBL_UI_C_WINDOW_XCB_H_INCLUDED_
#define _NBL_UI_C_WINDOW_XCB_H_INCLUDED_

#include "nbl/core/decl/smart_refctd_ptr.h"
#include "nbl/ui/IClipboardManagerXCB.h"
#include "nbl/ui/IWindowXCB.h"
#include "nbl/ui/XCBHandle.h"

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
	CWindowXCB(native_handle_t&& handle, core::smart_refctd_ptr<CWindowManagerXCB>&& winManager, SCreationParams&& params);
	~CWindowXCB();

	const native_handle_t* getNativeHandle() const override  {
		return &m_handle;
	}

	virtual IClipboardManager* getClipboardManager() override;
	virtual ICursorControl* getCursorControl() override;
	virtual IWindowManager* getManager() const override;

	virtual bool setWindowSize(uint32_t width, uint32_t height) override;
	virtual bool setWindowPosition(int32_t x, int32_t y) override;
	virtual bool setWindowRotation(bool landscape) override;
	virtual bool setWindowVisible(bool visible) override;
	virtual bool setWindowMaximized(bool maximized) override;

	virtual void setCaption(const std::string_view& caption) override;
	
private:
    CWindowXCB(core::smart_refctd_ptr<system::ISystem>&& sys, uint32_t _w, uint32_t _h, E_CREATE_FLAGS _flags);
	
	native_handle_t m_handle;
	core::smart_refctd_ptr<CWindowManagerXCB> m_windowManager;
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


};

}

#endif
