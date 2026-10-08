#ifndef _NBL_UI_C_WINDOWMANAGER_XCB_INCLUDED_
#define _NBL_UI_C_WINDOWMANAGER_XCB_INCLUDED_

#include "nbl/ui/IWindowManagerXcb.h"
#include "nbl/ui/IWindowXcb.h"

#ifdef _NBL_PLATFORM_LINUX_
#include <xcb/xcb.h>

#include <atomic>
#include <thread>

namespace nbl::ui
{

class CWindowXcb;

// See `IWindowManagerXcb` for the threading and lifetime rules
class NBL_API2 CWindowManagerXcb final : public IWindowManagerXcb
{
	public:
		struct SAtoms
		{
			xcb_atom_t WM_PROTOCOLS;
			xcb_atom_t WM_DELETE_WINDOW;
			xcb_atom_t WM_STATE;
			xcb_atom_t WM_CHANGE_STATE;
			xcb_atom_t UTF8_STRING;
			xcb_atom_t _NET_WM_NAME;
			xcb_atom_t _NET_WM_STATE;
			xcb_atom_t _NET_WM_STATE_HIDDEN;
			xcb_atom_t _NET_WM_STATE_MAXIMIZED_VERT;
			xcb_atom_t _NET_WM_STATE_MAXIMIZED_HORZ;
			xcb_atom_t _NET_WM_STATE_FULLSCREEN;
			xcb_atom_t _NET_WM_STATE_ABOVE;
			xcb_atom_t _MOTIF_WM_HINTS;
		};

		// takes ownership of the connection
		CWindowManagerXcb(xcb_connection_t* connection, xcb_screen_t* screen, const SAtoms& atoms);

		SDisplayInfo getPrimaryDisplayInfo() const override final;

		core::smart_refctd_ptr<IWindow> createWindow(IWindow::SCreationParams&& creationParams) override final;

		void destroyWindow(IWindow* wnd) override final;

		inline xcb_connection_t* getConnection() const {return m_connection.handle;}
		inline const SAtoms& getAtoms() const {return m_atoms;}
		inline xcb_window_t getRootWindow() const {return m_screen->root;}

		// ICCCM `WM_NORMAL_HINTS`, used to lock the size of windows the user can't resize
		void setSizeHints(xcb_window_t window, int32_t x, int32_t y, uint32_t width, uint32_t height, bool fixedSize) const;

	protected:
		~CWindowManagerXcb() override = default;

		bool setWindowSize_impl(IWindow* window, const uint32_t width, const uint32_t height) override;
		bool setWindowPosition_impl(IWindow* window, const int32_t x, const int32_t y) override;
		inline bool setWindowRotation_impl(IWindow* window, const bool landscape) override {return false;}
		bool setWindowVisible_impl(IWindow* window, const bool visible) override;
		bool setWindowMaximized_impl(IWindow* window, const bool maximized) override;

	private:
		// EWMH and ICCCM requests to the window manager are client messages sent to the root window
		void sendToRoot(xcb_window_t window, xcb_atom_t type, const uint32_t (&data)[5]) const;

		// everything below runs on the event thread
		xcb_window_t createNativeWindow(const IWindow::SCreationParams& params);
		void registerWindow(CWindowXcb* window, const bool map);
		void unregisterWindow(xcb_window_t window);
		void dispatchEvent(const xcb_generic_event_t* event);
		CWindowXcb* findWindow(xcb_window_t window) const;
		// `findWindow` that also remembers which window's callbacks are about to run
		inline CWindowXcb* beginDispatch(xcb_window_t window)
		{
			CWindowXcb* const found = findWindow(window);
			m_dispatchingWindow = found;
			return found;
		}

		// Declared first so it's destroyed last, after the event thread has stopped
		struct SConnection
		{
			inline ~SConnection()
			{
				if (handle)
					xcb_disconnect(handle);
			}

			xcb_connection_t* handle;
		} m_connection;
		xcb_screen_t* const m_screen;
		const SAtoms m_atoms;
		// only touched on the event thread
		core::unordered_map<xcb_window_t,CWindowXcb*> m_windows;
		const CWindowXcb* m_dispatchingWindow = nullptr;

		struct SRequestParams_NOOP
		{
			using retval_t = void;
		};
		struct SRequestParams_CreateWindow
		{
			using retval_t = xcb_window_t;
			const IWindow::SCreationParams* params;
		};
		struct SRequestParams_RegisterWindow
		{
			using retval_t = void;
			CWindowXcb* window;
			bool map;
		};
		struct SRequestParams_DestroyWindow
		{
			using retval_t = void;
			xcb_window_t window;
		};
		struct SRequest
		{
			std::variant<
				SRequestParams_NOOP,
				SRequestParams_CreateWindow,
				SRequestParams_RegisterWindow,
				SRequestParams_DestroyWindow
			> params = SRequestParams_NOOP();
		};
		static inline constexpr uint32_t CircularBufferSize = 256u;
		class CAsyncQueue final : public system::IAsyncQueueDispatcher<CAsyncQueue,SRequest,CircularBufferSize>
		{
				using base_t = system::IAsyncQueueDispatcher<CAsyncQueue,SRequest,CircularBufferSize>;

			public:
				// unlike Win32 we have members the thread reads, so only start it once they're initialized
				inline CAsyncQueue(CWindowManagerXcb* manager) : base_t(), m_manager(manager)
				{
					this->start();
					this->waitForInitComplete();
				}
				inline ~CAsyncQueue()
				{
					this->shutdown();
				}

				inline void init() {m_threadID = std::this_thread::get_id();}

				// like the Win32 manager, keep spinning so `background_work` can poll the X connection
				inline bool wakeupPredicate() const { return true; }
				inline bool continuePredicate() const { return true; }

				void background_work();

				void process_request(base_t::future_base_t* _future_base, SRequest& req);

				inline bool isEventThread() const {return std::this_thread::get_id()==m_threadID;}

			private:
				CWindowManagerXcb* const m_manager;
				std::thread::id m_threadID;
		};
		// Declared last so the event thread starts after all the other members are initialized
		CAsyncQueue m_eventThread;
};

}
#endif
#endif
