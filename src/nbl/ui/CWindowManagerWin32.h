#ifndef _NBL_UI_C_WINDOWMANAGER_WIN32_INCLUDED_
#define _NBL_UI_C_WINDOWMANAGER_WIN32_INCLUDED_

#include "nbl/ui/IWindowManagerWin32.h"
#include "nbl/ui/IWindowWin32.h"
#include "nbl/ui/ICursorControl.h"

#ifdef _NBL_PLATFORM_WINDOWS_
namespace nbl::ui
{

class NBL_API2 CWindowManagerWin32 final : public IWindowManagerWin32, public ICursorControl
{
	public:
		inline CWindowManagerWin32() = default;

		SDisplayInfo getPrimaryDisplayInfo() const override final;

		core::smart_refctd_ptr<IWindow> createWindow(IWindow::SCreationParams&& creationParams) override final;

		inline void destroyWindow(IWindow* wnd) override final
		{
			CAsyncQueue::future_t<void> future;
			m_windowThreadManager.request(&future,SRequestParams_DestroyWindow{static_cast<IWindowWin32*>(wnd)->getNativeHandle()});
		}


		//! ICursorControl methods
		inline void setVisible(bool visible) override
		{
			CAsyncQueue::future_t<void> future;
			m_windowThreadManager.request(&future,SRequestParams_ChangeCursorVisibility{visible});
		}

		inline bool isVisible() const override
		{
			CURSORINFO ci = { sizeof(CURSORINFO) };
			GetCursorInfo(&ci);
			return ci.flags; // returning flags cause they're equal to 0 when the cursor is hidden
		}

		inline void setPosition(SPosition pos) override
		{
			SetCursorPos(pos.x,pos.y);
		}

		inline void setRelativePosition(IWindow* window, SRelativePosition pos) override
		{
			SPosition nativePos;
			int32_t w = window->getWidth();
			int32_t h = window->getHeight();
			nativePos.x = (pos.x/2.f + 0.5f) * w + window->getX();
			nativePos.y = (pos.y/2.f + 0.5f) * h + window->getY();
			SetCursorPos(nativePos.x, nativePos.y);
		}

		inline SPosition getPosition() override
		{
			POINT cursorPos;
			GetCursorPos(&cursorPos);
			return {cursorPos.x,cursorPos.y};
		}
		inline SRelativePosition getRelativePosition(IWindow* window) override
		{
			POINT cursorPos;
			GetCursorPos(&cursorPos);
			return {
				((cursorPos.x+0.5f-window->getX())/float(window->getWidth())-0.5f) * 2,
				((cursorPos.y+0.5f-window->getY())/float(window->getWidth())-0.5f) * 2
			};
		}

	protected:
		//! back to IWindowManager methods
		bool setWindowSize_impl(IWindow* window, const uint32_t width, const uint32_t height) override;

		inline bool setWindowPosition_impl(IWindow* window, const int32_t x, const int32_t y) override
		{
			CAsyncQueue::future_t<void> future;
			m_windowThreadManager.request(&future,SRequestParams_SetWindowPos{
				.nativeWindow = static_cast<IWindowWin32*>(window)->getNativeHandle(),
				.x = x,
				.y = y
			});
			return true;
		}

		inline bool setWindowRotation_impl(IWindow* window, const bool landscape) override
		{
			return false;
		}

		inline bool setWindowVisible_impl(IWindow* window, const bool visible) override
		{
			CAsyncQueue::future_t<void> future;
			m_windowThreadManager.request(&future,SRequestParams_ShowWindow{
				.nativeWindow = static_cast<IWindowWin32*>(window)->getNativeHandle(),
				.state = visible ? SRequestParams_ShowWindow::STATE::SHOW:SRequestParams_ShowWindow::STATE::HIDE
			});
			return true;
		}

		inline bool setWindowMaximized_impl(IWindow* window, const bool maximized) override
		{
			CAsyncQueue::future_t<void> future;
			m_windowThreadManager.request(&future,SRequestParams_ShowWindow{
				.nativeWindow = static_cast<IWindowWin32*>(window)->getNativeHandle(),
				.state = maximized ? SRequestParams_ShowWindow::STATE::MAXIMIZE:SRequestParams_ShowWindow::STATE::MINIMIZE
			});
			return true;
		}

	private:
		struct SRequestParams_NOOP
		{
			using retval_t = void;
			inline void operator()(core::StorageTrivializer<retval_t>* retval) {assert(false);}
		};
		struct SRequestParams_CreateWindow
		{
			using retval_t = IWindowWin32::native_handle_t;
			void operator()(core::StorageTrivializer<retval_t>* retval);

			std::string windowCaption;
			uint32_t width, height;
			int32_t x, y;
			IWindowWin32::E_CREATE_FLAGS flags;
		};
		struct SRequestParams_DestroyWindow
		{
			using retval_t = void;
			inline void operator()(core::StorageTrivializer<retval_t>* retval)
			{
				DestroyWindow(nativeWindow);
			}

			IWindowWin32::native_handle_t nativeWindow;
		};
		struct SRequestParams_ChangeCursorVisibility
		{
			using retval_t = void;
			void operator()(core::StorageTrivializer<retval_t>* retval);

			bool visible;
		};
		struct SRequestParams_SetWindowSize
		{
			using retval_t = void;
			inline void operator()(core::StorageTrivializer<retval_t>* retval)
			{
				SetWindowPos(nativeWindow, nullptr, 0xdeadbeef, 0xdeadbeef, width, height, SWP_NOACTIVATE|SWP_NOZORDER|SWP_NOMOVE|SWP_NOREPOSITION);
			}

			IWindowWin32::native_handle_t nativeWindow;
			int32_t width, height;
		};
		struct SRequestParams_SetWindowPos
		{
			using retval_t = void;
			inline void operator()(core::StorageTrivializer<retval_t>* retval)
			{
				SetWindowPos(nativeWindow, nullptr, x, y, 0xdeadbeefu, 0xdeadbeefu, SWP_NOACTIVATE|SWP_NOZORDER|SWP_NOSIZE);
			}

			IWindowWin32::native_handle_t nativeWindow;
			int x, y;
		};
		struct SRequestParams_ShowWindow
		{
			using retval_t = void;
			void operator()(core::StorageTrivializer<retval_t>* retval);

			IWindowWin32::native_handle_t nativeWindow;
			enum class STATE : uint8_t
			{
				HIDE,
				SHOW,
				MINIMIZE,
				MAXIMIZE
			};
			STATE state;
		};
		struct SRequest
		{
			std::variant<
				SRequestParams_NOOP,
				SRequestParams_CreateWindow,
				SRequestParams_DestroyWindow,
				SRequestParams_ChangeCursorVisibility,
				SRequestParams_SetWindowSize,
				SRequestParams_SetWindowPos,
				SRequestParams_ShowWindow
			> params = SRequestParams_NOOP();
		};
		static inline constexpr uint32_t CircularBufferSize = 256u;
        class CAsyncQueue final : public system::IAsyncQueueDispatcher<CAsyncQueue,SRequest,CircularBufferSize>
		{
				using base_t = system::IAsyncQueueDispatcher<CAsyncQueue,SRequest,CircularBufferSize>;

			public:
				inline CAsyncQueue() : base_t(base_t::start_on_construction)
				{
					//waitForInitComplete(); init is a NOOP
				}

				inline void init() {}

				inline bool wakeupPredicate() const { return true; }
				inline bool continuePredicate() const { return true; }
				
				void background_work();

                void process_request(base_t::future_base_t* _future_base, SRequest& req);
		} m_windowThreadManager;
};

}
#endif
#endif