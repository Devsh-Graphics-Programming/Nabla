#ifndef _NBL_I_WINDOW_H_INCLUDED_
#define _NBL_I_WINDOW_H_INCLUDED_

#include "nbl/core/IReferenceCounted.h"

#include <type_traits>

#include "nbl/system/ISystem.h"

#include "nbl/ui/IClipboardManager.h"
#include "nbl/ui/IInputEventChannel.h"

namespace nbl::ui
{

class IWindowManager;
class ICursorControl;

class IWindow : public core::IReferenceCounted
{
    public:
        enum E_CREATE_FLAGS : uint32_t
        {
            ECF_FULLSCREEN = 1u << 0,
            ECF_HIDDEN = 1u << 1,
            ECF_BORDERLESS = 1u << 2,
            ECF_RESIZABLE = 1u << 3,
            ECF_MINIMIZED = 1u << 4,
            ECF_MAXIMIZED = 1u << 5,
            //! Forces mouse to stay inside the window
            ECF_MOUSE_CAPTURE = 1u << 6,
            //! Indicates whether the window is active or not
            ECF_INPUT_FOCUS = 1u << 7,
            //! Indicates whether mouse is hovering over the window even if the window is not active
            ECF_MOUSE_FOCUS = 1u << 8,
            ECF_ALWAYS_ON_TOP = 1u << 9,
            //! If disabled, the maximize button is grayed out
            ECF_CAN_MAXIMIZE = 1u << 10,
            //! If disabled, the minimize button is grayed out
            ECF_CAN_MINIMIZE = 1u << 11,
            //! If disabled, the window can't be resized via the UI, only programmatically
            ECF_CAN_RESIZE = 1u << 12,

            ECF_NONE = 0
        };

        class IEventCallback : public core::IReferenceCounted
        {
            public:
                // TODO: rethink our boolean returns
                [[nodiscard]] inline bool onWindowShown(IWindow* w)
                {
                    auto canShow = onWindowShown_impl();
                    if(canShow)
                    {
                        w->m_flags &= (~core::bitflag(ECF_HIDDEN));
                    }
                    return canShow;
                }
                [[nodiscard]] inline bool onWindowHidden(IWindow* w)
                {
                    auto canHide = onWindowHidden_impl();
                    if (canHide)
                    {
                        w->m_flags |= ECF_HIDDEN;
                    }
                    return canHide;
                }
                [[nodiscard]] inline bool onWindowMoved(IWindow* w, int32_t x, int32_t y)
                {
                    auto canMove = onWindowMoved_impl(x, y);
                    if (canMove)
                    {
                        w->m_x = x;
                        w->m_y = y;
                    }
                    return canMove;
                }
                [[nodiscard]] inline bool onWindowResized(IWindow* w, uint32_t width, uint32_t height)
                {
                    auto canResize = onWindowResized_impl(width, height);
                    if (canResize)
                    {
                        w->m_width = width;
                        w->m_height = height;
                    }
                    return canResize;
                }
                [[nodiscard]] inline bool onWindowRotated(IWindow* w)
                {
                    return onWindowRotated_impl();
                }
                [[nodiscard]] inline bool onWindowMinimized(IWindow* w)
                {
                    auto canMinimize = onWindowMinimized_impl();
                    if (canMinimize)
                    {
                        w->m_flags |= ECF_MINIMIZED;
                        w->m_flags &= (~core::bitflag(ECF_MAXIMIZED));
                    }
                    return canMinimize;
                }
                [[nodiscard]] inline bool onWindowMaximized(IWindow* w)
                {
                    auto canMaximize = onWindowMaximized_impl();
                    if (canMaximize)
                    {
                        w->m_flags |= ECF_MAXIMIZED;
                        w->m_flags &= (~core::bitflag(ECF_MINIMIZED));
                    }
                    return canMaximize;
                }
                [[nodiscard]] inline bool onWindowClosed(IWindow* w)
                {
                    return onWindowClosed_impl();
                }

                inline void onGainedMouseFocus(IWindow* w)
                {
                    onGainedMouseFocus_impl();
                }
                inline void onLostMouseFocus(IWindow* w)
                {
                    onLostMouseFocus_impl();
                }
                inline void onGainedKeyboardFocus(IWindow* w)
                {
                    onGainedKeyboardFocus_impl();
                }
                inline void onLostKeyboardFocus(IWindow* w)
                {
                    onLostKeyboardFocus_impl();
                }

                inline void onMouseConnected(IWindow* w, core::smart_refctd_ptr<IMouseEventChannel>&& mch)
                {
                    onMouseConnected_impl(std::move(mch));
                }
                inline void onMouseDisconnected(IWindow* w, IMouseEventChannel* mch)
                {
                    onMouseDisconnected_impl(mch);
                }
                inline void onKeyboardConnected(IWindow* w, core::smart_refctd_ptr<IKeyboardEventChannel>&& kbch)
                {
                    onKeyboardConnected_impl(std::move(kbch));
                }
                inline void onKeyboardDisconnected(IWindow* w, IKeyboardEventChannel* kbch)
                {
                    onKeyboardDisconnected_impl(kbch);
                }
        
            protected:
                NBL_API2 virtual bool onWindowShown_impl() { return true; }
                NBL_API2 virtual bool onWindowHidden_impl() { return true; }
                NBL_API2 virtual bool onWindowMoved_impl(int32_t x, int32_t y) { return true; }
                NBL_API2 virtual bool onWindowResized_impl(uint32_t w, uint32_t h) { return true; }
                NBL_API2 virtual bool onWindowRotated_impl() { return true; }
                NBL_API2 virtual bool onWindowMinimized_impl() { return true; }
                NBL_API2 virtual bool onWindowMaximized_impl() { return true; }
                NBL_API2 virtual bool onWindowClosed_impl() { return true; }
                NBL_API2 virtual void onGainedMouseFocus_impl() {}
                NBL_API2 virtual void onLostMouseFocus_impl() {}
                NBL_API2 virtual void onGainedKeyboardFocus_impl() {}
                NBL_API2 virtual void onLostKeyboardFocus_impl() {}

                // TODO: change the signature of the disconnected calls to be `const T* const`
                NBL_API2 virtual void onMouseConnected_impl(core::smart_refctd_ptr<IMouseEventChannel>&& mch) {}
                NBL_API2 virtual void onMouseDisconnected_impl(IMouseEventChannel* mch) {}
                NBL_API2 virtual void onKeyboardConnected_impl(core::smart_refctd_ptr<IKeyboardEventChannel>&& kbch) {}
                NBL_API2 virtual void onKeyboardDisconnected_impl(IKeyboardEventChannel* mch) {}
        };

        friend class IEventCallback;
        inline void setEventCallback(core::smart_refctd_ptr<IEventCallback>&& evCb) { m_cb = std::move(evCb); }

        inline bool isFullscreen()              { return (m_flags.value & ECF_FULLSCREEN); }
        inline bool isHidden()                  { return (m_flags.value & ECF_HIDDEN); }
        inline bool isBorderless()              { return (m_flags.value & ECF_BORDERLESS); }
        inline bool canProgrammaticallyResize() { return (m_flags.value & ECF_RESIZABLE); }
        inline bool isMinimized()               { return (m_flags.value & ECF_MINIMIZED); }
        inline bool isMaximized()               { return (m_flags.value & ECF_MAXIMIZED); }
        inline bool hasMouseCaptured()          { return (m_flags.value & ECF_MOUSE_CAPTURE); }
        inline bool hasInputFocus()             { return (m_flags.value & ECF_INPUT_FOCUS); }
        inline bool hasMouseFocus()             { return (m_flags.value & ECF_MOUSE_FOCUS); }
        inline bool isAlwaysOnTop()             { return (m_flags.value & ECF_ALWAYS_ON_TOP); }
        inline bool isMaximizable()             { return (m_flags.value & ECF_CAN_MAXIMIZE); }
        inline bool isResizable()               { return (m_flags.value & ECF_CAN_RESIZE); }

        inline core::bitflag<E_CREATE_FLAGS> getFlags() { return m_flags; }

        inline uint32_t getWidth() const { return m_width; }
        inline uint32_t getHeight() const { return m_height; }
        inline int32_t getX() const { return m_x; }
        inline int32_t getY() const { return m_y; }

        NBL_API2 virtual IClipboardManager* getClipboardManager() = 0;
        NBL_API2 virtual ICursorControl* getCursorControl() = 0;
        NBL_API2 virtual IWindowManager* getManager() const = 0;

        inline IEventCallback* getEventCallback() const { return m_cb.get(); }

        NBL_API2 virtual void setCaption(const std::string_view& caption) = 0;

        struct SCreationParams
        {
            core::smart_refctd_ptr<IEventCallback> callback;
            int32_t x, y;
            uint32_t width = 0u, height = 0u;
            core::bitflag<E_CREATE_FLAGS> flags = E_CREATE_FLAGS::ECF_NONE;
            uint32_t eventChannelCapacityLog2[IInputEventChannel::ET_COUNT];
            std::string windowCaption;
        };

    protected:
        // TODO need to update constructors of all derived CWindow* classes
        inline IWindow(SCreationParams&& params) :
            m_cb(std::move(params.callback)), m_width(params.width), m_height(params.height), m_x(params.x), m_y(params.y), m_flags(params.flags)
        {
        }
        inline virtual ~IWindow() = default;

        core::smart_refctd_ptr<IEventCallback> m_cb;
        uint32_t m_width = 0u, m_height = 0u;
        int32_t m_x, m_y; // gonna add it here until further instructions XD
        core::bitflag<E_CREATE_FLAGS> m_flags = static_cast<E_CREATE_FLAGS>(0u);
};

NBL_ENUM_ADD_BITWISE_OPERATORS(IWindow::E_CREATE_FLAGS);

}


#endif