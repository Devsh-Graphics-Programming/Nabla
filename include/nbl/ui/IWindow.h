#ifndef __NBL_I_WINDOW_H_INCLUDED__
#define __NBL_I_WINDOW_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"

#include <type_traits>

#include "nbl/system/ISystem.h"

#include "nbl/ui/IClipboardManager.h"
#include "nbl/ui/IInputEventChannel.h"

namespace nbl::ui
{

class ICursorControl;

class NBL_API IWindow : public core::IReferenceCounted
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

        ECF_NONE = 0
    };


   
    class NBL_API IEventCallback : public core::IReferenceCounted
    {
    public:
        [[nodiscard]] bool onWindowShown(IWindow* w) 
        {
            auto canShow = onWindowShown_impl();
            if(canShow)
            {
                w->m_flags &= (~core::bitflag(ECF_HIDDEN));
            }
            return canShow;
        }
        [[nodiscard]] bool onWindowHidden(IWindow* w)
        {
            auto canHide = onWindowHidden_impl();
            if (canHide)
            {
                w->m_flags |= ECF_HIDDEN;
            }
            return canHide;
        }
        [[nodiscard]] bool onWindowMoved(IWindow* w, int32_t x, int32_t y)
        {
            auto canMove = onWindowMoved_impl(x, y);
            if (canMove)
            {
                w->m_x = x;
                w->m_y = y;
            }
            return canMove;
        }
        [[nodiscard]] bool onWindowResized(IWindow* w, uint32_t width, uint32_t height)
        {
            auto canResize = onWindowResized_impl(width, height);
            if (canResize)
            {
                w->m_width = width;
                w->m_height = height;
            }
            return canResize;
        }
        [[nodiscard]] bool onWindowRotated(IWindow* w)
        {
            return onWindowRotated_impl();
        }
        [[nodiscard]] bool onWindowMinimized(IWindow* w)
        {
            auto canMinimize = onWindowMinimized_impl();
            if (canMinimize)
            {
                w->m_flags |= ECF_MINIMIZED;
                w->m_flags &= (~core::bitflag(ECF_MAXIMIZED));
            }
            return canMinimize;
        }
        [[nodiscard]] bool onWindowMaximized(IWindow* w)
        {
            auto canMaximize = onWindowMaximized_impl();
            if (canMaximize)
            {
                w->m_flags |= ECF_MAXIMIZED;
                w->m_flags &= (~core::bitflag(ECF_MINIMIZED));
            }
            return canMaximize;
        }
        [[nodiscard]] bool onWindowClosed(IWindow* w)
        {
            return onWindowClosed_impl();
        }
        void onGainedMouseFocus(IWindow* w)
        {
            onGainedMouseFocus_impl();
        }
        void onLostMouseFocus(IWindow* w)
        {
            onLostMouseFocus_impl();
        }
        void onGainedKeyboardFocus(IWindow* w)
        {
            onGainedKeyboardFocus_impl();
        }
        void onLostKeyboardFocus(IWindow* w)
        {
            onLostKeyboardFocus_impl();
        }

        void onMouseConnected(IWindow* w, core::smart_refctd_ptr<IMouseEventChannel>&& mch)
        {
            onMouseConnected_impl(std::move(mch));
        }
        void onMouseDisconnected(IWindow* w, IMouseEventChannel* mch)
        {
            onMouseDisconnected_impl(mch);
        }
        void onKeyboardConnected(IWindow* w, core::smart_refctd_ptr<IKeyboardEventChannel>&& kbch)
        {
            onKeyboardConnected_impl(std::move(kbch));
        }
        void onKeyboardDisconnected(IWindow* w, IKeyboardEventChannel* kbch)
        {
            onKeyboardDisconnected_impl(kbch);
        }
        
    protected:
        virtual bool onWindowShown_impl() { return true; }
        virtual bool onWindowHidden_impl() { return true; }
        virtual bool onWindowMoved_impl(int32_t x, int32_t y) { return true; }
        virtual bool onWindowResized_impl(uint32_t w, uint32_t h) { return true; }
        virtual bool onWindowRotated_impl() { return true; }
        virtual bool onWindowMinimized_impl() { return true; }
        virtual bool onWindowMaximized_impl() { return true; }
        virtual bool onWindowClosed_impl() { return true; }
        virtual void onGainedMouseFocus_impl() {}
        virtual void onLostMouseFocus_impl() {}
        virtual void onGainedKeyboardFocus_impl() {}
        virtual void onLostKeyboardFocus_impl() {}

        virtual void onMouseConnected_impl(core::smart_refctd_ptr<IMouseEventChannel>&& mch) {}
        virtual void onMouseDisconnected_impl(IMouseEventChannel* mch) {}
        virtual void onKeyboardConnected_impl(core::smart_refctd_ptr<IKeyboardEventChannel>&& kbch) {}
        virtual void onKeyboardDisconnected_impl(IKeyboardEventChannel* mch) {}
    };
    struct SCreationParams
    {
        //IWindow(core::smart_refctd_ptr<IEventCallback>&& _cb, core::smart_refctd_ptr<system::ISystem>&& _sys, uint32_t _w = 0u, uint32_t _h = 0u, E_CREATE_FLAGS _flags = static_cast<E_CREATE_FLAGS>(0)) :
        core::smart_refctd_ptr<IEventCallback> callback;
        core::smart_refctd_ptr<system::ISystem> system;
        int32_t x, y;
        uint32_t width = 0u, height = 0u;
        E_CREATE_FLAGS flags = static_cast<E_CREATE_FLAGS>(0);
        uint32_t eventChannelCapacityLog2[IInputEventChannel::ET_COUNT];
        std::string windowCaption;
    };
    friend struct IEventCallback;

    inline bool isFullscreen()      { return (m_flags.value & ECF_FULLSCREEN); }
    inline bool isHidden()          { return (m_flags.value & ECF_HIDDEN); }
    inline bool isBorderless()      { return (m_flags.value & ECF_BORDERLESS); }
    inline bool isResizable()       { return (m_flags.value & ECF_RESIZABLE); }
    inline bool isMinimized()       { return (m_flags.value & ECF_MINIMIZED); }
    inline bool isMaximized()       { return (m_flags.value & ECF_MAXIMIZED); }
    inline bool hasMouseCaptured()  { return (m_flags.value & ECF_MOUSE_CAPTURE); }
    inline bool hasInputFocus()     { return (m_flags.value & ECF_INPUT_FOCUS); }
    inline bool hasMouseFocus()     { return (m_flags.value & ECF_MOUSE_FOCUS); }
    inline bool isAlwaysOnTop()     { return (m_flags.value & ECF_ALWAYS_ON_TOP); }

    inline uint32_t getWidth() const { return m_width; }
    inline uint32_t getHeight() const { return m_height; }
    int32_t getX() const { return m_x; }
    int32_t getY() const { return m_y; }

    virtual IClipboardManager* getClipboardManager() = 0;
    virtual ICursorControl* getCursorControl() = 0;

    IEventCallback* getEventCallback() const { return m_cb.get(); }

    virtual void setCaption(const std::string_view& caption) = 0;
protected:
    // TODO need to update constructors of all derived CWindow* classes
    IWindow(SCreationParams&& params) :
        m_cb(std::move(params.callback)), m_sys(std::move(params.system)), m_width(params.width), m_height(params.height), m_x(params.x), m_y(params.y), m_flags(params.flags)
    {

    }

    virtual ~IWindow() = default;

    core::smart_refctd_ptr<IEventCallback> m_cb;
    core::smart_refctd_ptr<system::ISystem> m_sys;
    uint32_t m_width = 0u, m_height = 0u;
    int32_t m_x, m_y; // gonna add it here until further instructions XD
    core::bitflag<E_CREATE_FLAGS> m_flags = static_cast<E_CREATE_FLAGS>(0u);
public:
        void setEventCallback(core::smart_refctd_ptr<IEventCallback>&& evCb) { m_cb = std::move(evCb); }
};

}


#endif