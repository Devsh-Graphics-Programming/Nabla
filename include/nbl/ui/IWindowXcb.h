#ifndef __NBL_I_WINDOW_XCB_H_INCLUDED__
#define __NBL_I_WINDOW_XCB_H_INCLUDED__

#include "nbl/core/util/bitflag.h"
#include "nbl/ui/IWindow.h"
#include "nbl/ui/XcbConnection.h"
#include <xcb/xproto.h>

#ifdef _NBL_PLATFORM_LINUX_

namespace nbl::ui
{

class NBL_API2 IWindowXcb : public IWindow
{
    protected:
        virtual ~IWindowXcb() = default;
        inline IWindowXcb(SCreationParams&& params) : IWindow(std::move(params)) {}

    public:
        using IWindow::IWindow;

        const void* getNativeHandle() const { return nullptr; }
        virtual xcb_window_t getXcbWindow() const = 0;
        // virtual xcb_window_t getXcbRootWindow() const = 0;
        virtual xcb_connection_t* getXcbConnection() const = 0;

        virtual bool setWindowSize_impl(uint32_t width, uint32_t height) = 0;
        virtual bool setWindowPosition_impl(int32_t x, int32_t y) = 0;
        virtual bool setWindowRotation_impl(bool landscape) = 0;
        virtual bool setWindowVisible_impl(bool visible) = 0;
        virtual bool setWindowMaximized_impl(bool maximized) = 0;

        static XcbConnection::MotifWmHints fetchMotifMWHints(IWindow::E_CREATE_FLAGS flags) {
            core::bitflag<XcbConnection::MotifFlags> motifFlags(XcbConnection::MWM_HINTS_NONE);
            core::bitflag<XcbConnection::MotifFunctions> motifFunctions(XcbConnection::MWM_FUNC_NONE);
            core::bitflag<XcbConnection::MotifDecorations> motifDecorations(XcbConnection::MWM_DECOR_NONE);
            motifFlags |= XcbConnection::MWM_HINTS_DECORATIONS;

            if (flags & IWindow::ECF_BORDERLESS) {
                motifDecorations |= XcbConnection::MWM_DECOR_ALL;
            } else {
                motifDecorations |= XcbConnection::MWM_DECOR_BORDER;
                motifDecorations |= XcbConnection::MWM_DECOR_RESIZEH;
                motifDecorations |= XcbConnection::MWM_DECOR_TITLE;

                // minimize button
                if(flags & IWindow::ECF_MINIMIZED) {
                    motifDecorations |= XcbConnection::MWM_DECOR_MINIMIZE;
                    motifFunctions |= XcbConnection::MWM_FUNC_MINIMIZE;
                }
                
                // maximize button
                if(flags & IWindow::ECF_MAXIMIZED) {
                    motifDecorations |= XcbConnection::MWM_DECOR_MAXIMIZE;
                    motifFunctions |= XcbConnection::MWM_FUNC_MAXIMIZE;
                }

                // close button
                motifFunctions |= XcbConnection::MWM_FUNC_CLOSE;
            }

            if(motifFunctions.value != XcbConnection::MWM_FUNC_NONE) {
                motifFlags |= XcbConnection::MWM_HINTS_FUNCTIONS;
                motifFunctions |= XcbConnection::MWM_FUNC_RESIZE;
                motifFunctions |= XcbConnection::MWM_FUNC_MOVE;
            } else {
                motifFunctions = XcbConnection::MWM_FUNC_ALL;
            }

            XcbConnection::MotifWmHints hints;
            hints.flags = motifFlags.value;
            hints.functions = motifFunctions.value;
            hints.decorations = motifDecorations.value;
            hints.input_mode = 0;
            hints.status = 0;
            return hints;

        }
		
};

}

#endif

#endif