#ifndef C_XCB_CONNECTION_XCB
#define C_XCB_CONNECTION_XCB

#ifdef _NBL_PLATFORM_LINUX_

#include "nbl/core/decl/Types.h"
#include "nbl/core/string/StringLiteral.h"
#include "nbl/core/decl/smart_refctd_ptr.h"
#include "nbl/ui/CWindowManagerXCB.h"

#include <xcb/xcb.h>
#include <xcb/xcb_icccm.h>
#include <xcb/xproto.h>

#include <cstdint>

namespace nbl::ui 
{

class XCBConnection : public core::IReferenceCounted {
public:
    template<core::StringLiteral Name> 
    struct XCBAtomToken {
        xcb_atom_t token = 0;
        bool fetched = false;
    };

    XCBConnection(core::smart_refctd_ptr<CWindowManagerXCB>&& windowManager);
    virtual ~XCBConnection() override;

    template<core::StringLiteral Name> 
    inline xcb_atom_t resolveAtom(XCBAtomToken<Name>& token, bool only_if_exists = true, bool forced = false) const {
        const auto& xcb = m_windowManager->getXcbFunctionTable();
        if(token.fetched && !forced) {
            return token.token;
        }
        token.fetched = true;
        size_t size = Name.size() - 1; // -1 to remove the null terminator
        xcb_intern_atom_cookie_t cookie = xcb.pxcb_intern_atom(m_connection, only_if_exists, size, Name.value);
        if(xcb_intern_atom_reply_t* reply = xcb.pxcb_intern_atom_reply(m_connection, cookie, nullptr)) {
            token.token = reply->atom;
            free(reply);
            return token.token;
        }
        return token.token;
    }

    void setNetMWState(xcb_window_t rootWindow, xcb_window_t window, bool set, xcb_atom_t first, xcb_atom_t second = XCB_NONE) const;

    enum MotifFlags: uint32_t {
        MWM_HINTS_NONE = 0,
        MWM_HINTS_FUNCTIONS = (1L << 0),
        MWM_HINTS_DECORATIONS = (1L << 1),
        MWM_HINTS_INPUT_MODE = (1L << 2),
        MWM_HINTS_STATUS = (1L << 3),
    };

    enum MotifFunctions: uint32_t {
        MWM_FUNC_NONE = 0,
        MWM_FUNC_ALL      = (1L << 0),
        MWM_FUNC_RESIZE   = (1L << 1),
        MWM_FUNC_MOVE     = (1L << 2),
        MWM_FUNC_MINIMIZE = (1L << 3),
        MWM_FUNC_MAXIMIZE = (1L << 4),
        MWM_FUNC_CLOSE    = (1L << 5),
    };

    enum MotifDecorations: uint32_t {
        MWM_DECOR_NONE = 0,
        MWM_DECOR_ALL      = (1L << 0),
        MWM_DECOR_BORDER   = (1L << 1),
        MWM_DECOR_RESIZEH  = (1L << 2),
        MWM_DECOR_TITLE    = (1L << 3),
        MWM_DECOR_MENU     = (1L << 4),
        MWM_DECOR_MINIMIZE = (1L << 5),
        MWM_DECOR_MAXIMIZE = (1L << 6),
    };

    // insane magic in xcb for window hinting good luck finding documentation
    // https://fossies.org/linux/motif/lib/Xm/MwmUtil.h
    struct MotifWmHints {
        MotifFlags flags = MotifFlags::MWM_HINTS_NONE;
        MotifFunctions functions = MotifFunctions::MWM_FUNC_NONE;
        MotifDecorations decorations = MotifDecorations::MWM_DECOR_NONE;
        uint32_t input_mode = 0; // unused
        uint32_t status = 0; // unused
    };
    void setMotifWmHints(xcb_window_t window, const MotifWmHints& hint) const;

    inline xcb_connection_t* getRawConnection() const {
        return m_connection;
    }

    const CWindowManagerXCB::Xcb& getXcbFunctionTable() const { return m_windowManager->getXcbFunctionTable(); }
    const CWindowManagerXCB::XcbIcccm& getXcbIcccmFunctionTable() const { return m_windowManager->getXcbIcccmFunctionTable(); }

    const xcb_screen_t* primaryScreen();

private:
    core::smart_refctd_ptr<CWindowManagerXCB> m_windowManager;
    xcb_connection_t* m_connection = nullptr;

    mutable XCBConnection::XCBAtomToken<core::StringLiteral("_NET_WM_STATE")> m_NET_WM_STATE;
	mutable XCBConnection::XCBAtomToken<core::StringLiteral("_MOTIF_WM_HINTS")> m_MOTIF_WM_HINTS;
};

} // namespace nbl::ui

#endif

#endif // C_XCB_HANDLER_XCB
