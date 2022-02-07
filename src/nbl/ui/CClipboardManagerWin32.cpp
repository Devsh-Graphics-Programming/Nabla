#include "nbl/ui/CClipboardManagerWin32.h"
#include <Windows.h>
namespace nbl::ui
{

    std::string CClipboardManagerWin32::getClipboardText()
    {
        int32_t res = OpenClipboard(0);
        assert(res != 0);

        std::string data = (const char*)GetClipboardData(CF_TEXT);

        CloseClipboard();

        return data;
    }

    bool CClipboardManagerWin32::setClipboardText(const std::string_view& str)
    {
        int32_t res = OpenClipboard(0);
        if (res == 0) return false;

        EmptyClipboard();
        const char* data = str.data();
        const size_t data_size = strlen(data) + 1;
        HGLOBAL h = GlobalAlloc(GMEM_MOVEABLE, data_size);
        strcpy((char*)GlobalLock(h), LPCSTR(data));
        GlobalUnlock(h);
        SetClipboardData(CF_TEXT, h);

        CloseClipboard();

        return true;
    }
}