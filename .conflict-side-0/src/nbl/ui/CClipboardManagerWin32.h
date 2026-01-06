#ifndef _NBL_UI_C_CLIPBOARD_MANAGER_WIN32_INCLUDED_
#define _NBL_UI_C_CLIPBOARD_MANAGER_WIN32_INCLUDED_

#include "nbl/ui/IClipboardManager.h"

#ifdef _NBL_PLATFORM_WINDOWS_
#include <Windows.h>
namespace nbl::ui
{

class NBL_API2 CClipboardManagerWin32 final : public IClipboardManager
{
		using base_t = IClipboardManager;
	public:
		inline CClipboardManagerWin32() = default;

		inline std::string getClipboardText() override
		{
			int32_t res = OpenClipboard(0);
			assert(res != 0);

			std::string data = (const char*)GetClipboardData(CF_TEXT);

			CloseClipboard();

			return data;
		}
		virtual bool setClipboardText(const std::string_view& str) override
		{
			int32_t res = OpenClipboard(0);
			if (res == 0)
				return false;

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

		//virtual core::smart_refctd_ptr<asset::ICPUImage> getClipboardImage() override;
		//virtual bool setClipboardImage(asset::ICPUImage* image, asset::ICPUImage::SImageCopy data) override;
	private:
		//HGLOBAL* CPUImageToClipboardImage(asset::ICPUImage* image);
};

}


#endif
#endif