#include "nbl/ui/CClipboardManagerWin32.h"
#include <windows.h>

namespace nbl::ui
{
	std::string CClipboardManagerWin32::getClipboardText()
	{
		OpenClipboard(nullptr);
		std::string textData = (const char*)GetClipboardData(CF_TEXT);
		CloseClipboard();
		return textData;
	}
	bool CClipboardManagerWin32::setClipboardText(const std::string_view& data)
	{
		OpenClipboard(nullptr);
		EmptyClipboard();
		const size_t data_size = data.size() + 1;
		HGLOBAL h = GlobalAlloc(GMEM_FIXED, data_size);
		strcpy((char*)GlobalLock(h), LPCSTR(data.data()));
		GlobalUnlock(h);
		bool res = SetClipboardData(CF_TEXT, h) != nullptr;
		CloseClipboard();
		return res;
	}
	//TODO
	core::smart_refctd_ptr<asset::ICPUImage> CClipboardManagerWin32::getClipboardImage()
	{
		return nullptr;
	}
	//TODO
	bool CClipboardManagerWin32::setClipboardImage(asset::ICPUImage* image, asset::ICPUImage::SImageCopy data)
	{
		OpenClipboard(nullptr);
		auto creationParams = image->getCreationParameters();
		uint32_t imageClipboardFormat = CF_DIBV5;
		BITMAPV5HEADER* header = CPUImageToClipboardImage(image);
		memcpy(GlobalLock(h), &header, sizeof(ClipboardImageData));
		GlobalUnlock(h);
		bool res = SetClipboardData(imageClipboardFormat, h) != nullptr;
		CloseClipboard();
		return res;
	}
	//TODO
	HGLOBAL* CClipboardManagerWin32::CPUImageToClipboardImage(asset::ICPUImage* image)
	{
		HGLOBAL h = GlobalAlloc(GMEM_FIXED, sizeof(BITMAPV5HEADER));

		return nullptr;
	}
}

}