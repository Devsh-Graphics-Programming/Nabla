/*

MIT License

Copyright (c) 2019 InnerPiece Technology Co., Ltd.
https://innerpiece.io

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#define STB_IMAGE_IMPLEMENTATION
#define STBI_MINGW_ENABLE_SSE2

#include "stb/stb_image.h"
#include "portable-file-dialogs/portable-file-dialogs.h"
#include "Helpers.h"

#include <string>
#include <fstream>
#include <streambuf>

namespace irr
{
namespace ext
{
namespace cegui
{

#if defined(_WIN32)
    #include <windows.h> // LoadLibraryA(), GetProcAddress(), FreeLibrary()
#elif defined(__linux__)
    #include <sys/stat.h> // stat()
    #include <sys/types.h>
    #include <unistd.h>
    #include <dlfcn.h> // dlopen(), dlsym(), dlclose()
#endif

std::pair<bool, std::string> openFileDialog(
    const char* title,
    const std::vector<std::string>& filters)
{
    auto buffer = pfd::open_file(title, ".", filters).result();
    if (!buffer.empty())
        return std::make_pair(true, buffer[0]);
    return std::make_pair(false, std::string());
}

std::vector<std::string> Split(const std::string& s, const char delimiter)
{
    std::vector<std::string> v;
    std::istringstream f(s);
    std::string r;
    while (std::getline(f, r, delimiter))
        v.push_back(r);

    return v;
}

ImageBuffer::ImageBuffer(const char* file)
{
    buffer = stbi_load(file, &w, &h, &c, STBI_rgb_alpha);
}

ImageBuffer::~ImageBuffer()
{
    if (buffer)
        stbi_image_free(buffer);
}

// Might be replaced with IrrlichtBAW's file system API.
int Exists(const char* file)
{
#if defined(_WIN32)
    DWORD attribute = GetFileAttributes(file);
    return (attribute != INVALID_FILE_ATTRIBUTES && !(attribute & FILE_ATTRIBUTE_DIRECTORY));
#elif defined(__linux__)
    struct stat s;
    return stat(file, &s) == 0;
#endif
}

void Replace(std::string& str, const std::string& from, const std::string& to)
{
    size_t start = 0;
    while ((start = str.find(from, start)) != std::string::npos) {
        str.replace(start, from.length(), to);
        start += to.length(); // Handles case where 'to' is a substring of 'from'
    }
}

//ImageBuffer loadImage(const char* file)
//{
//    return ImageBuffer(file);
//}

std::string readWindowLayout(const std::string& layoutPath)
{
    std::ifstream file(layoutPath);
    std::string str((std::istreambuf_iterator<char>(file)),
                    std::istreambuf_iterator<char>());
    return str;
}

::CEGUI::Key::Scan toCEGUIKey(const irr::EKEY_CODE& code)
{
    switch (code) {
    case irr::EKEY_CODE::KEY_ESCAPE: return ::CEGUI::Key::Scan::Escape;
    case irr::EKEY_CODE::KEY_KEY_1: return ::CEGUI::Key::Scan::One;
    case irr::EKEY_CODE::KEY_KEY_2: return ::CEGUI::Key::Scan::Two;
    case irr::EKEY_CODE::KEY_KEY_3: return ::CEGUI::Key::Scan::Three;
    case irr::EKEY_CODE::KEY_KEY_4: return ::CEGUI::Key::Scan::Four;
    case irr::EKEY_CODE::KEY_KEY_5: return ::CEGUI::Key::Scan::Five;
    case irr::EKEY_CODE::KEY_KEY_6: return ::CEGUI::Key::Scan::Six;
    case irr::EKEY_CODE::KEY_KEY_7: return ::CEGUI::Key::Scan::Seven;
    case irr::EKEY_CODE::KEY_KEY_8: return ::CEGUI::Key::Scan::Eight;
    case irr::EKEY_CODE::KEY_KEY_9: return ::CEGUI::Key::Scan::Nine;
    case irr::EKEY_CODE::KEY_KEY_0: return ::CEGUI::Key::Scan::Zero;
    case irr::EKEY_CODE::KEY_MINUS: return ::CEGUI::Key::Scan::Minus;
    // case irr::EKEY_CODE::KEY_ : return ::CEGUI::Key::Scan::Equals;
    case irr::EKEY_CODE::KEY_BACK: return ::CEGUI::Key::Scan::Backspace;
    case irr::EKEY_CODE::KEY_TAB: return ::CEGUI::Key::Scan::Tab;
    case irr::EKEY_CODE::KEY_KEY_Q: return ::CEGUI::Key::Scan::Q;
    case irr::EKEY_CODE::KEY_KEY_W: return ::CEGUI::Key::Scan::W;
    case irr::EKEY_CODE::KEY_KEY_E: return ::CEGUI::Key::Scan::E;
    case irr::EKEY_CODE::KEY_KEY_R: return ::CEGUI::Key::Scan::R;
    case irr::EKEY_CODE::KEY_KEY_T: return ::CEGUI::Key::Scan::T;
    case irr::EKEY_CODE::KEY_KEY_I: return ::CEGUI::Key::Scan::I;
    case irr::EKEY_CODE::KEY_KEY_O: return ::CEGUI::Key::Scan::O;
    case irr::EKEY_CODE::KEY_KEY_P: return ::CEGUI::Key::Scan::P;
    case irr::EKEY_CODE::KEY_OEM_4: return ::CEGUI::Key::Scan::LeftBracket;
    case irr::EKEY_CODE::KEY_OEM_6: return ::CEGUI::Key::Scan::RightBracket;
    case irr::EKEY_CODE::KEY_RETURN: return ::CEGUI::Key::Scan::Return;
    case irr::EKEY_CODE::KEY_LCONTROL: return ::CEGUI::Key::Scan::LeftControl;
    case irr::EKEY_CODE::KEY_KEY_A: return ::CEGUI::Key::Scan::A;
    case irr::EKEY_CODE::KEY_KEY_S: return ::CEGUI::Key::Scan::S;
    case irr::EKEY_CODE::KEY_KEY_D: return ::CEGUI::Key::Scan::D;
    case irr::EKEY_CODE::KEY_KEY_F: return ::CEGUI::Key::Scan::F;
    case irr::EKEY_CODE::KEY_KEY_G: return ::CEGUI::Key::Scan::G;
    case irr::EKEY_CODE::KEY_KEY_H: return ::CEGUI::Key::Scan::H;
    case irr::EKEY_CODE::KEY_KEY_J: return ::CEGUI::Key::Scan::J;
    case irr::EKEY_CODE::KEY_KEY_K: return ::CEGUI::Key::Scan::K;
    case irr::EKEY_CODE::KEY_KEY_L: return ::CEGUI::Key::Scan::L;
    case irr::EKEY_CODE::KEY_OEM_1: return ::CEGUI::Key::Scan::Semicolon;
    case irr::EKEY_CODE::KEY_OEM_7: return ::CEGUI::Key::Scan::Apostrophe;
    case irr::EKEY_CODE::KEY_OEM_3: return ::CEGUI::Key::Scan::Grave;
    case irr::EKEY_CODE::KEY_LSHIFT: return ::CEGUI::Key::Scan::LeftShift;
    case irr::EKEY_CODE::KEY_OEM_5: return ::CEGUI::Key::Scan::Backslash;
    case irr::EKEY_CODE::KEY_KEY_Z: return ::CEGUI::Key::Scan::Z;
    case irr::EKEY_CODE::KEY_KEY_X: return ::CEGUI::Key::Scan::X;
    case irr::EKEY_CODE::KEY_KEY_C: return ::CEGUI::Key::Scan::C;
    case irr::EKEY_CODE::KEY_KEY_V: return ::CEGUI::Key::Scan::V;
    case irr::EKEY_CODE::KEY_KEY_B: return ::CEGUI::Key::Scan::B;
    case irr::EKEY_CODE::KEY_KEY_N: return ::CEGUI::Key::Scan::N;
    case irr::EKEY_CODE::KEY_KEY_M: return ::CEGUI::Key::Scan::M;
    case irr::EKEY_CODE::KEY_COMMA: return ::CEGUI::Key::Scan::Comma;
    case irr::EKEY_CODE::KEY_PERIOD: return ::CEGUI::Key::Scan::Period;
    case irr::EKEY_CODE::KEY_OEM_2: return ::CEGUI::Key::Scan::Slash;
    case irr::EKEY_CODE::KEY_RSHIFT: return ::CEGUI::Key::Scan::RightShift;
    case irr::EKEY_CODE::KEY_MULTIPLY: return ::CEGUI::Key::Scan::Multiply;
    case irr::EKEY_CODE::KEY_MENU: return ::CEGUI::Key::Scan::LeftAlt;
    case irr::EKEY_CODE::KEY_SPACE: return ::CEGUI::Key::Scan::Space;
    case irr::EKEY_CODE::KEY_CAPITAL: return ::CEGUI::Key::Scan::Capital;
    case irr::EKEY_CODE::KEY_F1: return ::CEGUI::Key::Scan::F1;
    case irr::EKEY_CODE::KEY_F2: return ::CEGUI::Key::Scan::F2;
    case irr::EKEY_CODE::KEY_F3: return ::CEGUI::Key::Scan::F3;
    case irr::EKEY_CODE::KEY_F4: return ::CEGUI::Key::Scan::F4;
    case irr::EKEY_CODE::KEY_F5: return ::CEGUI::Key::Scan::F5;
    case irr::EKEY_CODE::KEY_F6: return ::CEGUI::Key::Scan::F6;
    case irr::EKEY_CODE::KEY_F7: return ::CEGUI::Key::Scan::F7;
    case irr::EKEY_CODE::KEY_F8: return ::CEGUI::Key::Scan::F8;
    case irr::EKEY_CODE::KEY_F9: return ::CEGUI::Key::Scan::F9;
    case irr::EKEY_CODE::KEY_F10: return ::CEGUI::Key::Scan::F10;
    case irr::EKEY_CODE::KEY_NUMLOCK: return ::CEGUI::Key::Scan::NumLock;
    case irr::EKEY_CODE::KEY_SCROLL: return ::CEGUI::Key::Scan::ScrollLock;
    case irr::EKEY_CODE::KEY_NUMPAD7: return ::CEGUI::Key::Scan::Numpad7;
    case irr::EKEY_CODE::KEY_NUMPAD8: return ::CEGUI::Key::Scan::Numpad8;
    case irr::EKEY_CODE::KEY_NUMPAD9: return ::CEGUI::Key::Scan::Numpad9;
    case irr::EKEY_CODE::KEY_SUBTRACT: return ::CEGUI::Key::Scan::Subtract;
    case irr::EKEY_CODE::KEY_NUMPAD4: return ::CEGUI::Key::Scan::Numpad4;
    case irr::EKEY_CODE::KEY_NUMPAD5: return ::CEGUI::Key::Scan::Numpad5;
    case irr::EKEY_CODE::KEY_NUMPAD6: return ::CEGUI::Key::Scan::Numpad6;
    case irr::EKEY_CODE::KEY_ADD: return ::CEGUI::Key::Scan::Add;
    case irr::EKEY_CODE::KEY_NUMPAD1: return ::CEGUI::Key::Scan::Numpad1;
    case irr::EKEY_CODE::KEY_NUMPAD2: return ::CEGUI::Key::Scan::Numpad2;
    case irr::EKEY_CODE::KEY_NUMPAD3: return ::CEGUI::Key::Scan::Numpad3;
    case irr::EKEY_CODE::KEY_NUMPAD0: return ::CEGUI::Key::Scan::Numpad0;
    case irr::EKEY_CODE::KEY_DECIMAL: return ::CEGUI::Key::Scan::Decimal;
    case irr::EKEY_CODE::KEY_OEM_102: return ::CEGUI::Key::Scan::OEM_102;
    case irr::EKEY_CODE::KEY_F11: return ::CEGUI::Key::Scan::F11;
    case irr::EKEY_CODE::KEY_F12: return ::CEGUI::Key::Scan::F12;
    case irr::EKEY_CODE::KEY_F13: return ::CEGUI::Key::Scan::F13;
    case irr::EKEY_CODE::KEY_F14: return ::CEGUI::Key::Scan::F14;
    case irr::EKEY_CODE::KEY_F15: return ::CEGUI::Key::Scan::F15;
    case irr::EKEY_CODE::KEY_KANA: return ::CEGUI::Key::Scan::Kana;
    // case irr::EKEY_CODE::KEY_ : return ::CEGUI::Key::Scan::ABNT_C1;
    case irr::EKEY_CODE::KEY_CONVERT: return ::CEGUI::Key::Scan::Convert;
    case irr::EKEY_CODE::KEY_NONCONVERT: return ::CEGUI::Key::Scan::NoConvert;
    // case irr::EKEY_CODE::KEY_ : return ::CEGUI::Key::Scan::Yen;
    // case irr::EKEY_CODE::KEY_ : return ::CEGUI::Key::Scan::ABNT_C2;
    // case irr::EKEY_CODE::KEY_ : return ::CEGUI::Key::Scan::NumpadEquals;
    // case irr::EKEY_CODE::KEY_ : return ::CEGUI::Key::Scan::PrevTrack;
    // case irr::EKEY_CODE::KEY_ : return ::CEGUI::Key::Scan::At;
    // case irr::EKEY_CODE::KEY_ : return ::CEGUI::Key::Scan::Colon;
    // case irr::EKEY_CODE::KEY_ : return ::CEGUI::Key::Scan::Underline;
    case irr::EKEY_CODE::KEY_KANJI: return ::CEGUI::Key::Scan::Kanji;
    // case irr::EKEY_CODE::KEY_ : return ::CEGUI::Key::Scan::Stop;
    case irr::EKEY_CODE::KEY_OEM_AX : return ::CEGUI::Key::Scan::AX;
    // case irr::EKEY_CODE::KEY_ : return ::CEGUI::Key::Scan::Unlabeled;
    // case irr::EKEY_CODE::KEY_ : return ::CEGUI::Key::Scan::NextTrack;
    case irr::EKEY_CODE::KEY_RCONTROL: return ::CEGUI::Key::Scan::RightControl;
    // case irr::EKEY_CODE::KEY_ : return ::CEGUI::Key::Scan::Mute;
    // case irr::EKEY_CODE::KEY_ : return ::CEGUI::Key::Scan::Calculator;
    case irr::EKEY_CODE::KEY_PLAY: return ::CEGUI::Key::Scan::PlayPause;
    // case irr::EKEY_CODE::KEY_ : return ::CEGUI::Key::Scan::MediaStop;
    // case irr::EKEY_CODE::KEY_ : return ::CEGUI::Key::Scan::VolumeDown;
    // case irr::EKEY_CODE::KEY_ : return ::CEGUI::Key::Scan::VolumeUp;
    // case irr::EKEY_CODE::KEY_ : return ::CEGUI::Key::Scan::WebHome;
    case irr::EKEY_CODE::KEY_DIVIDE: return ::CEGUI::Key::Scan::Divide;
    // case irr::EKEY_CODE::KEY_ : return ::CEGUI::Key::Scan::SysRq;
    case irr::EKEY_CODE::KEY_HOME: return ::CEGUI::Key::Scan::Home;
    case irr::EKEY_CODE::KEY_UP: return ::CEGUI::Key::Scan::ArrowUp;
    case irr::EKEY_CODE::KEY_PRIOR: return ::CEGUI::Key::Scan::PageUp;
    case irr::EKEY_CODE::KEY_LEFT: return ::CEGUI::Key::Scan::ArrowLeft;
    case irr::EKEY_CODE::KEY_RIGHT: return ::CEGUI::Key::Scan::ArrowRight;
    case irr::EKEY_CODE::KEY_END : return ::CEGUI::Key::Scan::End;
    case irr::EKEY_CODE::KEY_DOWN: return ::CEGUI::Key::Scan::ArrowDown;
    case irr::EKEY_CODE::KEY_NEXT: return ::CEGUI::Key::Scan::PageDown;
    case irr::EKEY_CODE::KEY_INSERT: return ::CEGUI::Key::Scan::Insert;
    case irr::EKEY_CODE::KEY_DELETE: return ::CEGUI::Key::Scan::Delete;
    case irr::EKEY_CODE::KEY_LWIN: return ::CEGUI::Key::Scan::LeftWindows;
    case irr::EKEY_CODE::KEY_RWIN: return ::CEGUI::Key::Scan::RightWindows;
    case irr::EKEY_CODE::KEY_APPS: return ::CEGUI::Key::Scan::AppMenu;
    // case irr::EKEY_CODE::KEY_ : return ::CEGUI::Key::Scan::Power;
    // case irr::EKEY_CODE::KEY_ : return ::CEGUI::Key::Scan::Sleep;
    // case irr::EKEY_CODE::KEY_ : return ::CEGUI::Key::Scan::Wake;
    // case irr::EKEY_CODE::KEY_ : return ::CEGUI::Key::Scan::WebSearch;
    // case irr::EKEY_CODE::KEY_ : return ::CEGUI::Key::Scan::WebFavorites;
    // case irr::EKEY_CODE::KEY_ : return ::CEGUI::Key::Scan::WebRefresh;
    // case irr::EKEY_CODE::KEY_ : return ::CEGUI::Key::Scan::WebStop;
    // case irr::EKEY_CODE::KEY_ : return ::CEGUI::Key::Scan::WebForward;
    // case irr::EKEY_CODE::KEY_ : return ::CEGUI::Key::Scan::WebBack;
    // case irr::EKEY_CODE::KEY_ : return ::CEGUI::Key::Scan::MyComputer;
    // case irr::EKEY_CODE::KEY_ : return ::CEGUI::Key::Scan::Mail;
    // case irr::EKEY_CODE::KEY_ : return ::CEGUI::Key::Scan::MediaSelect;
    default:
      return ::CEGUI::Key::Scan::Unknown;
  }
}

} // namespace cegui
} // namespace ext
} // namespace irr
