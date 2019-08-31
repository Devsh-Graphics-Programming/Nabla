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

#ifndef _IRR_EXT_CEGUI_HELPERS_INCLUDED_
#define _IRR_EXT_CEGUI_HELPERS_INCLUDED_

#include <functional>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <CEGUI/CEGUI.h>
#include <CEGUI/String.h>
#include <CEGUI/InputEvent.h>
#include "Keycodes.h"

namespace irr
{
namespace ext
{
namespace cegui
{

class ImageBuffer {
public:
    unsigned char* buffer = nullptr;
    int w, h, c;

    ImageBuffer(const char* file);
    ~ImageBuffer();
};

// Needs replacement if there are better alternative(s).
void Replace(std::string& str, const std::string& from, const std::string& to);
std::vector<std::string> Split(const std::string& s, const char delimiter = ' ');

// Checks if a given file exists. Like mentioned Helpers.cpp, this might be replaced with IrrlichtBAW's file system API.
int Exists(const char* file);

std::pair<bool, std::string> openFileDialog(const char* title, const std::vector<std::string>& filters);

// Loads an image file to a wrapped buffer. Replace your own IrrlichtBAW loader here
//ImageBuffer loadImage(const char* file);

// Basically std::to_string(float), but with customizable floating point precision
template <typename T>
::CEGUI::String toStringFloat(const T rvalue, const int n = 6)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << rvalue;
    return ::CEGUI::String(out.str());
}

// sprintf() for std::string
template <typename... Args>
::CEGUI::String ssprintf(const std::string& format, Args... args)
{
    size_t size = snprintf(nullptr, 0, format.c_str(), args...) + 1; // Extra space for '\0'
    std::unique_ptr<char[]> buf(new char[size]);
    snprintf(buf.get(), size, format.c_str(), args...);
    return ::CEGUI::String(std::string(buf.get(), buf.get() + size - 1)); // We don't want the '\0' inside
}

std::string readWindowLayout(const std::string& layoutName);
::CEGUI::Key::Scan toCEGUIKey(const irr::EKEY_CODE& code);

} // namespace cegui
} // namespace ext
} // namespace irr

#endif // _IRR_EXT_CEGUI_HELPERS_INCLUDED_
