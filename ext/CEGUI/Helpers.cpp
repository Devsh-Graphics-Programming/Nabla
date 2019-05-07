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

bool loadColorPickerExtension()
{
    static constexpr const char* signature = "initialiseCEGUICommonDialogs";
#if defined(_WIN32)
    auto module = LoadLibraryA("libCEGUICommonDialogs-0.dll");
    if (module) {
        const auto function = GetProcAddress(module, signature);
        if (function) {
            function();
            return FreeLibrary(module) != 0;
        }
    }
#elif defined(__linux__)
    typedef void (*entry)();

    // The .so file doesn't get fixated as in Windows, additional path searches
    // are needed.
    for (const auto& file : { "./", "../lib/", "/usr/lib/" }) {
        auto module = dlopen(
            (std::string(file) + "libCEGUICommonDialogs-0.so").c_str(), RTLD_LAZY);
        if (module) {
            const auto function = (entry)dlsym(module, signature);
            if (function) {
                function();
                return dlclose(module) == 0;
            }
        }
    }
#endif
    return false;
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

ImageBuffer loadImage(const char* file)
{
    return ImageBuffer(file);
}

std::string readWindowLayout(const std::string& layoutPath)
{
    std::ifstream file(layoutPath);
    std::string str((std::istreambuf_iterator<char>(file)),
                    std::istreambuf_iterator<char>());
    return str;
}

} // namespace cegui
} // namespace ext
} // namespace irr
