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

#include "CEGUIOpenGLClip.h"
#include "irrlicht.h"
#include "COpenGLExtensionHandler.h"

namespace irr
{
namespace ext
{
namespace cegui
{

void setOpenGLClip()
{
    video::COpenGLExtensionHandler::extGlClipControl(GL_LOWER_LEFT, GL_NEGATIVE_ONE_TO_ONE);
}

void resetOpenGLClip()
{
    video::COpenGLExtensionHandler::extGlClipControl(GL_UPPER_LEFT, GL_ZERO_TO_ONE);
}

} // namespace cegui
} // namespace ext
} // namespace irr
