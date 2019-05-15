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

#ifndef _IRR_BRDF_EXPLORER_APP_INCLUDED_
#define _IRR_BRDF_EXPLORER_APP_INCLUDED_

#include <map>

namespace irr
{
class IrrlichtDevice;
namespace video
{
class IVideoDriver;
}

namespace CEGUI
{
class EventArgs;
}

namespace ext
{
namespace cegui
{
class GUIManager;
}
}

class BRDFExplorerApp {
    public:
        enum class ETEXTURE_SLOT {
            TEXTURE_AO,
            TEXTURE_BUMP,
            TEXTURE_SLOT_1,
            TEXTURE_SLOT_2,
            TEXTURE_SLOT_3,
            TEXTURE_SLOT_4,
        };

        using TTextureSlotMap = std::map<ETEXTURE_SLOT, std::tuple<const char*, const char*, const char*>>;

    public:
        BRDFExplorerApp(IrrlichtDevice* device);
        ~BRDFExplorerApp();

        void renderGUI();

        // Loads a given texture buffer into slot of type T.
        // T can be one of the TextureType enum types.
        // Caller is responsible for freeing the buffer afterwards.
        void loadTextureSlot(ETEXTURE_SLOT slot, const unsigned char* buffer, unsigned w, unsigned h);

    private:
        static constexpr float sliderRIRange = 1.0f;
        static constexpr float sliderMetallicRange = 1.0f;
        static constexpr float sliderRoughness1Range = 1.0f;
        static constexpr float sliderRoughness2Range = 1.0f;
        static constexpr float sliderBumpHeightRange = 20.0f;

        void AOTextureBrowseEvent(const CEGUI::EventArgs&);
        void AOTextureBrowseEvent_EditBox(const CEGUI::EventArgs&);
        void BumpTextureBrowseEvent(const CEGUI::EventArgs&);
        void BumpTextureBrowseEvent_EditBox(const CEGUI::EventArgs&);

    private:
        video::IVideoDriver* Driver = nullptr;
        ext::cegui::GUIManager* GUI = nullptr;
        TTextureSlotMap TextureSlotMap;
        bool IsIsotropic = false;
};

} // namespace irr

#endif // _IRR_BRDF_EXPLORER_APP_INCLUDED_
