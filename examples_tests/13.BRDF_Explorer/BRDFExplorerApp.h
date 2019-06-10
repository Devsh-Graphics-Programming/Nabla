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
#include <string>
#include <vector>
#include <irr/video/IGPUMesh.h>
#include <SMaterial.h>
#include <ICameraSceneNode.h>

namespace CEGUI
{
class EventArgs;
}

namespace irr
{
class IrrlichtDevice;
namespace video
{
class IVideoDriver;
}
namespace asset
{
class IAssetManager;
class ICPUTexture;
class ICPUMesh;
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
        enum ETEXTURE_SLOT {
            TEXTURE_AO,
            TEXTURE_BUMP,
            TEXTURE_SLOT_1,
            TEXTURE_SLOT_2,
            TEXTURE_SLOT_3,
            TEXTURE_SLOT_4,
        };

        using TTextureSlotMap = std::map<ETEXTURE_SLOT, std::tuple<const char*, const char*, const char*>>;

    public:
        BRDFExplorerApp(IrrlichtDevice* device, irr::scene::ICameraSceneNode* _camera);
        ~BRDFExplorerApp();

        void renderGUI();
        void renderMesh();

        // Loads a given texture buffer into slot of type T.
        // T can be one of the TextureType enum types.
        // Caller is responsible for freeing the buffer afterwards.
        void loadTextureSlot(ETEXTURE_SLOT slot, irr::asset::ICPUTexture* _texture);
        void loadTextureSlot(ETEXTURE_SLOT slot, irr::video::IVirtualTexture* _texture, const std::string& _texName);

    private:
        irr::asset::ICPUTexture* loadCPUTexture(const std::string& _path);

        struct SCPUGPUMesh {
            irr::asset::ICPUMesh* cpu;
            irr::video::IGPUMesh* gpu;
        };
        SCPUGPUMesh loadMesh(const std::string& _path);
        void loadMeshAndReplaceTextures(const std::string& _path);

        static constexpr float sliderRIRange = 1.0f;
        static constexpr float sliderMetallicRange = 1.0f;
        static constexpr float sliderRoughness1Range = 1.0f;
        static constexpr float sliderRoughness2Range = 1.0f;
        static constexpr float sliderBumpHeightRange = 20.0f;
        static constexpr float defaultOpacity = 0.85f;

        void initDropdown();
        void initTooltip();

        void updateTooltip(const char* name, const char* text);
        void eventAOTextureBrowse(const ::CEGUI::EventArgs&);
        void eventAOTextureBrowse_EditBox(const ::CEGUI::EventArgs&);
        void eventBumpTextureBrowse(const ::CEGUI::EventArgs&);
        void eventBumpTextureBrowse_EditBox(const ::CEGUI::EventArgs&);
        void eventTextureBrowse(const CEGUI::EventArgs& e);
        void eventMeshBrowse(const CEGUI::EventArgs& e);

        // currently using 1st meshbuffer's textures
        constexpr static uint32_t MESHBUFFER_NUM = 0u;

        // Default title & filtering for the choose-your-file dialog
        static constexpr const char* ImageFileDialogTitle = "Select Texture";
        static constexpr const char* MeshFileDialogTitle = "Select Mesh";

        const std::vector<std::string> ImageFileDialogFilters = {
            "Everything (*.*)", "*",
            "Image (*.jpg, *.jpeg, *.png, *.bmp, *.tga, *.dds, *.gif)",
            "*.jpg *.jpeg *.png *.bmp *.tga *.dds *.gif"
        };
        const std::vector<std::string> MeshFileDialogFilters = {
            "Everything (*.*)", "*",
            "Mesh (*.ply *.stl *.baw *.x *.obj)",
            "*.ply *.stl *.baw *.x *.obj"
        };

        void showErrorMessage(const char* title, const char* message);

    private:
        scene::ICameraSceneNode* Camera = nullptr;
        video::IVideoDriver* Driver = nullptr;
        asset::IAssetManager& AssetManager;
        ext::cegui::GUIManager* GUI = nullptr;
        TTextureSlotMap TextureSlotMap;
        bool IsIsotropic = false;
        bool IsLightAnimated = false;

        irr::video::IGPUMesh* Mesh = nullptr;
        irr::video::SGPUMaterial Material;

        irr::video::IVirtualTexture* DefaultTexture = nullptr;
};

} // namespace irr

#endif // _IRR_BRDF_EXPLORER_APP_INCLUDED_
