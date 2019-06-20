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

#include "BRDFExplorerApp.h"
#include "../../ext/CEGUI/ExtCEGUI.h"
#include <CEGUI/RendererModules/OpenGL/Texture.h>
#include <IShaderConstantSetCallBack.h>

#include "workaroundFunctions.h"

using namespace irr;

namespace
{
class CShaderConstantSetCallback : public video::IShaderConstantSetCallBack
{
    struct SShaderConstant {
        int32_t location;
        video::E_SHADER_CONSTANT_TYPE type;
    };

    const irr::BRDFExplorerApp::SGUIState& GUIState;
    scene::ICameraSceneNode* Camera;

    static constexpr SShaderConstant uVP {20, video::ESCT_FLOAT_MAT4};
    static constexpr SShaderConstant uEmissive {0, video::ESCT_FLOAT_VEC3};
    static constexpr SShaderConstant uAlbedo {1, video::ESCT_FLOAT_VEC3};
    static constexpr SShaderConstant uRoughness1 {2, video::ESCT_FLOAT};
    static constexpr SShaderConstant uRoughness2 {3, video::ESCT_FLOAT};
    static constexpr SShaderConstant uIoR {4, video::ESCT_FLOAT};
    static constexpr SShaderConstant uMetallic {5, video::ESCT_FLOAT};
    static constexpr SShaderConstant uHeightFactor {6, video::ESCT_FLOAT};
    static constexpr SShaderConstant uLightColor {7, video::ESCT_FLOAT_VEC3};
    static constexpr SShaderConstant uLightPos {8, video::ESCT_FLOAT_VEC3};
    static constexpr SShaderConstant uEyePos {9, video::ESCT_FLOAT_VEC3};

public:
    CShaderConstantSetCallback(scene::ICameraSceneNode* _camera, const irr::BRDFExplorerApp::SGUIState& _guiState) : Camera{ _camera }, GUIState{_guiState} {}

    virtual void PostLink(video::IMaterialRendererServices* services, const video::E_MATERIAL_TYPE& materialType, const core::vector<video::SConstantLocationNamePair>& constants)
    {
    }

    virtual void OnSetConstants(video::IMaterialRendererServices* services, int32_t)
    {
        // vertex shader
        auto vp = Camera->getConcatenatedMatrix();
        services->setShaderConstant(vp.pointer(), uVP.location, uVP.type, 1u);

        // fragment shader
        services->setShaderConstant(&GUIState.Emissive.Color.X, uEmissive.location, uEmissive.type, 1u);
        if (GUIState.Albedo.SourceDropdown == irr::BRDFExplorerApp::EDS_CONSTANT)
            services->setShaderConstant(&GUIState.Albedo.ConstantColor.X, uAlbedo.location, uAlbedo.type, 1u);
        if (GUIState.Roughness.SourceDropdown == irr::BRDFExplorerApp::EDS_CONSTANT)
            services->setShaderConstant(&GUIState.Roughness.ConstValue1, uRoughness1.location, uRoughness1.type, 1u);
        if (GUIState.RefractionIndex.SourceDropdown == irr::BRDFExplorerApp::EDS_CONSTANT)
            services->setShaderConstant(&GUIState.RefractionIndex.ConstValue, uIoR.location, uIoR.type, 1u);
        if (GUIState.Metallic.SourceDropdown == irr::BRDFExplorerApp::EDS_CONSTANT)
            services->setShaderConstant(&GUIState.Metallic.ConstValue, uMetallic.location, uMetallic.type, 1u);
        //services->setShaderConstant(&GUIState.BumpMapping.Height, uHeightFactor.location, uHeightFactor.type, 1u);
        if (!GUIState.Light.Animated)
            services->setShaderConstant(&GUIState.Light.ConstantPosition, uLightPos.location, uLightPos.type, 1u);
        services->setShaderConstant(&GUIState.Light.Color, uLightColor.location, uLightColor.type, 1u);

        auto eyePos = Camera->getPosition();
        services->setShaderConstant(&eyePos.X, uEyePos.location, uEyePos.type, 1u);
    }

    virtual void OnUnsetMaterial() {}
};
}

class CShaderManager
{
public:
    struct SParams
    {
        bool constantAlbedo;
        bool isotropicRoughness;
        bool constantRoughness;
        bool roughnessIsZero;
        bool constantRI;
        bool constantMetallic;
        bool metallicIsZero;
        bool metallicIsOne;
        bool AOEnabled;
    };

private:
    using Key_t = uint16_t;

    core::unordered_map<Key_t, video::E_MATERIAL_TYPE> Shaders;
    video::IGPUProgrammingServices* Services = nullptr;
    asset::IIncludeHandler* IncludeHandler = nullptr;
    const irr::BRDFExplorerApp::SGUIState& GUIState;
    scene::ICameraSceneNode* Camera = nullptr;

    enum E_SHADER_FLAGS : Key_t
    {
        ESF_CONST_ALBEDO = 1<<0,
        ESF_ISOTROPIC_ROUGHNESS = 1<<1,
        ESF_CONST_ROUGHNESS = 1<<2,
        ESF_ZERO_ROUGHNESS = 1<<3,
        ESF_CONST_RI = 1<<4,
        ESF_CONST_METALLIC = 1<<5,
        ESF_ZERO_METALLIC = 1<<6,
        ESF_ONE_METALLIC = 1<<7,
        ESF_AO_ENABLED = 1<<8
    };

    static constexpr const char* VERTEX_SHADER_SRC = 
R"(#version 430 core

layout (location = 0) in vec3 vPosition;
layout (location = 2) in vec2 vTexCoords;
layout (location = 3) in vec3 vNormal;

out vec3 WorldPos;
out vec2 TexCoords;
out vec3 Normal;

layout (location = 20) uniform mat4 uVPMat;

void main()
{
    vec3 world = vPosition.xyz;
    WorldPos = world;
    TexCoords = vTexCoords;
    Normal = vNormal;
    gl_Position = uVPMat * vec4(world, 1.0);
}
)";

    static constexpr uint32_t firstSetBit(Key_t _x)
    {
        uint32_t n{};
        while (!(_x & Key_t{1u})) {
            ++n;
            _x >>= 1;
        }
        return n;
    }
    Key_t flagsToKey(const SParams& _p) {
        assert(!(_p.metallicIsZero && _p.metallicIsOne)); //this would be weird

        Key_t key{};
        key |= Key_t{_p.constantAlbedo}<<firstSetBit(ESF_CONST_ALBEDO);
        key |= Key_t{_p.isotropicRoughness}<<firstSetBit(ESF_ISOTROPIC_ROUGHNESS);
        key |= Key_t{_p.constantRoughness}<<firstSetBit(ESF_CONST_ROUGHNESS);
        key |= Key_t{_p.roughnessIsZero}<<firstSetBit(ESF_ZERO_ROUGHNESS);
        key |= Key_t{_p.constantRI}<<firstSetBit(ESF_CONST_RI);
        key |= Key_t{_p.constantMetallic}<<firstSetBit(ESF_CONST_METALLIC);
        key |= Key_t{_p.metallicIsZero}<<firstSetBit(ESF_ZERO_METALLIC);
        key |= Key_t{_p.metallicIsOne}<<firstSetBit(ESF_ONE_METALLIC);
        key |= Key_t{_p.AOEnabled}<<firstSetBit(ESF_AO_ENABLED);

        return key;
    }

    std::string genGetters(const SParams& _params)
    {
        std::string source = "float getRoughness(in vec2 texCoords) {\n";
        if (_params.constantRoughness)
            source += "\treturn uRoughness1;";
        else
            source += "\treturn texture(uRoughnessMap, texCoords).x;";
        source += "\n}\n";

        source += "float getMetallic(in vec2 texCoords) {\n";
        if (_params.constantMetallic)
        {
            if (_params.metallicIsZero)
                source += "\treturn 0.0;";
            else if (_params.metallicIsOne)
                source += "\treturn 1.0;";
            else
                source += "\treturn uMetallic;";
        }
        else
            source += "\treturn texture(uMetallicMap, texCoords).x;";
        source += "\n}\n";

        source += "float getIoR(in vec2 texCoords) {\n";
        if (_params.constantRI)
            source += "\treturn uIoR;";
        else
            source += "\treturn texture(uIoRMap, texCoords).x;";
        source += "\n}\n";

        source += "float getAO(in vec2 texCoords) {\n";
        if (_params.AOEnabled)
            source += "\treturn texture(uAOMap, texCoords).x;";
        else
            source += "\treturn 1.0;";
        source += "\n}\n";

        source += "vec3 getAlbedo(in vec2 texCoords) {\n";
        if (_params.constantAlbedo)
            source += "\treturn uAlbedo;";
        else
            source += "\treturn texture(uAlbedoMap, texCoords).rgb;";
        source += "\n}\n";

        return source;
    }

    video::E_MATERIAL_TYPE addShader(const SParams& _params)
    {
        std::string source =
R"(#version 430 core

layout (location = 0) out vec4 OutColor;

in vec3 WorldPos;
in vec2 TexCoords;
in vec3 Normal;

layout (location = 0) uniform vec3 uEmissive;
layout (location = 1) uniform vec3 uAlbedo;
layout (location = 2) uniform float uRoughness1;
layout (location = 3) uniform float uRoughness2;
layout (location = 4) uniform float uIoR;
layout (location = 5) uniform float uMetallic;
layout (location = 6) uniform float uHeightFactor;
layout (location = 7) uniform vec3 uLightColor;
layout (location = 8) uniform vec3 uLightPos;
layout (location = 9) uniform vec3 uEyePos;
layout (binding = 0) uniform sampler2D uAlbedoMap;
layout (binding = 1) uniform sampler2D uRoughnessMap;
layout (binding = 2) uniform sampler2D uIoRMap;
layout (binding = 3) uniform sampler2D uMetallicMap;
layout (binding = 4) uniform sampler2D uBumpMap;
layout (binding = 5) uniform sampler2D uAOMap;

float getRoughness(in vec2 texCoords);
float getMetallic(in vec2 texCoords);
float getIoR(in vec2 texCoords);
float getAO(in vec2 texCoords);
vec3 getAlbedo(in vec2 texCoords);

#define FLT_MIN 1.175494351e-38
#define FLT_MAX 3.402823466e+38
#define FLT_INF (1.0/0.0)
)"
+
IncludeHandler->getIncludeStandard("irr/builtin/glsl/brdf/diffuse/oren_nayar.glsl")
+
IncludeHandler->getIncludeStandard("irr/builtin/glsl/brdf/specular/ndf/ggx_trowbridge_reitz.glsl")
+
IncludeHandler->getIncludeStandard("irr/builtin/glsl/brdf/specular/geom/ggx_smith.glsl")
+
IncludeHandler->getIncludeStandard("irr/builtin/glsl/brdf/specular/fresnel/fresnel_schlick.glsl")
+
R"(
float diffuse(in float a2, in vec3 N, in vec3 L, in vec3 V, in float NdotL, in float NdotV);
vec3 specular(in float a2, in float NdotL, in float NdotV, in float NdotH, in float VdotH, in vec3 F0, out vec3 out_fresnel);

void main() {
    const vec3 N = normalize(Normal);
    const vec3 relLightPos = uLightPos - WorldPos;
    float NdotL = dot(N, relLightPos);

	vec3 color = vec3(0.0);
	if (NdotL>FLT_MIN)
	{
		const float relLightPosLen2 = dot(relLightPos, relLightPos);
		NdotL *= inversesqrt(relLightPosLen2);

		// there are better identities to get all of these
		const vec3 V = normalize(uEyePos - WorldPos);
		const vec3 L = normalize(relLightPos);
		const vec3 H = normalize(L + V);

		const float NdotH = max(dot(N, H), 0.0);
		const float NdotV = max(dot(N, V), 0.0);
		const float VdotH = max(dot(V, H), 0.0);
		// identity comment end (but also do you need to clamp all of them?)

		const vec2 texCoords = vec2(TexCoords.x, 1.0-TexCoords.y);

		const float a2 = getRoughness(texCoords);
		const float metallic = getMetallic(texCoords);
		const vec3 albedo = getAlbedo(texCoords);
		const float ior = getIoR(texCoords);
		const float ao = getAO(texCoords);

		float tmp = (1.0-ior)/(1.0+ior);
		const vec3 F0 = mix(vec3(tmp*tmp), albedo, metallic);
		vec3 fresnel;

		float diffuse = diffuse(a2, N, L, V, NdotL, NdotV) * (1.0 - metallic);
		vec3 spec = specular(a2, NdotL, NdotV, NdotH, VdotH, F0, fresnel);

		color += ((diffuse * albedo * (vec3(1.0) - fresnel)) + spec) * NdotL * uLightColor / relLightPosLen2;
	}
	OutColor = vec4(color, 1.0);
}
)";
        source += genGetters(_params);

        source += "float diffuse(in float a2, in vec3 N, in vec3 L, in vec3 V, in float NdotL, in float NdotV) {\n";
        if (_params.constantMetallic && !_params.metallicIsOne)
        {
            if (_params.constantRoughness && _params.roughnessIsZero)
                source += "\treturn NdotL;";
            else
                source += "\treturn oren_nayar(a2, N, L, V, NdotL, NdotV);";
        }
        else source += "\treturn 0.0;";
        source += "\n}\n";
        source += 
R"(vec3 specular(in float a2, in float NdotL, in float NdotV, in float NdotH, in float VdotH, in vec3 F0, out vec3 out_fresnel) {
	//assert(NdotL>FLT_MIN);
    out_fresnel = FresnelSchlick(F0, VdotH);
    if (NdotV<FLT_MIN)
        return vec3(0.0);
    if (a2<FLT_MIN)
		return vec3(/*NdotH>=(1.0-FLT_MIN) ? FLT_INF:*/0.0);

    float ndf = GGXTrowbridgeReitz(a2, NdotH);
    float geom = GGXSmith(a2, NdotL, NdotV); // TODO: Correlated Smith!

    return ndf*geom*out_fresnel / (4.0 * NdotV * NdotL); // TODO: Cancel denominator with smith numerator
}
)";
        //auto f = fopen("fragsrc.txt", "w");
        //fprintf(f, "%s", source.c_str());
        //fclose(f);

        // separate CB for each shader because shader-constants' values are most likely cached, so a single CB cannot be used for multiple shaders
        video::IShaderConstantSetCallBack* cb = new CShaderConstantSetCallback(Camera, GUIState);
        video::E_MATERIAL_TYPE shader = static_cast<video::E_MATERIAL_TYPE>(Services->addHighLevelShaderMaterial(VERTEX_SHADER_SRC, nullptr, nullptr, nullptr, source.c_str(), 3u, video::EMT_SOLID, cb));
        cb->drop();
        return Shaders.insert({flagsToKey(_params), shader}).first->second;
    }

public:
    CShaderManager(
        video::IGPUProgrammingServices* _services,
        asset::IIncludeHandler* _inclHandler,
        const irr::BRDFExplorerApp::SGUIState& _guiState,
        scene::ICameraSceneNode* _camera) :
        Services{_services},
        IncludeHandler{_inclHandler},
        GUIState{_guiState},
        Camera{_camera}
    {
    }

    video::E_MATERIAL_TYPE getShader(const SParams& _params)
    {
        decltype(Shaders)::const_iterator found;
        if ((found = Shaders.find(flagsToKey(_params))) != Shaders.cend())
            return found->second;

        return addShader(_params);
    }
};

namespace irr
{

BRDFExplorerApp::BRDFExplorerApp(IrrlichtDevice* device, irr::scene::ICameraSceneNode* _camera)
    :   Camera(_camera),
        Driver(device->getVideoDriver()),
        AssetManager(device->getAssetManager()),
        GUI(ext::cegui::createGUIManager(device)),
        ShaderManager(new CShaderManager(Driver->getGPUProgrammingServices(), device->getIncludeHandler(), GUIState, Camera))
{
    TextureSlotMap = {
        { ETEXTURE_SLOT::TEXTURE_AO,
        std::make_tuple("AOTextureBuffer", // Texture buffer name
            "AOTexture", // Texture name
            "MaterialParamsWindow/AOWindow/ImageButton") },

        { ETEXTURE_SLOT::TEXTURE_BUMP,
            std::make_tuple("BumpTextureBuffer", // Texture buffer name
                "BumpTexture", // Texture name
                "MaterialParamsWindow/BumpWindow/ImageButton") },

        { ETEXTURE_SLOT::TEXTURE_SLOT_1,
            std::make_tuple("T1TextureBuffer", // Texture buffer name
                "T1Texture", // Texture name
                "TextureViewWindow/Texture0Window/Texture") },

        { ETEXTURE_SLOT::TEXTURE_SLOT_2,
            std::make_tuple("T2TextureBuffer", // Texture buffer name
                "T2Texture", // Texture name
                "TextureViewWindow/Texture1Window/Texture") },

        { ETEXTURE_SLOT::TEXTURE_SLOT_3,
            std::make_tuple("T3TextureBuffer", // Texture buffer name
                "T3Texture", // Texture name
                "TextureViewWindow/Texture2Window/Texture") },

        { ETEXTURE_SLOT::TEXTURE_SLOT_4,
            std::make_tuple("T4TextureBuffer", // Texture buffer name
                "T4Texture", // Texture name
                "TextureViewWindow/Texture3Window/Texture") },
    };

    GUI->init();
    GUI->createRootWindowFromLayout(
        ext::cegui::readWindowLayout("../../media/brdf_explorer/MainWindow.layout")
    );
    auto onColorPicked = [](const ::CEGUI::Colour& _ceguiColor, core::vector3df& _irrColor) {
        _irrColor.X = _ceguiColor.getRed();
        _irrColor.Y = _ceguiColor.getGreen();
        _irrColor.Z = _ceguiColor.getBlue();
    };
    GUI->createColourPicker(false, "LightParamsWindow/ColorWindow", "Color", "pickerLightColor", std::bind(onColorPicked, std::placeholders::_1, std::ref(GUIState.Light.Color)));
    GUI->createColourPicker(true, "MaterialParamsWindow/EmissiveWindow", "Emissive", "pickerEmissiveColor", std::bind(onColorPicked, std::placeholders::_1, std::ref(GUIState.Emissive.Color)));
    GUI->createColourPicker(true, "MaterialParamsWindow/AlbedoWindow", "Albedo", "pickerAlbedoColor", std::bind(onColorPicked, std::placeholders::_1, std::ref(GUIState.Albedo.ConstantColor)));

    // Fill all the available texture slots using the default (no texture) image
    //const auto image_default = ext::cegui::loadImage("../../media/brdf_explorer/DefaultEmpty.png");
    irr::asset::ICPUTexture* cputexture_default = loadCPUTexture("../../media/brdf_explorer/DefaultEmpty.png");
    DefaultTexture = Driver->getGPUObjectsFromAssets(&cputexture_default, (&cputexture_default)+1).front();

    loadTextureSlot(ETEXTURE_SLOT::TEXTURE_AO, DefaultTexture, cputexture_default->getCacheKey());
    loadTextureSlot(ETEXTURE_SLOT::TEXTURE_BUMP, DefaultTexture, cputexture_default->getCacheKey());
    loadTextureSlot(ETEXTURE_SLOT::TEXTURE_SLOT_1, DefaultTexture, cputexture_default->getCacheKey());
    loadTextureSlot(ETEXTURE_SLOT::TEXTURE_SLOT_2, DefaultTexture, cputexture_default->getCacheKey());
    loadTextureSlot(ETEXTURE_SLOT::TEXTURE_SLOT_3, DefaultTexture, cputexture_default->getCacheKey());
    loadTextureSlot(ETEXTURE_SLOT::TEXTURE_SLOT_4, DefaultTexture, cputexture_default->getCacheKey());

    auto root = GUI->getRootWindow();
    // Material window: Subscribe to sliders' events and set its default value to
    // 0.0.
    GUI->registerSliderEvent(
        "MaterialParamsWindow/RefractionIndexWindow/Slider", sliderRIRange, 0.01f,
        [root,this](const ::CEGUI::EventArgs&) {
            auto refractionIndex = static_cast<::CEGUI::Slider*>(
                root->getChild(
                    "MaterialParamsWindow/RefractionIndexWindow/Slider"))
                                 ->getCurrentValue();
            root->getChild(
                    "MaterialParamsWindow/RefractionIndexWindow/LabelPercent")
                ->setText(ext::cegui::toStringFloat(refractionIndex, 2));

            GUIState.RefractionIndex.ConstValue = refractionIndex;
        });

    GUI->registerSliderEvent(
        "MaterialParamsWindow/MetallicWindow/Slider", sliderMetallicRange, 0.01f,
        [root,this](const ::CEGUI::EventArgs&) {
            auto metallic = static_cast<::CEGUI::Slider*>(
                root->getChild("MaterialParamsWindow/MetallicWindow/Slider"))
                                 ->getCurrentValue();
            root->getChild("MaterialParamsWindow/MetallicWindow/LabelPercent")
                ->setText(ext::cegui::toStringFloat(metallic, 2));

            GUIState.Metallic.ConstValue = metallic;
        });

    GUI->registerSliderEvent(
        "MaterialParamsWindow/RoughnessWindow/Slider", sliderRoughness1Range,
        0.01f, [this](const ::CEGUI::EventArgs&) {
            auto root = GUI->getRootWindow();

            const auto v = static_cast<::CEGUI::Slider*>(
                root->getChild("MaterialParamsWindow/RoughnessWindow/Slider"))
                               ->getCurrentValue();
            const auto s = ext::cegui::toStringFloat(v, 2);

            root->getChild("MaterialParamsWindow/RoughnessWindow/LabelPercent1")
                ->setText(s);

            GUIState.Roughness.ConstValue1 = v;

            if (GUIState.Roughness.IsIsotropic) {
                root->getChild("MaterialParamsWindow/RoughnessWindow/LabelPercent2")
                    ->setText(s);
                static_cast<::CEGUI::Slider*>(
                    root->getChild("MaterialParamsWindow/RoughnessWindow/Slider2"))
                    ->setCurrentValue(v);
            }
        });

    GUI->registerSliderEvent(
        "MaterialParamsWindow/RoughnessWindow/Slider2", sliderRoughness2Range,
        0.01f, [root,this](const ::CEGUI::EventArgs&) {
            auto roughness = static_cast<::CEGUI::Slider*>(
                root->getChild("MaterialParamsWindow/RoughnessWindow/Slider2"))
                                 ->getCurrentValue();
            root->getChild("MaterialParamsWindow/RoughnessWindow/LabelPercent2")
                ->setText(ext::cegui::toStringFloat(roughness, 2));

            GUIState.Roughness.ConstValue2 = roughness;
        });

    // Set the sliders' text objects to their default value (whatever value the
    // slider was set to).
    {
        // Roughness slider, first one
        root->getChild("MaterialParamsWindow/RoughnessWindow/LabelPercent2")
            ->setText(ext::cegui::toStringFloat(
                static_cast<::CEGUI::Slider*>(
                    root->getChild("MaterialParamsWindow/RoughnessWindow/Slider2"))
                    ->getCurrentValue(),
                2));

        // Roughness slider, second one
        root->getChild("MaterialParamsWindow/RoughnessWindow/LabelPercent1")
            ->setText(ext::cegui::toStringFloat(
                static_cast<::CEGUI::Slider*>(
                    root->getChild("MaterialParamsWindow/RoughnessWindow/Slider"))
                    ->getCurrentValue(),
                2));

        // Refractive index slider
        root->getChild("MaterialParamsWindow/RefractionIndexWindow/LabelPercent")
            ->setText(ext::cegui::toStringFloat(
                static_cast<::CEGUI::Slider*>(
                    root->getChild(
                        "MaterialParamsWindow/RefractionIndexWindow/Slider"))
                    ->getCurrentValue(),
                2));

        // Metallic slider
        root->getChild("MaterialParamsWindow/MetallicWindow/LabelPercent")
            ->setText(ext::cegui::toStringFloat(
                static_cast<::CEGUI::Slider*>(
                    root->getChild("MaterialParamsWindow/MetallicWindow/Slider"))
                    ->getCurrentValue(),
                2));

        // Bump-mapping's height slider
        root->getChild("MaterialParamsWindow/BumpWindow/LabelPercent")
            ->setText(ext::cegui::toStringFloat(
                static_cast<::CEGUI::Slider*>(
                    root->getChild("MaterialParamsWindow/BumpWindow/Spinner"))
                    ->getCurrentValue(),
                2));
    }

    // light animation checkbox
    auto lightAnimated = static_cast<::CEGUI::ToggleButton*>(root->getChild("LightParamsWindow/AnimationWindow/Checkbox"));
    lightAnimated->subscribeEvent(
        ::CEGUI::ToggleButton::EventSelectStateChanged,
        [this](const ::CEGUI::EventArgs& e) {
            const ::CEGUI::WindowEventArgs& we = static_cast<const ::CEGUI::WindowEventArgs&>(e);
            GUIState.Light.Animated = static_cast<::CEGUI::ToggleButton*>(we.window)->isSelected();

            auto root = GUI->getRootWindow();
            root->getChild("LightParamsWindow/PositionWindow")->setDisabled(GUIState.Light.Animated);
        }
    );

    auto lightZ = static_cast<::CEGUI::Spinner*>(root->getChild("LightParamsWindow/PositionWindow/LightZ"));
    lightZ->subscribeEvent(
        ::CEGUI::Spinner::EventValueChanged,
        [this](const ::CEGUI::EventArgs& e) {
            const ::CEGUI::WindowEventArgs& we = static_cast<const ::CEGUI::WindowEventArgs&>(e);
            GUIState.Light.ConstantPosition.Z = static_cast<::CEGUI::Spinner*>(we.window)->getCurrentValue();
        }
    );
    auto lightY = static_cast<::CEGUI::Spinner*>(root->getChild("LightParamsWindow/PositionWindow/LightY"));
    lightY->subscribeEvent(
        ::CEGUI::Spinner::EventValueChanged,
        [this](const ::CEGUI::EventArgs& e) {
            const ::CEGUI::WindowEventArgs& we = static_cast<const ::CEGUI::WindowEventArgs&>(e);
            GUIState.Light.ConstantPosition.Y = static_cast<::CEGUI::Spinner*>(we.window)->getCurrentValue();
        }
    );
    auto lightX = static_cast<::CEGUI::Spinner*>(root->getChild("LightParamsWindow/PositionWindow/LightX"));
    lightX->subscribeEvent(
        ::CEGUI::Spinner::EventValueChanged,
        [this](const ::CEGUI::EventArgs& e) {
            const ::CEGUI::WindowEventArgs& we = static_cast<const ::CEGUI::WindowEventArgs&>(e);
            GUIState.Light.ConstantPosition.X = static_cast<::CEGUI::Spinner*>(we.window)->getCurrentValue();
        }
    );
    
    // Isotropic checkbox
    auto isotropic = static_cast<::CEGUI::ToggleButton*>(root->getChild("MaterialParamsWindow/RoughnessWindow/Checkbox"));
    isotropic->subscribeEvent(
        ::CEGUI::ToggleButton::EventSelectStateChanged,
        [this](const ::CEGUI::EventArgs& e) {
            auto root = GUI->getRootWindow();

            const ::CEGUI::WindowEventArgs& we = static_cast<const ::CEGUI::WindowEventArgs&>(e);
            GUIState.Roughness.IsIsotropic = static_cast<::CEGUI::ToggleButton*>(we.window)->isSelected();
            static_cast<::CEGUI::Slider*>(
                root->getChild("MaterialParamsWindow/RoughnessWindow/Slider2"))
                ->setDisabled(GUIState.Roughness.IsIsotropic);

            if (GUIState.Roughness.IsIsotropic) {
                root->getChild("MaterialParamsWindow/RoughnessWindow/LabelPercent2")
                    ->setText(ext::cegui::toStringFloat(
                        static_cast<::CEGUI::Slider*>(
                            root->getChild(
                                "MaterialParamsWindow/RoughnessWindow/Slider"))
                            ->getCurrentValue(),
                        2));

                static_cast<::CEGUI::Slider*>(
                    root->getChild("MaterialParamsWindow/RoughnessWindow/Slider2"))
                    ->setCurrentValue(
                        static_cast<::CEGUI::Slider*>(
                            root->getChild(
                                "MaterialParamsWindow/RoughnessWindow/Slider"))
                            ->getCurrentValue());
            }
        });

    // Load Model button
    auto button_loadModel = static_cast<::CEGUI::PushButton*>(
        root->getChild("LoadModelButton"));

    button_loadModel->subscribeEvent(::CEGUI::PushButton::EventClicked,
        ::CEGUI::Event::Subscriber(&BRDFExplorerApp::eventMeshBrowse, this));

    // AO texturing & bump-mapping texturing window
    auto button_browse_AO = static_cast<::CEGUI::PushButton*>(
        root->getChild("MaterialParamsWindow/AOWindow/Button"));

    button_browse_AO->subscribeEvent(::CEGUI::PushButton::EventClicked,
        ::CEGUI::Event::Subscriber(&BRDFExplorerApp::eventAOTextureBrowse, this));

    static_cast<::CEGUI::DefaultWindow*>(
        root->getChild("MaterialParamsWindow/AOWindow/ImageButton"))
        ->subscribeEvent(::CEGUI::Window::EventMouseClick,
            ::CEGUI::Event::Subscriber(&BRDFExplorerApp::eventAOTextureBrowse, this));

    static_cast<::CEGUI::Editbox*>(root->getChild("MaterialParamsWindow/AOWindow/Editbox"))
        ->subscribeEvent(::CEGUI::Editbox::EventTextAccepted,
            ::CEGUI::Event::Subscriber(&BRDFExplorerApp::eventAOTextureBrowse_EditBox, this));

    auto ao_enabled = static_cast<::CEGUI::ToggleButton*>(root->getChild("MaterialParamsWindow/AOWindow/Checkbox"));
    ao_enabled->subscribeEvent(
        ::CEGUI::ToggleButton::EventSelectStateChanged,
        [this](const ::CEGUI::EventArgs& e) {
            auto root = GUI->getRootWindow();

            const ::CEGUI::WindowEventArgs& we = static_cast<const ::CEGUI::WindowEventArgs&>(e);
            GUIState.AmbientOcclusion.Enabled = static_cast<::CEGUI::ToggleButton*>(we.window)->isSelected();
        });

    auto button_browse_bump_map = static_cast<::CEGUI::PushButton*>(
        root->getChild("MaterialParamsWindow/BumpWindow/Button"));

    button_browse_bump_map->subscribeEvent(::CEGUI::PushButton::EventClicked,
        ::CEGUI::Event::Subscriber(&BRDFExplorerApp::eventBumpTextureBrowse, this));

    static_cast<::CEGUI::DefaultWindow*>(root->getChild("MaterialParamsWindow/BumpWindow/ImageButton"))
        ->subscribeEvent(::CEGUI::Window::EventMouseClick,
            ::CEGUI::Event::Subscriber(&BRDFExplorerApp::eventBumpTextureBrowse, this));

    static_cast<::CEGUI::Editbox*>(root->getChild("MaterialParamsWindow/BumpWindow/Editbox"))
        ->subscribeEvent(::CEGUI::Editbox::EventTextAccepted,
            ::CEGUI::Event::Subscriber(&BRDFExplorerApp::eventBumpTextureBrowse_EditBox, this));

    GUI->registerSliderEvent(
        "MaterialParamsWindow/BumpWindow/Spinner", sliderBumpHeightRange, 1.0f,
        [root,this](const ::CEGUI::EventArgs&) {
            auto height = static_cast<::CEGUI::Slider*>(
                root->getChild("MaterialParamsWindow/BumpWindow/Spinner"))
                                 ->getCurrentValue();
            root->getChild("MaterialParamsWindow/BumpWindow/LabelPercent")
                ->setText(ext::cegui::toStringFloat(height, 2));
            GUIState.BumpMapping.Height = height;
        });
    initDropdown();
    initTooltip();

    // Setting up the texture preview window
    std::array<::CEGUI::PushButton*, 4> texturePreviewIcon = {
        static_cast<::CEGUI::PushButton*>(
            root->getChild("TextureViewWindow/Texture0Window/Texture")),
        static_cast<::CEGUI::PushButton*>(
            root->getChild("TextureViewWindow/Texture1Window/Texture")),
        static_cast<::CEGUI::PushButton*>(
            root->getChild("TextureViewWindow/Texture2Window/Texture")),
        static_cast<::CEGUI::PushButton*>(
            root->getChild("TextureViewWindow/Texture3Window/Texture"))
    };

    for (const auto& v : texturePreviewIcon)
    {
        v->subscribeEvent(::CEGUI::PushButton::EventClicked,
            ::CEGUI::Event::Subscriber(&BRDFExplorerApp::eventTextureBrowse, this));
    }

    // Setting up the master windows & their default opacity
    auto* window_material = static_cast<::CEGUI::FrameWindow*>(root->getChild("MaterialParamsWindow"));
    window_material->subscribeEvent(
        ::CEGUI::FrameWindow::EventCloseClicked, [root](const ::CEGUI::EventArgs&) {
            static_cast<::CEGUI::FrameWindow*>(
                root->getChild("MaterialParamsWindow"))
                ->setVisible(false);
        });
    GUI->setOpacity("MaterialParamsWindow", defaultOpacity);

    auto* window_light = static_cast<::CEGUI::FrameWindow*>(root->getChild("LightParamsWindow"));
    window_light->subscribeEvent(::CEGUI::FrameWindow::EventCloseClicked,
        [root](const CEGUI::EventArgs&) {
            static_cast<CEGUI::FrameWindow*>(
                root->getChild("LightParamsWindow"))
                ->setVisible(false);
        });
    GUI->setOpacity("LightParamsWindow", defaultOpacity);

    auto* window_texture = static_cast<::CEGUI::FrameWindow*>(root->getChild("TextureViewWindow"));
    window_texture->subscribeEvent(
        ::CEGUI::FrameWindow::EventCloseClicked, [root](const ::CEGUI::EventArgs&) {
            static_cast<::CEGUI::FrameWindow*>(
                root->getChild("TextureViewWindow"))
                ->setVisible(false);
        });
    GUI->setOpacity("TextureViewWindow", defaultOpacity);
}

void BRDFExplorerApp::initDropdown()
{
    static const std::vector<const char*> drop_ID = {
        "Constant", "Texture 0", "Texture 1", "Texture 2", "Texture 3"
    };
    const auto default_halignment = ::CEGUI::HA_RIGHT;
    const auto default_width = ::CEGUI::UDim(0.5f, 0.0f);
    const auto default_position = ::CEGUI::UVector2(::CEGUI::UDim(0.0f, 0.0f), ::CEGUI::UDim(0.125f, 0.0f));

    auto root = GUI->getRootWindow();

    auto* albedo_drop = GUI->createDropDownList(
        "MaterialParamsWindow/AlbedoDropDownList", "DropDown_Albedo", drop_ID,
        [this](const ::CEGUI::EventArgs&) {
            auto root = GUI->getRootWindow();
            auto* list = static_cast<::CEGUI::Combobox*>(root->getChild(
                "MaterialParamsWindow/AlbedoDropDownList/DropDown_Albedo"));
            list->setProperty("NormalEditTextColour", GUI->WhiteProperty);

            root->getChild("MaterialParamsWindow/AlbedoWindow")
                ->setDisabled(list->getSelectedItem()->getText() != "Constant");

            GUIState.Albedo.SourceDropdown = getDropdownState(DROPDOWN_ALBEDO_NAME);
        });

    albedo_drop->setHorizontalAlignment(default_halignment);
    albedo_drop->setWidth(default_width);
    albedo_drop->setPosition(default_position);

    auto* roughness_drop = GUI->createDropDownList(
        "MaterialParamsWindow/RoughnessDropDownList", "DropDown_Roughness",
        drop_ID, [this](const ::CEGUI::EventArgs&) {
            auto root = GUI->getRootWindow();
            auto* list = static_cast<CEGUI::Combobox*>(root->getChild(
                "MaterialParamsWindow/RoughnessDropDownList/DropDown_Roughness"));
            list->setProperty("NormalEditTextColour", GUI->WhiteProperty);

            root->getChild("MaterialParamsWindow/RoughnessWindow")
                ->setDisabled(list->getSelectedItem()->getText() != "Constant");

            GUIState.Roughness.SourceDropdown = getDropdownState(DROPDOWN_ROUGHNESS_NAME);
        });

    roughness_drop->setHorizontalAlignment(default_halignment);
    roughness_drop->setWidth(default_width);
    roughness_drop->setPosition(default_position);

    auto* ri_drop = GUI->createDropDownList(
        "MaterialParamsWindow/RIDropDownList", "DropDown_RI", drop_ID,
        [this](const ::CEGUI::EventArgs&) {
            auto root = GUI->getRootWindow();

            auto* list = static_cast<::CEGUI::Combobox*>(
                root->getChild("MaterialParamsWindow/RIDropDownList/DropDown_RI"));
            list->setProperty("NormalEditTextColour", GUI->WhiteProperty);

            root->getChild("MaterialParamsWindow/RefractionIndexWindow")
                ->setDisabled(list->getSelectedItem()->getText() != "Constant");

            GUIState.RefractionIndex.SourceDropdown = getDropdownState(DROPDOWN_RI_NAME);
        });

    ri_drop->setHorizontalAlignment(default_halignment);
    ri_drop->setWidth(default_width);
    ri_drop->setPosition(default_position);

    auto* metallic_drop = GUI->createDropDownList(
        "MaterialParamsWindow/MetallicDropDownList", "DropDown_Metallic", drop_ID,
        [this](const ::CEGUI::EventArgs&) {
            auto root = GUI->getRootWindow();

            auto* list = static_cast<::CEGUI::Combobox*>(root->getChild(
                "MaterialParamsWindow/MetallicDropDownList/DropDown_Metallic"));
            list->setProperty("NormalEditTextColour", GUI->WhiteProperty);

            root->getChild("MaterialParamsWindow/MetallicWindow")
                ->setDisabled(list->getSelectedItem()->getText() != "Constant");

            GUIState.Metallic.SourceDropdown = getDropdownState(DROPDOWN_METALLIC_NAME);
        });

    metallic_drop->setHorizontalAlignment(default_halignment);
    metallic_drop->setWidth(default_width);
    metallic_drop->setPosition(default_position);
}

void BRDFExplorerApp::initTooltip()
{
    auto root = GUI->getRootWindow();

    static_cast<CEGUI::DefaultWindow*>(
        root->getChild("MaterialParamsWindow/BumpWindow/ImageButton"))
        ->setTooltipText("Left-click to select a bump-mapping texture.");
    static_cast<CEGUI::DefaultWindow*>(
        root->getChild("MaterialParamsWindow/AOWindow/ImageButton"))
        ->setTooltipText("Left-click to select an AO texture.");
    static_cast<CEGUI::DefaultWindow*>(
        root->getChild("TextureViewWindow/Texture0Window/Texture"))
        ->setTooltipText("Left-click to select a new texture.");
    static_cast<CEGUI::DefaultWindow*>(
        root->getChild("TextureViewWindow/Texture1Window/Texture"))
        ->setTooltipText("Left-click to select a new texture.");
    static_cast<CEGUI::DefaultWindow*>(
        root->getChild("TextureViewWindow/Texture2Window/Texture"))
        ->setTooltipText("Left-click to select a new texture.");
    static_cast<CEGUI::DefaultWindow*>(
        root->getChild("TextureViewWindow/Texture3Window/Texture"))
        ->setTooltipText("Left-click to select a new texture.");
}

void BRDFExplorerApp::renderGUI()
{
    GUI->render();
}

void BRDFExplorerApp::renderMesh()
{
    if (!Mesh)
        return;

    CShaderManager::SParams params;
    params.constantAlbedo = (GUIState.Albedo.SourceDropdown==EDS_CONSTANT);
    params.constantMetallic = (GUIState.Metallic.SourceDropdown==EDS_CONSTANT);
    params.constantRI = (GUIState.RefractionIndex.SourceDropdown==EDS_CONSTANT);
    params.constantRoughness = (GUIState.Roughness.SourceDropdown==EDS_CONSTANT);
    params.AOEnabled = GUIState.AmbientOcclusion.Enabled;
    params.isotropicRoughness = GUIState.Roughness.IsIsotropic;
    params.metallicIsOne = (GUIState.Metallic.ConstValue==1.f);
    params.metallicIsZero = (GUIState.Metallic.ConstValue==0.f);
    params.roughnessIsZero = (GUIState.Roughness.ConstValue1==0.f);

    Material.MaterialType = ShaderManager->getShader(params);

    irr::video::IGPUMeshBuffer* meshbuffer = Mesh->getMeshBuffer(MESHBUFFER_NUM);
    Driver->setMaterial(Material);
    Driver->drawMeshBuffer(meshbuffer);
}

void BRDFExplorerApp::loadTextureSlot(ETEXTURE_SLOT slot, irr::asset::ICPUTexture* _texture)
{
    auto tupl = TextureSlotMap[slot];
    auto root = GUI->getRootWindow();
    auto& renderer = GUI->getRenderer();

    auto gputex = Driver->getGPUObjectsFromAssets(&_texture, (&_texture)+1).front();
    ::CEGUI::Sizef texSize;
    texSize.d_width = gputex->getSize()[0];
    texSize.d_height = gputex->getSize()[1];

    Material.setTexture(slot-TEXTURE_SLOT_1, gputex);

    ::CEGUI::Texture& ceguiTexture = !renderer.isTextureDefined(_texture->getCacheKey())
        ? irrTex2ceguiTex(getTextureGLname(gputex), texSize, _texture->getCacheKey(), renderer)
        : renderer.getTexture(_texture->getCacheKey());

    ::CEGUI::BasicImage& image = !::CEGUI::ImageManager::getSingleton().isDefined(std::get<0>(tupl))
        ? static_cast<::CEGUI::BasicImage&>(::CEGUI::ImageManager::getSingleton().create(
              "BasicImage", std::get<0>(tupl)))
        : static_cast<::CEGUI::BasicImage&>(
              ::CEGUI::ImageManager::getSingleton().get(std::get<0>(tupl)));
    image.setTexture(&ceguiTexture);
    image.setArea(::CEGUI::Rectf(0, 0, texSize.d_width, texSize.d_height));
    image.setAutoScaled(::CEGUI::AutoScaledMode::ASM_Both);

    static const std::vector<const char*> property = { "NormalImage", "HoverImage", "PushedImage" };

    for (const auto& v : property) {
        root->getChild(std::get<2>(tupl))->setProperty(v, std::get<0>(tupl));
    }
}

void BRDFExplorerApp::loadTextureSlot(ETEXTURE_SLOT slot, irr::video::IVirtualTexture* _texture, const std::string& _texName)
{
    auto tupl = TextureSlotMap[slot];
    auto root = GUI->getRootWindow();
    auto& renderer = GUI->getRenderer();

    auto gputex = _texture;
    ::CEGUI::Sizef texSize;
    texSize.d_width = gputex->getSize()[0];
    texSize.d_height = gputex->getSize()[1];

    Material.setTexture(slot-TEXTURE_SLOT_1, gputex);

    ::CEGUI::Texture& ceguiTexture = !renderer.isTextureDefined(_texName)
        ? irrTex2ceguiTex(getTextureGLname(gputex), texSize, _texName, renderer)
        : renderer.getTexture(_texName);

    ::CEGUI::BasicImage& image = !::CEGUI::ImageManager::getSingleton().isDefined(std::get<0>(tupl))
        ? static_cast<::CEGUI::BasicImage&>(::CEGUI::ImageManager::getSingleton().create(
              "BasicImage", std::get<0>(tupl)))
        : static_cast<::CEGUI::BasicImage&>(
              ::CEGUI::ImageManager::getSingleton().get(std::get<0>(tupl)));
    image.setTexture(&ceguiTexture);
    image.setArea(::CEGUI::Rectf(0, 0, texSize.d_width, texSize.d_height));
    image.setAutoScaled(::CEGUI::AutoScaledMode::ASM_Both);

    static const std::vector<const char*> property = { "NormalImage", "HoverImage", "PushedImage" };

    for (const auto& v : property) {
        root->getChild(std::get<2>(tupl))->setProperty(v, std::get<0>(tupl));
    }
}

irr::asset::ICPUTexture* BRDFExplorerApp::loadCPUTexture(const std::string& _path)
{
    irr::asset::IAssetLoader::SAssetLoadParams lparams;
    return static_cast<irr::asset::ICPUTexture*>(AssetManager.getAsset(_path, lparams));
}

auto BRDFExplorerApp::loadMesh(const std::string& _path) -> SCPUGPUMesh
{
    irr::asset::IAssetLoader::SAssetLoadParams lparams;
    irr::asset::ICPUMesh* cpumesh = static_cast<irr::asset::ICPUMesh*>(AssetManager.getAsset(_path, lparams));
    if (!cpumesh)
        return {nullptr, nullptr};

    irr::video::IGPUMesh* gpumesh = Driver->getGPUObjectsFromAssets(&cpumesh, (&cpumesh)+1).front();

    return {cpumesh, gpumesh};
}

void BRDFExplorerApp::loadMeshAndReplaceTextures(const std::string& _path)
{
    auto loadedMesh = loadMesh(_path);
    if (!loadedMesh.cpu)
        return;

    Mesh = loadedMesh.gpu;

    const irr::video::SGPUMaterial& itsMaterial = Mesh->getMeshBuffer(MESHBUFFER_NUM)->getMaterial();

    for (uint32_t t = 0u; t < 4u; ++t)
    {
        if (Material.getTexture(t)==DefaultTexture && itsMaterial.getTexture(t))
        {
            irr::video::IVirtualTexture* newtex = itsMaterial.getTexture(t);
            std::string texname = loadedMesh.cpu->getMeshBuffer(MESHBUFFER_NUM)->getMaterial().getTexture(t)->getCacheKey();
            loadTextureSlot(static_cast<ETEXTURE_SLOT>(TEXTURE_SLOT_1 + t), newtex, texname);
        }
    }
}

void BRDFExplorerApp::updateTooltip(const char* name, const char* text)
{
    std::string s(text);
    ext::cegui::Replace(s, "\\", "\\\\");

    static_cast<CEGUI::DefaultWindow*>(GUI->getRootWindow()->getChild(name))
        ->setTooltipText(s.c_str());
}

auto BRDFExplorerApp::getDropdownState(const char* _dropdownName) const -> E_DROPDOWN_STATE
{
    auto root = GUI->getRootWindow();
    auto* list = static_cast<::CEGUI::Combobox*>(root->getChild(_dropdownName));

    auto mapStrToEnum = [] (const std::string& _str) {
        const char* Texture = "Texture";
        if (_str.compare(0, strlen(Texture), Texture) == 0)
            return static_cast<E_DROPDOWN_STATE>(EDS_TEX0 + _str[strlen(Texture)+1]-'0');
        else return EDS_CONSTANT;
    };

    return mapStrToEnum(list->getSelectedItem()->getText());
}

void BRDFExplorerApp::showErrorMessage(const char* title, const char* message)
{
    auto root = GUI->getRootWindow();
    if (!root->isChild("MessageBoxRoot")) {
        CEGUI::Window* layout = CEGUI::WindowManager::getSingleton().loadLayoutFromFile(
            "MessageBox.layout");
        layout->setVisible(false);
        layout->setAlwaysOnTop(true);
        layout->setSize(
            CEGUI::USize(CEGUI::UDim(0.5, 0.0f), CEGUI::UDim(0.2f, 0.0f)));
        layout->setHorizontalAlignment(CEGUI::HA_CENTRE);
        layout->setVerticalAlignment(CEGUI::VA_CENTRE);

        static_cast<CEGUI::PushButton*>(
            layout->getChild("FrameWindow/ButtonWindow/Button"))
            ->subscribeEvent(
                CEGUI::PushButton::EventClicked,
                [root](const CEGUI::EventArgs&) {
                    root->getChild("MessageBoxRoot")->setVisible(false);
                });

        root->addChild(layout);
    }

    auto header = static_cast<CEGUI::DefaultWindow*>(root->getChild("MessageBoxRoot"));
    header->setVisible(true);
    header->activate();

    auto frame = static_cast<CEGUI::FrameWindow*>(header->getChild("FrameWindow"));
    frame->setText(title);
    static_cast<CEGUI::DefaultWindow*>(frame->getChild("Label"))
        ->setText(message);
}

void BRDFExplorerApp::eventAOTextureBrowse(const ::CEGUI::EventArgs&)
{
    const auto p = GUI->openFileDialog(ImageFileDialogTitle, ImageFileDialogFilters);

    if (p.first) {
        auto box = static_cast<CEGUI::Editbox*>(
            GUI->getRootWindow()->getChild("MaterialParamsWindow/AOWindow/Editbox"));

        irr::asset::ICPUTexture* cputexture = loadCPUTexture(p.second);
        loadTextureSlot(ETEXTURE_SLOT::TEXTURE_AO, cputexture);

        box->setText(p.second);
        updateTooltip(
            "MaterialParamsWindow/AOWindow/ImageButton",
            ext::cegui::ssprintf("%s (%ux%u)\nLeft-click to select a new texture.",
                p.second.c_str(), cputexture->getSize()[0], cputexture->getSize()[1])
                .c_str());
    }
}

void BRDFExplorerApp::eventAOTextureBrowse_EditBox(const ::CEGUI::EventArgs&)
{
    auto box = static_cast<CEGUI::Editbox*>(
        GUI->getRootWindow()->getChild("MaterialParamsWindow/AOWindow/Editbox"));

    if (ext::cegui::Exists(box->getText().c_str())) {
        irr::asset::ICPUTexture* cputexture = loadCPUTexture(box->getText());
        loadTextureSlot(ETEXTURE_SLOT::TEXTURE_AO, cputexture);

        updateTooltip(
            "MaterialParamsWindow/AOWindow/ImageButton",
            irr::ext::cegui::ssprintf("%s (%ux%u)\nLeft-click to select a new texture.",
                box->getText().c_str(), cputexture->getSize()[0], cputexture->getSize()[1])
                .c_str());
    } else {
        std::string s;
        s += std::string(box->getText().c_str()) + ": The file couldn't be opened.";
        ext::cegui::Replace(s, "\\", "\\\\");
        showErrorMessage("Error", s.c_str());
    }
}

void BRDFExplorerApp::eventBumpTextureBrowse(const ::CEGUI::EventArgs&)
{
    const auto p = GUI->openFileDialog(ImageFileDialogTitle, ImageFileDialogFilters);

    if (p.first) {
        auto box = static_cast<CEGUI::Editbox*>(
            GUI->getRootWindow()->getChild("MaterialParamsWindow/BumpWindow/Editbox"));
        irr::asset::ICPUTexture* cputexture = loadCPUTexture(p.second);
        loadTextureSlot(ETEXTURE_SLOT::TEXTURE_BUMP, cputexture);

        box->setText(p.second);
        updateTooltip(
            "MaterialParamsWindow/BumpWindow/ImageButton",
            ext::cegui::ssprintf("%s (%ux%u)\nLeft-click to select a new texture.",
                p.second.c_str(), cputexture->getSize()[0], cputexture->getSize()[1])
                .c_str());
    }
}

void BRDFExplorerApp::eventBumpTextureBrowse_EditBox(const ::CEGUI::EventArgs&)
{
    auto box = static_cast<CEGUI::Editbox*>(
        GUI->getRootWindow()->getChild("MaterialParamsWindow/BumpWindow/Editbox"));

    if (ext::cegui::Exists(box->getText().c_str())) {
        irr::asset::ICPUTexture* cputexture = loadCPUTexture(box->getText());
        loadTextureSlot(ETEXTURE_SLOT::TEXTURE_BUMP, cputexture);

        updateTooltip(
            "MaterialParamsWindow/BumpWindow/ImageButton",
            ext::cegui::ssprintf("%s (%ux%u)\nLeft-click to select a new texture.",
                box->getText().c_str(), cputexture->getSize()[0], cputexture->getSize()[1])
                .c_str());
    } else {
        std::string s;
        s += std::string(box->getText().c_str()) + ": The file couldn't be opened.";
        ext::cegui::Replace(s, "\\", "\\\\");
        showErrorMessage("Error", s.c_str());
    }
}

void BRDFExplorerApp::eventTextureBrowse(const CEGUI::EventArgs& e)
{
    const CEGUI::WindowEventArgs& we = static_cast<const CEGUI::WindowEventArgs&>(e);
    const auto parent = static_cast<CEGUI::PushButton*>(we.window)->getParent()->getName();
    const auto p = GUI->openFileDialog(ImageFileDialogTitle, ImageFileDialogFilters);


    const auto path_label = ext::cegui::ssprintf("TextureViewWindow/%s/LabelWindow/Label", parent.c_str());
    const auto path_texture = ext::cegui::ssprintf("TextureViewWindow/%s/Texture", parent.c_str());

    if (p.first) {
        auto box = static_cast<CEGUI::Editbox*>(GUI->getRootWindow()->getChild(path_label));
        const auto v = ext::cegui::Split(p.second, '\\');

        ETEXTURE_SLOT type;
        if (parent == "Texture0Window")
            type = ETEXTURE_SLOT::TEXTURE_SLOT_1;
        else if (parent == "Texture1Window")
            type = ETEXTURE_SLOT::TEXTURE_SLOT_2;
        else if (parent == "Texture2Window")
            type = ETEXTURE_SLOT::TEXTURE_SLOT_3;
        else if (parent == "Texture3Window")
            type = ETEXTURE_SLOT::TEXTURE_SLOT_4;

        irr::asset::ICPUTexture* cputexture = loadCPUTexture(p.second);
        loadTextureSlot(type, cputexture);

        box->setText(v[v.size() - 1]);
        updateTooltip(
            path_texture.c_str(),
            ext::cegui::ssprintf("%s (%ux%u)\nLeft-click to select a new texture.",
                p.second.c_str(), cputexture->getSize()[0], cputexture->getSize()[1])
                .c_str());
    }
}

void BRDFExplorerApp::eventMeshBrowse(const CEGUI::EventArgs& e)
{
    const auto p = GUI->openFileDialog(MeshFileDialogTitle, MeshFileDialogFilters);

    if (p.first)
    {
        loadMeshAndReplaceTextures(p.second);
    }
}


BRDFExplorerApp::~BRDFExplorerApp()
{
    delete ShaderManager;
}

} // namespace irr
