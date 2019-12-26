#ifndef __IRR_C_GRAPHICS_PIPELINE_LOADER_MTL_H_INCLUDED__
#define __IRR_C_GRAPHICS_PIPELINE_LOADER_MTL_H_INCLUDED__

#include "IrrCompileConfig.h"
#include "irr/asset/IAssetLoader.h"

namespace irr
{
namespace asset
{	
	//! OpenEXR loader capable of loading .exr files
	class CGraphicsPipelineLoaderMTL final : public asset::IAssetLoader
	{
        struct SMtl
        {
            std::string name;

            //Ka
            core::vector3df_SIMD ambient = core::vector3df_SIMD(1.f);
            //Kd
            core::vector3df_SIMD diffuse = core::vector3df_SIMD(1.f);
            //Ks
            core::vector3df_SIMD specular = core::vector3df_SIMD(1.f);
            //Ke
            core::vector3df_SIMD emissive = core::vector3df_SIMD(1.f);
            //Tf
            core::vector3df_SIMD transmissionFilter = core::vector3df_SIMD(1.f);
            //Ns, specular exponent in phong model
            float shininess = 32.f;
            //d
            float opacity = 1.f;
            //illum
            uint32_t illumModel = 0u;
            //-bm
            float bumpFactor = 1.f;

            //PBR
            //Ni, index of refraction
            float IoR = 1.6f;
            //Pr
            float roughness = 0.f;
            //Pm
            float metallic = 0.f;
            //Ps
            float sheen;
            //Pc
            float clearcoatThickness;
            //Pcr
            float clearcoatRoughness;
            //aniso
            float anisotropy = 0.f;
            //anisor
            float anisoRotation = 0.f;

            enum E_MAP_TYPE : uint32_t
            {
                EMP_AMBIENT,
                EMP_DIFFUSE,
                EMP_SPECULAR,
                EMP_EMISSIVE,
                EMP_SHININESS,
                EMP_OPACITY,
                EMP_BUMP,
                EMP_NORMAL,
                EMP_DISPLACEMENT,
                EMP_ROUGHNESS,
                EMP_METALLIC,
                EMP_SHEEN,
                EMP_REFL_POSX,
                EMP_REFL_NEGX,
                EMP_REFL_POSY,
                EMP_REFL_NEGY,
                EMP_REFL_POSZ,
                EMP_REFL_NEGZ,

                EMP_COUNT
            };

            //paths to image files, note that they're relative to the mtl file
            std::string maps[EMP_COUNT];
            //-clamp
            uint32_t clamp;
            static_assert(sizeof(clamp)*8ull >= EMP_COUNT, "SMtl::clamp is too small!");
        };

	public:
		bool isALoadableFileFormat(io::IReadFile* _file) const override;

		const char** getAssociatedFileExtensions() const override
		{
			static const char* extensions[]{ "mtl", nullptr };
			return extensions;
		}

		uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_RENDERPASS_INDEPENDENT_PIPELINE; }

		asset::SAssetBundle loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) override;

    private:
        core::vector<SMtl> readMaterials(io::IReadFile* _file) const;
        const char* readTexture(const char* _bufPtr, const char* const _bufEnd, SMtl* _currMaterial, const char* _mapType) const;
	};
}
}

#endif
