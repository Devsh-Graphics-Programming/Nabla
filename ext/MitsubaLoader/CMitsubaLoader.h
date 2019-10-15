#ifndef __C_MITSUBA_LOADER_H_INCLUDED__
#define __C_MITSUBA_LOADER_H_INCLUDED__

#include "matrix4SIMD.h"
#include "irr/asset/asset.h"
#include "IFileSystem.h"

#include "../../ext/MitsubaLoader/CGlobalMitsubaMetadata.h"

namespace irr
{
namespace ext
{
namespace MitsubaLoader
{

class CElementBSDF;

struct NastyTemporaryBitfield
{
#define MITS_TWO_SIDED		0x80000000u
#define MITS_USE_TEXTURE	0x40000000u
	uint32_t _bitfield;
};

class CMitsubaLoader : public asset::IAssetLoader
{
	public:
		//! Constructor
		CMitsubaLoader(asset::IAssetManager* _manager);

	protected:
		asset::IAssetManager* manager;


		//! Destructor
		virtual ~CMitsubaLoader() = default;

		//! TODO: change to CPU graphics pipeline
		using bsdf_ass_type = video::SCPUMaterial; // = core::smart_refctd_ptr<asset::ICPURenderpassIndependentPipeline>;
		bsdf_ass_type getBSDF(const std::string& relativeDir, CElementBSDF* bsdf, uint32_t _hierarchyLevel, asset::IAssetLoader::IAssetLoaderOverride* _override);
		core::unordered_map<CElementBSDF*,bsdf_ass_type> pipelineCache;

		//! TODO: even later when texture changes come
		using tex_ass_type = video::SMaterialLayer<asset::ICPUTexture>; // = std::pair<core::smart_refctd_ptr<asset::ICPUTextureView>,core::smart_refctd_ptr<asset::ICPUSampler> >;
		tex_ass_type getTexture(const std::string& relativeDir, CElementTexture* texture, uint32_t _hierarchyLevel, asset::IAssetLoader::IAssetLoaderOverride* _override);
		core::unordered_map<CElementTexture*,tex_ass_type> textureCache;

	public:
		//! Check if the file might be loaded by this class
		/** Check might look into the file.
		\param file File handle to check.
		\return True if file seems to be loadable. */
		bool isALoadableFileFormat(io::IReadFile* _file) const override;

		//! Returns an array of string literals terminated by nullptr
		const char** getAssociatedFileExtensions() const override;

		//! Returns the assets loaded by the loader
		/** Bits of the returned value correspond to each IAsset::E_TYPE
		enumeration member, and the return value cannot be 0. */
		uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_MESH/*|asset::IAsset::ET_SCENE|asset::IAsset::ET_IMPLEMENTATION_SPECIFIC_METADATA*/; }

		//! Loads an asset from an opened file, returns nullptr in case of failure.
		asset::SAssetBundle loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) override;
};

}
}
}
#endif