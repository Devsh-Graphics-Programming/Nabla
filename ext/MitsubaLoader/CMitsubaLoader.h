#ifndef __C_MITSUBA_LOADER_H_INCLUDED__
#define __C_MITSUBA_LOADER_H_INCLUDED__

#include "matrix4SIMD.h"
#include "irr/asset/asset.h"
#include "IFileSystem.h"

namespace irr
{
namespace ext
{
namespace MitsubaLoader
{

class Emitter {};

class CGlobalMitsubaMetadata : public core::IReferenceCounted
{
	public:
	//protected:
		core::vector<Emitter> emitters;
};

class IMitsubaMetadata : public asset::IAssetMetadata
{
	public:
		IMitsubaMetadata(core::smart_refctd_ptr<CGlobalMitsubaMetadata>&& _gmeta) : globalMetadata(_gmeta) {}

		const char* getLoaderName() const override {return "Mistuba";}

		const std::string id;
		const core::smart_refctd_ptr<CGlobalMitsubaMetadata> globalMetadata;
};

class IMeshMetadata : public IMitsubaMetadata
{
	public:
	protected:
		core::matrix4SIMD transform;
};

class IMeshBufferMetadata : public IMitsubaMetadata
{
};

//! not used yet
class IGraphicsPipelineMetadata : public IMitsubaMetadata
{
	public:
	protected:
};


class CMitsubaLoader : public asset::IAssetLoader
{
public:
	//! Constructor
	CMitsubaLoader();

protected:
	//! Destructor
	virtual ~CMitsubaLoader() = default;

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