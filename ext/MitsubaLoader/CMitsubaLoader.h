#ifndef __C_MITSUBA_LOADER_H_INCLUDED__
#define __C_MITSUBA_LOADER_H_INCLUDED__

#include "irr/asset/IAssetLoader.h"
#include "ISceneManager.h"
#include "IFileSystem.h"
#include "irr/asset/ICPUMesh.h"
#include "irr/asset/bawformat/legacy/CBAWLegacy.h"
#include "irr/asset/bawformat/CBlobsLoadingManager.h"
#include "irr/asset/ICPUSkinnedMeshBuffer.h"

#include <iostream>

namespace irr { namespace ext { namespace MitsubaLoader {

class CMitsubaLoader : public asset::IAssetLoader
{
public:
	//! Constructor
	CMitsubaLoader(IrrlichtDevice* device);

protected:
	//! Destructor
	virtual ~CMitsubaLoader() = default;

public:
	//! Check if the file might be loaded by this class
	/** Check might look into the file.
	\param file File handle to check.
	\return True if file seems to be loadable. */
	virtual bool isALoadableFileFormat(io::IReadFile* _file) const override;

	//! Returns an array of string literals terminated by nullptr
	virtual const char** getAssociatedFileExtensions() const override;

	//! Returns the assets loaded by the loader
	/** Bits of the returned value correspond to each IAsset::E_TYPE
	enumeration member, and the return value cannot be 0. */
	//virtual uint64_t getSupportedAssetTypesBitfield() const { return 0; }

	//! Loads an asset from an opened file, returns nullptr in case of failure.
	virtual asset::IAsset* loadAsset(io::IReadFile* _file, const SAssetLoadParams& _params, IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) override;

private:
	IrrlichtDevice* m_device;
	asset::IAssetManager& m_assetManager;
};

}
}
}
#endif