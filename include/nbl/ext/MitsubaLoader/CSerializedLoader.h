// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXT_MITSUBA_LOADER_C_SERIALIZED_LOADER_H_INCLUDED_
#define _NBL_EXT_MITSUBA_LOADER_C_SERIALIZED_LOADER_H_INCLUDED_

#include "nbl/asset/asset.h"

namespace nbl::ext::MitsubaLoader
{

//! Meshloader capable of loading obj meshes.
class CSerializedLoader final : public asset::IRenderpassIndependentPipelineLoader
{
	protected:
		//! Destructor
		inline ~CSerializedLoader() {}

	public:
		//! Constructor
		CSerializedLoader(asset::IAssetManager* _manager) : IRenderpassIndependentPipelineLoader(_manager) {}

		inline bool isALoadableFileFormat(io::IReadFile* _file) const override
		{
			FileHeader header;

			const size_t prevPos = _file->getPos();
			_file->seek(0u);
			_file->read(&header, sizeof(header));
			_file->seek(prevPos);

			return header==FileHeader();
		}

		inline const char** getAssociatedFileExtensions() const override
		{
			static const char* ext[]{ "serialized", nullptr };
			return ext;
		}

		inline uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_MESH; }

		//! creates/loads an animated mesh from the file.
		asset::SAssetBundle loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) override;

	private:

		struct FileHeader
		{
			uint16_t format = 0x041Cu;
			uint16_t version = 0x0004u;

			inline bool operator!=(const FileHeader& other)
			{
				return format!=other.format || version!=other.version;
			}
			inline bool operator==(const FileHeader& other) { return !operator!=(other); }
		};
		
		struct SContext
		{
			IAssetLoader::SAssetLoadContext inner;
			uint32_t meshCount;
			core::smart_refctd_dynamic_array<uint64_t> meshOffsets;
		};
};

}

#endif

