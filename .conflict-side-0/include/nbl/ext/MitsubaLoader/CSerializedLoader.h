// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXT_MITSUBA_C_SERIALIZED_LOADER_H_INCLUDED_
#define _NBL_EXT_MITSUBA_C_SERIALIZED_LOADER_H_INCLUDED_


#include "nbl/system/declarations.h"

#include "nbl/asset/asset.h"
#include "nbl/asset/interchange/IGeometryLoader.h"


namespace nbl::ext::MitsubaLoader
{

#ifdef _NBL_COMPILE_WITH_ZLIB_
//! Meshloader capable of loading obj meshes.
class CSerializedLoader final : public asset::IGeometryLoader
{
	public:
		inline CSerializedLoader() = default;

		inline bool isALoadableFileFormat(system::IFile* _file, const system::logger_opt_ptr logger=nullptr) const override
		{
			FileHeader header;

			system::IFile::success_t success;
			_file->read(success,&header,0,sizeof(header));
			if (!success)
				return false;

			return header==FileHeader();
		}

		inline const char** getAssociatedFileExtensions() const override
		{
			static const char* ext[]{ "serialized", nullptr };
			return ext;
		}

		//! creates/loads an animated mesh from the file.
		asset::SAssetBundle loadAsset(system::IFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) override;

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
			// TODO : this should really be a two arrays, offsets then sizes
			core::smart_refctd_dynamic_array<uint64_t> meshOffsets;
		};
};
#endif


}
#endif
