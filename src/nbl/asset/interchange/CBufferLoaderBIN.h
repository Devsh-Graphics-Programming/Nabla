// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_C_BUFFER_LOADER_H_INCLUDED_
#define _NBL_ASSET_C_BUFFER_LOADER_H_INCLUDED_

#include "nbl/asset/interchange/IAssetLoader.h"
#include "nbl/asset/ICPUBuffer.h"

namespace nbl::asset
{

//! Binaryloader capable of loading source code in binary format
class CBufferLoaderBIN final : public asset::IAssetLoader
{
	protected:
		~CBufferLoaderBIN() = default;

	public:
		CBufferLoaderBIN() {}
		bool isALoadableFileFormat(system::IFile* _file, const system::logger_opt_ptr logger) const override;

		const char** getAssociatedFileExtensions() const override
		{
			static const char* extensions[]{ "bin", nullptr };
			return extensions;
		}

		uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_BUFFER; } 

		asset::SAssetBundle loadAsset(system::IFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) override;

	private:
		struct SContext
		{
			SContext(const size_t& sizeInBytes) : sourceCodeBuffer(ICPUBuffer::create({{ sizeInBytes }})) {}
			system::IFile* file;
			core::smart_refctd_ptr<ICPUBuffer> sourceCodeBuffer;
		};
};

}
#endif
