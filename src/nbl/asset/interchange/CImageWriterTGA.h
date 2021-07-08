// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_ASSET_C_IMAGE_WRITER_TGA_H_INCLUDED__
#define __NBL_ASSET_C_IMAGE_WRITER_TGA_H_INCLUDED__

#include "BuildConfigOptions.h"

#ifdef _NBL_COMPILE_WITH_TGA_WRITER_

#include "nbl/asset/interchange/IAssetWriter.h"
<<<<<<< HEAD

namespace nbl::asset
=======
#include "nbl/system/ISystem.h"
namespace nbl
{
namespace asset
>>>>>>> remotes/origin/danylo_system
{

class CImageWriterTGA : public asset::IAssetWriter
{
    core::smart_refctd_ptr<system::ISystem> m_system;
    public:
<<<<<<< HEAD
	//! constructor
	CImageWriterTGA();
=======
	    //! constructor
	    explicit CImageWriterTGA(core::smart_refctd_ptr<system::ISystem>&& sys);
>>>>>>> remotes/origin/danylo_system

        virtual const char** getAssociatedFileExtensions() const override
        {
            static const char* ext[]{ "tga", nullptr };
            return ext;
        }

        virtual uint64_t getSupportedAssetTypesBitfield() const override
        {
        	return asset::IAsset::ET_IMAGE_VIEW;
        }

        virtual uint32_t getSupportedFlags() override { return 0u; }

        virtual uint32_t getForcedFlags() { return asset::EWF_BINARY; }

<<<<<<< HEAD
        virtual bool writeAsset(
        	io::IWriteFile* _file, const SAssetWriteParams& _params,
        	IAssetWriterOverride* _override = nullptr
        ) override;
=======
        virtual bool writeAsset(system::IFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override = nullptr) override;
>>>>>>> remotes/origin/danylo_system
};

} // namespace nbl::asset

#endif
#endif

