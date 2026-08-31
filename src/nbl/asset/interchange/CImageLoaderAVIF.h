// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_IMAGE_LOADER_AVIF_H_INCLUDED__
#define __NBL_ASSET_C_IMAGE_LOADER_AVIF_H_INCLUDED__

#include "nbl/core/declarations.h"

#ifdef _NBL_COMPILE_WITH_AVIF_LOADER_

#include "nbl/asset/interchange/IAssetLoader.h"


namespace nbl
{
    namespace asset
    {

        //! Surface Loader for AVIF images
        class CImageLoaderAVIF : public asset::IAssetLoader
        {
        public:
            struct SContext
            {
                char* filename = nullptr;
                system::logger_opt_ptr logger = nullptr;
            };
        private:
        protected:
            //! destructor
            virtual ~CImageLoaderAVIF();

        public:
            //! constructor
            CImageLoaderAVIF();

            virtual bool isALoadableFileFormat(system::IFile* _file, const system::logger_opt_ptr logger) const override;

            virtual const char** getAssociatedFileExtensions() const override
            {
                static const char* ext[]{ "avif", nullptr };
                return ext;
            }

            virtual uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_IMAGE; }

            virtual asset::SAssetBundle loadAsset(system::IFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) override;
        };

    } // end namespace video
} // end namespace nbl


#endif
#endif

