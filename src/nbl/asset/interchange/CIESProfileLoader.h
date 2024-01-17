// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef __NBL_ASSET_C_IES_PROFILE_LOADER_H_INCLUDED__
#define __NBL_ASSET_C_IES_PROFILE_LOADER_H_INCLUDED__

#include "nbl/asset/ICPUImage.h"
#include "nbl/asset/ICPUShader.h"

#include "nbl/asset/IAssetManager.h"

#include "nbl/asset/interchange/IAssetLoader.h"

#include <sstream>

namespace nbl {
namespace asset {

class CIESProfileMetadata final : public asset::IAssetMetadata {
public:
    CIESProfileMetadata(double maxIntensity, double integral)
        : IAssetMetadata(), maxIntensity(maxIntensity), integral(integral) {}

    CIESProfileMetadata(const CIESProfileMetadata& other) : CIESProfileMetadata(other.maxIntensity, other.integral) {}

    _NBL_STATIC_INLINE_CONSTEXPR const char* LoaderName = "CIESProfileLoader";
    const char* getLoaderName() const override { return LoaderName; }
    
    double getMaxIntensity() const { return maxIntensity; }
    double getIntegral() const { return integral; }

private:
    double maxIntensity;
    double integral;
};

class CIESProfile;

class CIESProfileLoader final : public asset::IAssetLoader {
public:
  _NBL_STATIC_INLINE_CONSTEXPR size_t TEXTURE_WIDTH = 1024;
  _NBL_STATIC_INLINE_CONSTEXPR size_t TEXTURE_HEIGHT = 1024;

  //! Check if the file might be loaded by this class
  /** Check might look into the file.
  \param file File handle to check.
  \return True if file seems to be loadable. */
  bool isALoadableFileFormat(io::IReadFile *_file) const override {
    const size_t begginingOfFile = _file->getPos();
    _file->seek(0ull);
    std::string versionBuffer(5, ' ');
    _file->read(versionBuffer.data(), versionBuffer.size());
    _file->seek(begginingOfFile);
    return versionBuffer == "IESNA";
  }

  //! Returns an array of string literals terminated by nullptr
  const char **getAssociatedFileExtensions() const override {
    static const char *extensions[]{"ies", nullptr};
    return extensions;
  }

  //! Returns the assets loaded by the loader
  /** Bits of the returned value correspond to each IAsset::E_TYPE
  enumeration member, and the return value cannot be 0. */
  uint64_t getSupportedAssetTypesBitfield() const override {
    return asset::IAsset::ET_IMAGE_VIEW;
  }

  //! Loads an asset from an opened file, returns nullptr in case of failure.
  asset::SAssetBundle
      loadAsset(io::IReadFile* _file,
          const asset::IAssetLoader::SAssetLoadParams& _params,
          asset::IAssetLoader::IAssetLoaderOverride* _override = nullptr,
          uint32_t _hierarchyLevel = 0u) override;

private:
    core::smart_refctd_ptr<asset::ICPUImage>
        createTexture(const CIESProfile& profile, size_t width, size_t height);
};

} // namespace asset
} // namespace nbl

#endif
