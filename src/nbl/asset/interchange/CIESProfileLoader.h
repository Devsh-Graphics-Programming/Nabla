// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef __NBL_ASSET_C_IES_PROFILE_LOADER_H_INCLUDED__
#define __NBL_ASSET_C_IES_PROFILE_LOADER_H_INCLUDED__

#ifdef _NBL_COMPILE_WITH_IES_LOADER_

#include "nabla.h"
#include "nbl/asset/ICPUImage.h"
#include "nbl/asset/ICPUShader.h"
#include "nbl/video/IGPUShader.h"

#include "nbl/asset/IAssetManager.h"

#include "nbl/asset/interchange/IAssetLoader.h"

#include <sstream>

namespace nbl {
namespace asset {

class CIESProfile {
public:
  enum PhotometricType : uint32_t {
    TYPE_NONE,
    TYPE_C,
    TYPE_B,
    TYPE_A,
  };

  CIESProfile() = default;
  CIESProfile(PhotometricType type, size_t hSize, size_t vSize)
      : type(type), hAngles(hSize), vAngles(vSize), data(hSize * vSize) {}
  ~CIESProfile() = default;
  core::vector<double> &getHoriAngles() { return hAngles; }
  const core::vector<double> &getHoriAngles() const { return hAngles; }
  core::vector<double> &getVertAngles() { return vAngles; }
  const core::vector<double> &getVertAngles() const { return vAngles; }
  void addHoriAngle(double hAngle) {
      hAngles.push_back(hAngle);
      data.resize(getHoriSize() * getVertSize());
  }
  size_t getHoriSize() const { return hAngles.size(); }
  size_t getVertSize() const { return vAngles.size(); }
  void setValue(size_t i, size_t j, double val) {
    data[getVertSize() * i + j] = val;
  }
  double getValue(size_t i, size_t j) const {
    return data[getVertSize() * i + j];
  }
  double sample(double vAngle, double hAngle) const;
  double getMaxValue() const {
    return *std::max_element(std::begin(data), std::end(data));
  }

private:
  PhotometricType type;
  core::vector<double> hAngles;
  core::vector<double> vAngles;
  core::vector<double> data;
};

class CIESProfileParser {
public:
  CIESProfileParser(char *buf, size_t size) { ss << std::string(buf, size); }

  bool parse(CIESProfile& result);

private:
  int getInt(const char *errorMsg) {
    int in;
    if (ss >> in)
      return in;
    error = true;
    if (!this->errorMsg)
      this->errorMsg = errorMsg;
    return 0;
  }

  double getDouble(const char *errorMsg) {
    double in;
    if (ss >> in)
      return in;
    error = true;
    if (!this->errorMsg)
      this->errorMsg = errorMsg;
    return -1.0;
  }

  bool error{false};
  const char *errorMsg{nullptr};
  std::stringstream ss;
};

class CIESProfileMetadata final : public asset::IAssetMetadata {
public:
  CIESProfileMetadata(double maxIntensity)
      : IAssetMetadata(), maxIntensity(maxIntensity) {}

  _NBL_STATIC_INLINE_CONSTEXPR const char *LoaderName = "CIESProfileLoader";
  const char *getLoaderName() const override { return LoaderName; }

  double getMaxIntensity() const { return maxIntensity; }

private:
  double maxIntensity;
};

class CIESProfileLoader final : public asset::IAssetLoader {
public:
  _NBL_STATIC_INLINE_CONSTEXPR size_t TEXTURE_WIDTH = 1024;
  _NBL_STATIC_INLINE_CONSTEXPR size_t TEXTURE_HEIGHT = 2048;
  _NBL_STATIC_INLINE_CONSTEXPR double MAX_VANGLE = 180.0;
  _NBL_STATIC_INLINE_CONSTEXPR double MAX_HANGLE = 360.0;

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
    return asset::IAsset::ET_IMAGE;
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

#endif
