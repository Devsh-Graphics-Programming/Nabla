#include "nbl/asset/interchange/CIESProfileLoader.h"

using namespace nbl;
using namespace asset;


double CIESProfile::sample(double vAngle, double hAngle) const {
    vAngle = fmod(vAngle, vAngles.back()); // warp angle
    hAngle = hAngles.back() == 0.0
                 ? 0.0
                 : fmod(hAngle,
                        hAngles.back()); // when last horizontal angle is zero
                                         // it's symmetric across all planes

    // bilinear interpolation
    auto lb = [](const core::vector<double> &angles, double angle) -> int {
      int idx = std::upper_bound(std::begin(angles), std::end(angles), angle) -
                std::begin(angles);
      return std::max(idx-1, 0);
    };
    auto ub = [](const core::vector<double> &angles, double angle) -> int {
      int idx = std::upper_bound(std::begin(angles), std::end(angles), angle) -
                std::begin(angles);
      return std::min(idx, (int)angles.size() - 1);
    };
    int j0 = lb(vAngles, vAngle);
    int j1 = ub(vAngles, vAngle);
    int i0 = lb(hAngles, hAngle);
    int i1 = ub(hAngles, hAngle);
    double uResp = i1 == i0 ? 1.0 : 1.0 / (hAngles[i1] - hAngles[i0]);
    double vResp = j1 == j0 ? 1.0 : 1.0 / (vAngles[j1] - vAngles[j0]);
    double u = (hAngle - hAngles[i0]) * uResp;
    double v = (vAngle - vAngles[j0]) * vResp;
    double s0 = getValue(i0, j0) * (1.0-v) + getValue(i0, j1) * (v);
    double s1 = getValue(i1, j0) * (1.0-v) + getValue(i1, j1) * (v);
    return s0 * (1.0 - u) + s1 * u;
}


 bool CIESProfileParser::parse(CIESProfile& result) {
     // skip metadata
     std::string line;

     while (std::getline(ss, line)) {
         if (line.back() == '\r')
             line.pop_back();
         if (line == "TILT=INCLUDE" || line == "TILT=NONE")
             break;
     }

     if (line == "TILT=INCLUDE") {
         double lampToLuminaire = getDouble("lampToLuminaire truncated");
         int numTilt = getDouble("numTilt truncated");
         for (int i = 0; i < numTilt; i++)
             getDouble("tilt angle truncated");
         for (int i = 0; i < numTilt; i++)
             getDouble("tilt multiplying factor truncated");
     }
     else if (line != "TILT=NONE") {
         errorMsg = "TILT not specified";
         return false;
     }

     int numLamps = getInt("numLamps truncated");
     double lumensPerLamp = getDouble("lumensPerLamp truncated");
     double candelaMultiplier = getDouble("candelaMultiplier truncated");
     int vSize = getInt("vSize truncated");
     int hSize = getInt("hSize truncated");

     int type_ = getInt("type truncated");
     if (error)
         return false;
     if (type_ <= 0 || type_ > 3) {
         errorMsg = "unrecognized type";
         return false;
     }
     CIESProfile::PhotometricType type =
         static_cast<CIESProfile::PhotometricType>(type_);
     assert(type == CIESProfile::PhotometricType::TYPE_C &&
         "Only type C is supported for now");

     int unitsType = getInt("unitsType truncated");
     double width = getDouble("width truncated"),
         length = getDouble("length truncated"),
         height = getDouble("height truncated");
     double ballastFactor = getDouble("ballastFactor truncated");
     double reserved = getDouble("reserved truncated");
     double inputWatts = getDouble("inputWatts truncated");
     if (error)
         return false;

     result = CIESProfile(type, hSize, vSize);
     auto& vAngles = result.getVertAngles();
     for (int i = 0; i < vSize; i++) {
         vAngles[i] = getDouble("vertical angle truncated");
         if (i != 0 && vAngles[i - 1] > vAngles[i])
             return false; // Angles should be sorted
     }
     assert((vAngles[0] == 0.0 || vAngles[0] == 90.0) &&
         "First angle must be 0 or 90 in type C");
     assert((vAngles[vSize - 1] == 90.0 || vAngles[vSize - 1] == 180.0) &&
         "Last angle must be 90 or 180 in type C");

     auto& hAngles = result.getHoriAngles();
     for (int i = 0; i < hSize; i++) {
         hAngles[i] = getDouble("horizontal angle truncated");
         if (i != 0 && hAngles[i - 1] > hAngles[i])
             return false; // Angles should be sorted
     }
     assert((hAngles[0] == 0.0) && "First angle must be 0 in type C");
     assert((hAngles[hSize - 1] == 0.0 || hAngles[hSize - 1] == 90.0 ||
         hAngles[hSize - 1] == 180.0 || hAngles[hSize - 1] == 360.0) &&
         "Last angle must be 0, 90, 180 or 360 in type C");

     double factor = ballastFactor * candelaMultiplier;
     for (int i = 0; i < hSize; i++) {
         for (int j = 0; j < vSize; j++) {
             result.setValue(i, j, factor * getDouble("intensity value truncated"));
         }
     }

     if (hAngles.back() == 180.0) {
         for (int i = (int)hAngles.size() - 2; i >= 0; i--) {
             result.addHoriAngle(360.0 - hAngles[i]);
             for (int j = 0; j < vSize; j++)
                 result.setValue(result.getHoriSize() - 1, j, result.getValue(i, j));
         }
     }

     return !error;
 }


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


asset::SAssetBundle
CIESProfileLoader::loadAsset(io::IReadFile* _file,
    const asset::IAssetLoader::SAssetLoadParams& _params,
    asset::IAssetLoader::IAssetLoaderOverride* _override,
    uint32_t _hierarchyLevel) {
    if (!_file)
        return {};

    core::vector<char> data(_file->getSize());
    _file->read(data.data(), _file->getSize());

    CIESProfileParser parser(data.data(), data.size());

    CIESProfile profile;
    if (!parser.parse(profile))
        return {};

    auto image = createTexture(profile, TEXTURE_WIDTH, TEXTURE_HEIGHT);
    if (!image)
        return {};

    auto meta =
        core::make_smart_refctd_ptr<CIESProfileMetadata>(profile.getMaxValue());
    return asset::SAssetBundle(std::move(meta), { std::move(image) });
}

core::smart_refctd_ptr<asset::ICPUImage>
CIESProfileLoader::createTexture(const CIESProfile& profile, size_t width, size_t height) {
    asset::ICPUImage::SCreationParams imgInfo;
    imgInfo.type = asset::ICPUImage::ET_2D;
    imgInfo.extent.width = width;
    imgInfo.extent.height = height;
    imgInfo.extent.depth = 1u;
    imgInfo.mipLevels = 1u;
    imgInfo.arrayLayers = 1u;
    imgInfo.samples = asset::ICPUImage::ESCF_1_BIT;
    imgInfo.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);
    imgInfo.format = asset::EF_R16_UNORM;
    auto outImg = asset::ICPUImage::create(std::move(imgInfo));

    asset::ICPUImage::SBufferCopy region;
    size_t texelBytesz = asset::getTexelOrBlockBytesize(imgInfo.format);
    size_t bufferRowLength =
        asset::IImageAssetHandlerBase::calcPitchInBlocks(width, texelBytesz);
    region.bufferRowLength = bufferRowLength;
    region.imageExtent = imgInfo.extent;
    region.imageSubresource.baseArrayLayer = 0u;
    region.imageSubresource.layerCount = 1u;
    region.imageSubresource.mipLevel = 0u;
    region.bufferImageHeight = 0u;
    region.bufferOffset = 0u;
    auto buffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(
        texelBytesz * bufferRowLength * height);

    double maxValue = profile.getMaxValue();
    double maxValueRecip = 1.0 / maxValue;

    double vertAngleRate = MAX_VANGLE / height;
    double horiAngleRate = MAX_HANGLE / width;
    char* bufferPtr = reinterpret_cast<char*>(buffer->getPointer());
    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            double I = profile.sample(i * vertAngleRate, j * horiAngleRate);
            uint16_t value = static_cast<uint16_t>(
                std::clamp(I * maxValueRecip * 65535.0, 0.0, 65535.0));
            *reinterpret_cast<uint16_t*>(
                bufferPtr + i * bufferRowLength * texelBytesz + j * texelBytesz) =
                value;
        }
    }

    outImg->setBufferAndRegions(
        std::move(buffer),
        core::make_refctd_dynamic_array<
        core::smart_refctd_dynamic_array<asset::IImage::SBufferCopy>>(
            1ull, region));
    return outImg;
}
