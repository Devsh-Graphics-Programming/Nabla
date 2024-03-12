#include "nbl/asset/interchange/CIESProfileLoader.h"

using namespace nbl;
using namespace asset;


namespace nbl {
namespace asset {

class CIESProfile {
public:
    _NBL_STATIC_INLINE_CONSTEXPR double MAX_VANGLE = 180.0;
    _NBL_STATIC_INLINE_CONSTEXPR double MAX_HANGLE = 360.0;

    enum PhotometricType : uint32_t {
        TYPE_NONE,
        TYPE_C,
        TYPE_B,
        TYPE_A,
    };

    CIESProfile() = default;
    CIESProfile(PhotometricType type, size_t hSize, size_t vSize)
        : type(type), hAngles(hSize), vAngles(vSize), data(hSize* vSize) {}
    ~CIESProfile() = default;
    core::vector<double>& getHoriAngles() { return hAngles; }
    const core::vector<double>& getHoriAngles() const { return hAngles; }
    core::vector<double>& getVertAngles() { return vAngles; }
    const core::vector<double>& getVertAngles() const { return vAngles; }
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
    double getIntegral() const;
private:
    PhotometricType type;
    core::vector<double> hAngles;
    core::vector<double> vAngles;
    core::vector<double> data;
};

}
}

class CIESProfileParser {
public:
    CIESProfileParser(char* buf, size_t size) { ss << std::string(buf, size); }

    const char* getErrorMsg() const {
        return errorMsg;
    }
    bool parse(CIESProfile& result);

private:
    int getInt(const char* errorMsg) {
        int in;
        if (ss >> in)
            return in;
        error = true;
        if (!this->errorMsg)
            this->errorMsg = errorMsg;
        return 0;
    }

    double getDouble(const char* errorMsg) {
        double in;
        if (ss >> in)
            return in;
        error = true;
        if (!this->errorMsg)
            this->errorMsg = errorMsg;
        return -1.0;
    }

    bool error{ false };
    const char* errorMsg{ nullptr };
    std::stringstream ss;
};

double CIESProfile::sample(double vAngle, double hAngle) const {
    if (vAngle > vAngles.back()) return 0.0;
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

double CIESProfile::getIntegral() const {
    size_t numHSamples = 2 * getHoriSize();
    size_t numVSamples = 2 * getVertSize();
    double dTheta = core::PI<double>() / numVSamples;
    double dPhi = 2 * core::PI<double>() / numHSamples;
    double dThetaInAngle = MAX_VANGLE / numVSamples;
    double dPhiInAngle = MAX_HANGLE / numHSamples;
    double res = 0;
    for (size_t i = 0; i < numVSamples; i++) {
        for (size_t j = 0; j < numHSamples; j++) {
            res += dPhi * dTheta * sin(dTheta * i) * sample(i* dThetaInAngle,j*dPhiInAngle);
        }
    }
    return res;
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
     if (type != CIESProfile::PhotometricType::TYPE_C) {
         errorMsg = "Only type C is supported for now";
         return false;
     }

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
     }
     if (!std::is_sorted(vAngles.begin(), vAngles.end())) {
         errorMsg = "Angles should be sorted";
         return false;
     }
     if (vAngles[0] != 0.0 && vAngles[0] != 90.0) {
         errorMsg = "First angle must be 0 or 90 in type C";
         return false;
     }
     if (vAngles[vSize - 1] != 90.0 && vAngles[vSize - 1] != 180.0) {
         errorMsg = "Last angle must be 90 or 180 in type C";
         return false;
     }

     auto& hAngles = result.getHoriAngles();
     for (int i = 0; i < hSize; i++) {
         hAngles[i] = getDouble("horizontal angle truncated");
         if (i != 0 && hAngles[i - 1] > hAngles[i])
             return false; // Angles should be sorted
     }

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
    if (!parser.parse(profile)) {
        os::Printer::log("ERROR: Emission profile parsing error: " + std::string(parser.getErrorMsg()), ELL_ERROR);
        return {};
    }

    auto image = createTexture(profile, TEXTURE_WIDTH, TEXTURE_HEIGHT);
    if (!image)
        return {};

    auto meta =
        core::make_smart_refctd_ptr<CIESProfileMetadata>(profile.getMaxValue(), profile.getIntegral());

    ICPUImageView::SCreationParams viewParams = {};
    viewParams.image = image;
    viewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0);
    viewParams.viewType = IImageView<ICPUImage>::ET_2D;
    viewParams.format = viewParams.image->getCreationParameters().format;
    viewParams.subresourceRange.aspectMask = static_cast<IImage::E_ASPECT_FLAGS>(0);
    viewParams.subresourceRange.levelCount = viewParams.image->getCreationParameters().mipLevels;
    viewParams.subresourceRange.layerCount = 1u;

    return asset::SAssetBundle (std::move(meta), {ICPUImageView::create(std::move(viewParams))});
}

static inline core::vectorSIMDf octahdronUVToDir(float u, float v) {
    core::vectorSIMDf pos = core::vectorSIMDf(2 * (u - 0.5), 2 * (v - 0.5), 0.0);
    float abs_x = core::abs(pos.x), abs_y = core::abs(pos.y);
    pos.z = 1.0 - abs_x - abs_y;
    if (pos.z < 0.0) {
        pos.x = core::sign(pos.x) * (1.0 - abs_y);
        pos.y = core::sign(pos.y) * (1.0 - abs_x);
    }

    return core::normalize(pos);
}

static inline std::pair<float,float> sphericalDirToAngles(core::vectorSIMDf dir) {
    float theta = std::acos(dir.z);
    float phi = std::atan2(dir.y, dir.x);
    return { theta, phi < 0 ? phi + 2 * core::PI<float>() : phi };
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

    double vertInv = 1.0 / height;
    double horiInv = 1.0 / width;
    char* bufferPtr = reinterpret_cast<char*>(buffer->getPointer());
    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            auto dir = octahdronUVToDir(j * vertInv, i * horiInv);
            auto [theta, phi] = sphericalDirToAngles(dir);
            double I = profile.sample(core::degrees(theta), core::degrees(phi));
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
