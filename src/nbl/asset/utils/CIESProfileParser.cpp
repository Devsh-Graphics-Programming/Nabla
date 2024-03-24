#include "CIESProfileParser.h"

using namespace nbl;
using namespace asset;

int CIESProfileParser::getInt(const char* errorMsg)
{
    int in;
    if (ss >> in)
        return in;
    error = true;
    if (!this->errorMsg)
        this->errorMsg = errorMsg;
    return 0;
}

double CIESProfileParser::getDouble(const char* errorMsg)
{
    double in;
    if (ss >> in)
        return in;
    error = true;
    if (!this->errorMsg)
        this->errorMsg = errorMsg;
    return -1.0;
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
    auto& vAngles = result.vAngles;
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

    auto& hAngles = result.hAngles;
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
                result.setValue(result.hAngles.size() - 1, j, result.getValue(i, j));
        }
    }

    result.maxValue = *std::max_element(std::begin(result.data), std::end(result.data));

    return !error;
}