from DeviceGen import *

if __name__ == "__main__":
    limits = loadJSON(buildPath(os.path.join('..', '..', 'src', 'nbl', 'video', 'device_limits.json')))
    writeDeviceHeader(
        buildPath(os.path.join('test_device_limits.h')),
        limits
    )
    features = loadJSON(buildPath(os.path.join('..', '..', 'src', 'nbl', 'video', 'device_features.json')))
    writeDeviceHeader(
        buildPath(os.path.join('test_device_features.h')),
        features
    )