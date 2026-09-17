#ifndef _NBL_EXT_CAMERAS_C_FILE_UTILITIES_HPP_
#define _NBL_EXT_CAMERAS_C_FILE_UTILITIES_HPP_

#include <string>
#include <string_view>

#include "nbl/asset/ICPUBuffer.h"
#include "nbl/system/IFile.h"
#include "nbl/system/ISystem.h"

namespace nbl::ext::cameras
{

/// @brief Whole-file read/write helpers shared by the camera persistence and scripted-runtime loaders.
///
/// Nothing here is camera specific; the helpers only give the persistence code one consistent way to
/// open a file through `ISystem` and to report why that failed.
struct CFileUtilities final
{
public:
    /// @brief Read a whole file into a new CPU buffer, returns `nullptr` when the file cannot be opened or read.
    static inline core::smart_refctd_ptr<asset::ICPUBuffer> readBinaryFile(
        system::ISystem& system,
        const system::path& filePath,
        std::string* error = nullptr,
        const std::string_view openError = {})
    {
        system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> future;
        system.createFile(future, filePath, system::IFile::ECF_READ | system::IFile::ECF_MAPPABLE);
        auto file = future.acquire();
        if (!file || !file->get())
        {
            if (error && !openError.empty())
                *error = std::string(openError);
            return nullptr;
        }

        auto& input = *file->get();
        const auto fileSize = input.getSize();

        asset::ICPUBuffer::SCreationParams params = {};
        params.size = fileSize;
        auto buffer = asset::ICPUBuffer::create(std::move(params));
        if (!buffer)
        {
            if (error && !openError.empty())
                *error = std::string(openError);
            return nullptr;
        }
        if (fileSize == 0ull)
            return buffer;

        system::IFile::success_t readResult;
        input.read(readResult, buffer->getPointer(), 0, fileSize);
        if (!static_cast<bool>(readResult))
        {
            if (error && !openError.empty())
                *error = std::string(openError);
            return nullptr;
        }
        return buffer;
    }

    /// @brief Read a whole file and interpret its payload as UTF-8 text.
    static inline bool readTextFile(
        system::ISystem& system,
        const system::path& filePath,
        std::string& outText,
        std::string* error = nullptr,
        const std::string_view openError = {})
    {
        const auto payload = readBinaryFile(system, filePath, error, openError);
        if (!payload)
            return false;

        outText.assign(reinterpret_cast<const char*>(payload->getPointer()), payload->getSize());
        return true;
    }

    /// @brief Overwrite a file with the provided text payload.
    static inline bool writeTextFile(
        system::ISystem& system,
        const system::path& filePath,
        const std::string_view text)
    {
        system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> future;
        system.createFile(future, filePath, system::IFile::ECF_WRITE);
        auto file = future.acquire();
        if (!file || !file->get())
            return false;
        if (text.empty())
            return true;

        system::IFile::success_t writeResult;
        (*file)->write(writeResult, text.data(), 0, text.size());
        return static_cast<bool>(writeResult);
    }
};

} // namespace nbl::ext::cameras

#endif // _NBL_EXT_CAMERAS_C_FILE_UTILITIES_HPP_
