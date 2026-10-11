#include "CHttpDownloader.h"
#include <curl/curl.h>
#include <vector>
#include <cstdio>
#include <cstring>

#include "nbl/system/CFileView.h"
#include "nbl/system/IFileViewAllocator.h"

namespace nbl::system
{
    void CHttpDownloader::CAsyncQueue::init()
    {
        //Cannot log here, sadly
        const CURLcode res = curl_global_init(CURL_GLOBAL_DEFAULT);
        m_curlInitResult = static_cast<int>(res);
        m_curlReady = (res == CURLE_OK);
    }

    void CHttpDownloader::CAsyncQueue::exit()
    {
        if (m_curlReady)
            curl_global_cleanup();
    }

    void CHttpDownloader::CAsyncQueue::process_request(base_t::future_base_t* future, SRequest_DOWNLOAD& request)
    {
        using retval_t = core::smart_refctd_ptr<IFile>;
        auto* storage = base_t::future_storage_cast<retval_t>(future);

        if (!m_curlReady)
        {
            m_logger.log("curl_global_init failed (%s), cannot download %s", ILogger::ELL_ERROR,
                curl_easy_strerror(static_cast<CURLcode>(m_curlInitResult)), request.url.c_str());
            storage->construct(nullptr);
            return;
        }

        std::vector<uint8_t> bytes;
        CURL* curl = curl_easy_init();

        char errorBuffer[CURL_ERROR_SIZE] = {};
        curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, errorBuffer);

        curl_easy_setopt(curl, CURLOPT_URL, request.url.c_str());
        curl_easy_setopt(curl, CURLOPT_PROTOCOLS_STR, "http,https");
        curl_easy_setopt(curl, CURLOPT_REDIR_PROTOCOLS_STR, "http,https");
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_MAXREDIRS, MAX_REDIRECTS);
        curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);

        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, CONNECT_TIMEOUT_SECONDS);
        curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, LOW_SPEED_LIMIT_BYTES_PER_SECOND);
        curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, LOW_SPEED_TIME_SECONDS);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, TOTAL_TIMEOUT_SECONDS);

        curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeToBuffer);  //callback to write data to buffer
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &bytes);

        const CURLcode res = curl_easy_perform(curl);

        long httpStatus = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpStatus);
        curl_easy_cleanup(curl);

        if (res != CURLE_OK)
        {
            m_logger.log("Download of %s failed: %s (curl code %d, HTTP status %ld)", ILogger::ELL_ERROR,
                request.url.c_str(), errorBuffer[0] ? errorBuffer : curl_easy_strerror(res), static_cast<int>(res), httpStatus);
            storage->construct(nullptr);
            return;
        }

        auto file = createFile(path(request.url).filename(), bytes);
        if (file)
        {
            m_logger.log("Downloaded %s (%zu bytes)", ILogger::ELL_DEBUG, request.url.c_str(), bytes.size());
        }
        else
        {
            m_logger.log("Downloaded %s (%zu bytes) but could not create the in-memory file", ILogger::ELL_ERROR, request.url.c_str(), bytes.size());
        }
        storage->construct(std::move(file));
    }

    void CHttpDownloader::downloadFile(future_t<core::smart_refctd_ptr<IFile>>& future, const std::string& url)
    {
        m_queue.request(&future, url);
    }

    size_t CHttpDownloader::writeToBuffer(void* buffer, size_t size, size_t numMembers, void* userData)
    {
        auto* out = static_cast<std::vector<uint8_t>*>(userData);
        const size_t bytes = size * numMembers;
        out->insert(out->end(), static_cast<uint8_t*>(buffer), static_cast<uint8_t*>(buffer) + bytes);

        return bytes;
    }

    core::smart_refctd_ptr<IFile> CHttpDownloader::createFile(path&& filePath, std::vector<uint8_t>& data)
    {
        CPlainHeapAllocator allocator(nullptr);

        void* memoryPtr = allocator.alloc(data.size());
        if (memoryPtr == nullptr)
        {
            return nullptr;
        }

        std::memcpy(memoryPtr, data.data(), data.size());

        return core::make_smart_refctd_ptr<CFileView<CPlainHeapAllocator>>(
            std::move(filePath),
            core::bitflag<IFileBase::E_CREATE_FLAGS>(IFileBase::ECF_READ),
            IFileBase::time_point_t::clock::now(),
            memoryPtr, data.size(), std::move(allocator));
    }
}
