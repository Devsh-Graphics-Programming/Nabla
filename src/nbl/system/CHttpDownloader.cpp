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
        curl_global_init(CURL_GLOBAL_DEFAULT);
    }

    void CHttpDownloader::CAsyncQueue::exit()
    {
        curl_global_cleanup();
    }

    void CHttpDownloader::CAsyncQueue::process_request(base_t::future_base_t* future, SRequest_DOWNLOAD& request)
    {
        using retval_t = core::smart_refctd_ptr<IFile>;
        auto* storage = base_t::future_storage_cast<retval_t>(future);

        std::vector<uint8_t> bytes;
        CURL* curl = curl_easy_init();

        if (curl == nullptr)
        {
            storage->construct(nullptr);
            return;
        }

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

        const bool ok = curl_easy_perform(curl) == CURLE_OK;
        curl_easy_cleanup(curl);

        storage->construct(ok ? createFile(path(request.url).filename(), bytes) : nullptr);
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
