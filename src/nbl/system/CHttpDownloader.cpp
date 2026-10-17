#include "CHttpDownloader.h"
#include <curl/curl.h>
#include <vector>
#include <cstring>
#include <algorithm>

#include "nbl/system/CFileView.h"
#include "nbl/system/IFileViewAllocator.h"

namespace nbl::system
{
    CHttpDownloader::CHttpDownloader(core::smart_refctd_ptr<ILogger>&& logger) : curlInit(false), stillRunning(0), m_quit(false), m_logger(std::move(logger))
    {
        initCURLThread();
    }

    CHttpDownloader::~CHttpDownloader()
    {
        exit();
    }

    void CHttpDownloader::initCURLThread()
    {
        if (m_downloadThread.joinable())
            return;

        const CURLcode result = curl_global_init(CURL_GLOBAL_DEFAULT);
        if (result != CURLE_OK)
        {
            m_logger.log("curl_global_init failed: %s", ILogger::ELL_ERROR, curl_easy_strerror(result));
            return;
        }
        m_logger.log("CURL global init succeeded");
        curlInit = true;

        m_multipleHandles = CurlMultiPtr(curl_multi_init());
        if (!m_multipleHandles)
        {
            m_logger.log("curl_multi_init failed", ILogger::ELL_ERROR);
            return;
        }
        m_logger.log("CURL multiple handles initialized");

        stillRunning = 0;
        m_quit = false;

        m_downloadThread = std::thread([this]()
        {
            processRequests();
        });
    }

    void CHttpDownloader::exit()
    {
        m_quit = true;

        if (m_downloadThread.joinable())
        {
            curl_multi_wakeup(m_multipleHandles.get());
            m_downloadThread.join();
        }

        currentTransfers.clear();
        m_multipleHandles.reset();

        if (curlInit)
        {
            curl_global_cleanup();
            curlInit = false;
            m_logger.log("Tearing down CURL");
        }
    }

    void CHttpDownloader::downloadFile(const std::string& url, const std::function<void(bool success, const std::vector<uint8_t>& data)>& callback)
    {
        {
            std::lock_guard<std::mutex> lock(mutexDownloadQueue);
            requestQueue.push(DownloadRequest{ .url = url, .callback = callback });
        }

        if (m_multipleHandles)
            curl_multi_wakeup(m_multipleHandles.get());
    }

    void CHttpDownloader::update()
    {
        std::queue<Completion> done;
        {
            std::lock_guard<std::mutex> lock(mutexCompletionQueue);
            std::swap(done, completionQueue);
        }

        while (!done.empty())
        {
            Completion& completion = done.front();
            if (completion.callback)
                completion.callback(completion.success, completion.bytes);
            done.pop();
        }
    }

    void CHttpDownloader::pushCompletion(Completion&& completion)
    {
        std::lock_guard<std::mutex> lock(mutexCompletionQueue);
        completionQueue.push(std::move(completion));
    }

    void CHttpDownloader::startTransfer(DownloadRequest&& request)
    {
        auto transfer = std::make_unique<Transfer>();

        transfer->url = std::move(request.url);
        transfer->callback = std::move(request.callback);
        transfer->handle = CurlHandle(curl_easy_init());

        CURL* handle = transfer->handle.get();
        if (!handle)
        {
            m_logger.log("curl_easy_init failed for %s", ILogger::ELL_ERROR, transfer->url.c_str());
            pushCompletion({ .success = false, .bytes = {}, .callback = std::move(transfer->callback) });
            return;
        }

        curl_easy_setopt(handle, CURLOPT_ERRORBUFFER, transfer->errorBuffer);

        curl_easy_setopt(handle, CURLOPT_URL, transfer->url.c_str());
        curl_easy_setopt(handle, CURLOPT_PROTOCOLS_STR, "http,https");
        curl_easy_setopt(handle, CURLOPT_REDIR_PROTOCOLS_STR, "http,https");
        curl_easy_setopt(handle, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(handle, CURLOPT_MAXREDIRS, CURL_MAX_REDIRECTS);
        curl_easy_setopt(handle, CURLOPT_FAILONERROR, 1L);

        curl_easy_setopt(handle, CURLOPT_CONNECTTIMEOUT, CURL_CONNECT_TIMEOUT_SECONDS);
        curl_easy_setopt(handle, CURLOPT_LOW_SPEED_LIMIT, CURL_LOW_SPEED_LIMIT_BYTES_PER_SECOND);
        curl_easy_setopt(handle, CURLOPT_LOW_SPEED_TIME, CURL_LOW_SPEED_TIME_SECONDS);
        curl_easy_setopt(handle, CURLOPT_TIMEOUT, CURL_TOTAL_TIMEOUT_SECONDS);

        curl_easy_setopt(handle, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, writeToBuffer);  //callback to write data to buffer
        curl_easy_setopt(handle, CURLOPT_WRITEDATA, &transfer->bytes);

        //Lets us find the Transfer again when curl reports this handle as done
        curl_easy_setopt(handle, CURLOPT_PRIVATE, transfer.get());

        const CURLMcode added = curl_multi_add_handle(m_multipleHandles.get(), handle);
        if (added != CURLM_OK)
        {
            m_logger.log("Could not start download of %s: %s", ILogger::ELL_ERROR, transfer->url.c_str(), curl_multi_strerror(added));
            pushCompletion({ .success = false, .bytes = {}, .callback = std::move(transfer->callback) });
            return;
        }

        currentTransfers.push_back(std::move(transfer));
    }

    void CHttpDownloader::finishTransfer(CURLMsg* msg)
    {
        CURL* easy = msg->easy_handle;
        const CURLcode result = msg->data.result;

        char* privateData = nullptr;
        curl_easy_getinfo(easy, CURLINFO_PRIVATE, &privateData);
        Transfer* finished = reinterpret_cast<Transfer*>(privateData);

        long httpStatus = 0;
        curl_easy_getinfo(easy, CURLINFO_RESPONSE_CODE, &httpStatus);

        curl_multi_remove_handle(m_multipleHandles.get(), easy);

        if (!finished)
            return;

        const bool success = (result == CURLE_OK);
        if (success)
        {
            m_logger.log("Downloaded %s (%zu bytes)", ILogger::ELL_DEBUG, finished->url.c_str(), finished->bytes.size());
        }
        else
        {
            m_logger.log("Download of %s failed: %s (curl code %d, HTTP status %ld)", ILogger::ELL_ERROR,
                finished->url.c_str(), finished->errorBuffer[0] ? finished->errorBuffer : curl_easy_strerror(result), static_cast<int>(result), httpStatus);
        }

        pushCompletion({ .success = success, .bytes = std::move(finished->bytes), .callback = std::move(finished->callback) });

        //Destroys the Transfer, which also cleans up the easy handle (already removed from the multi handle)
        std::erase_if(currentTransfers, [finished](const std::unique_ptr<Transfer>& transfer) { return transfer.get() == finished; });
    }

    void CHttpDownloader::processRequests()
    {
        CURLM* multi = m_multipleHandles.get();

        while (!m_quit)
        {
            //Start queued requests, up to the simultaneous download limit
            //If more requests come in, they are queued
            while (currentTransfers.size() < CURL_MAX_SIMULTANEOUS_DOWNLOADS)
            {
                DownloadRequest request;
                {
                    std::lock_guard<std::mutex> lock(mutexDownloadQueue);
                    if (requestQueue.empty())
                        break;

                    request = std::move(requestQueue.front());
                    requestQueue.pop();
                }
                startTransfer(std::move(request));
            }

            curl_multi_perform(multi, &stillRunning);

            bool finishedAny = false;
            int msgsLeft = 0;
            while (CURLMsg* msg = curl_multi_info_read(multi, &msgsLeft))
            {
                if (msg->msg == CURLMSG_DONE)
                {
                    finishTransfer(msg);
                    finishedAny = true;
                }
            }

            curl_multi_poll(multi, nullptr, 0, finishedAny ? 0 : 1000, nullptr);
        }

        for (auto& transfer : currentTransfers)
            curl_multi_remove_handle(multi, transfer->handle.get());

        currentTransfers.clear();
    }

    size_t CHttpDownloader::writeToBuffer(void* buffer, size_t size, size_t numMembers, void* userData)
    {
        auto* out = static_cast<std::vector<uint8_t>*>(userData);
        const size_t bytes = size * numMembers;

        //Returning less than what was given makes curl abort the transfer with CURLE_WRITE_ERROR
        if (out->size() + bytes > CURL_MAX_DOWNLOAD_BYTES)
            return 0;

        out->insert(out->end(), static_cast<uint8_t*>(buffer), static_cast<uint8_t*>(buffer) + bytes);

        return bytes;
    }

    core::smart_refctd_ptr<IFile> CHttpDownloader::createFile(path&& filePath, const std::vector<uint8_t>& data)
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
