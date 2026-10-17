#pragma once

#include <atomic>
#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

#include "curl/multi.h"
#include "nbl/system/IAsyncQueueDispatcher.h"
#include "nbl/system/IFile.h"
#include "nbl/system/ILogger.h"

namespace nbl::system
{
    constexpr long CURL_CONNECT_TIMEOUT_SECONDS = 10;
    constexpr long CURL_LOW_SPEED_LIMIT_BYTES_PER_SECOND = 1024;
    constexpr long CURL_LOW_SPEED_TIME_SECONDS = 30;
    constexpr long CURL_TOTAL_TIMEOUT_SECONDS = 600;
    constexpr long CURL_MAX_REDIRECTS = 5;
    constexpr size_t CURL_MAX_DOWNLOAD_BYTES = 256ull * 1024ull * 1024ull;
    constexpr uint8_t CURL_MAX_SIMULTANEOUS_DOWNLOADS = 5;

    constexpr uint8_t NUM_THREADs = 5;

    struct DownloadRequest
    {
        std::string url;
        std::function<void(bool success, const std::vector<uint8_t>& data)> callback;
    };

    //Using deleters since I wrap the curl handles in std::unique_ptrs
    struct CurlMultipleHandleDeleter
    {
        void operator()(CURLM* handle) const
        {
            if (handle)
                curl_multi_cleanup(handle);
        }
    };

    struct CurlEasyHandleDeleter
    {
        void operator()(CURL* handle) const
        {
            if (handle)
                curl_easy_cleanup(handle);
        }
    };

    using CurlMultiPtr = std::unique_ptr<CURLM, CurlMultipleHandleDeleter>;
    using CurlHandle = std::unique_ptr<CURL, CurlEasyHandleDeleter>;

    //Used to store completion data. Owns everything by value so it can be handed to the main thread.
    struct Completion
    {
        bool success = false;
        std::vector<uint8_t> bytes;

        std::function<void(bool success, const std::vector<uint8_t>& data)> callback;
    };

    struct Transfer
    {
        //Must be declared before the handle, so it outlives it (curl keeps a pointer to it)
        char errorBuffer[CURL_ERROR_SIZE] = {};

        std::string url;
        CurlHandle handle;
        std::vector<uint8_t> bytes{};

        std::function<void(bool, const std::vector<uint8_t>&)> callback;
    };

    class NBL_API2 CHttpDownloader
    {

    public:
        CHttpDownloader(core::smart_refctd_ptr<ILogger>&& logger);

        ~CHttpDownloader();
        bool initCURL();

        //Adds a new download to the list.
        void downloadFile(const std::string& url, const std::function<void(bool success, const std::vector<uint8_t>& data)>& callback);

        //Call from the main thread (e.g. once per frame). Runs the callbacks of finished downloads.
        void update();

        //Wraps downloaded bytes in an in-memory read-only file
        static core::smart_refctd_ptr<IFile> createFile(path&& filePath, const std::vector<uint8_t>& data);

    private:
        void initCURLWithProcessingThread();

        void exit();

        void processRequests();
        void startTransfer(DownloadRequest&& request);
        void finishTransfer(CURLMsg* msg);
        void pushCompletion(Completion&& completion);

        static size_t writeToBuffer(void* buffer, size_t size, size_t numMembers, void* userData);

        bool curlInit;
        int stillRunning;
        std::atomic<bool> m_quit;

        //stores requests from the main thread
        std::queue<DownloadRequest> requestQueue;

        //stores finished downloads for the main thread
        std::queue<Completion> completionQueue;

        //Stores current active transfers (download thread only)
        std::vector<std::unique_ptr<Transfer>> currentTransfers;

        logger_opt_smart_ptr m_logger;

        std::mutex mutexDownloadQueue;
        std::mutex mutexCompletionQueue;

        CurlMultiPtr m_multipleHandles;

        std::thread m_downloadThread;
    };
}
