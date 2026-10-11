#pragma once

#include <string>
#include <vector>
#include <functional>

#include "nbl/system/IAsyncQueueDispatcher.h"
#include "nbl/system/IFile.h"
#include "nbl/system/ILogger.h"

namespace nbl::system
{
    constexpr long CONNECT_TIMEOUT_SECONDS = 10;
    constexpr long LOW_SPEED_LIMIT_BYTES_PER_SECOND = 1024;   // abort if slower than this
    constexpr long LOW_SPEED_TIME_SECONDS = 30;               // for this long should the download try to run
    constexpr long TOTAL_TIMEOUT_SECONDS = 600;               // hard backstop for the whole transfer
    constexpr long MAX_REDIRECTS = 5;
    constexpr size_t MAX_DOWNLOAD_BYTES = 256ull * 1024ull * 1024ull;

    constexpr int CIRCULAR_QUEUE_SIZE = 256;

    class NBL_API2 CHttpDownloader
    {
    private:
        struct SRequest_DOWNLOAD
        {
            std::string url = {};
            std::vector<uint8_t> data = {};
            std::function<void(bool success, const std::vector<uint8_t>& data)> buffer_func;
        };

        class NBL_API2 CAsyncQueue final : public IAsyncQueueDispatcher<CAsyncQueue, SRequest_DOWNLOAD, CIRCULAR_QUEUE_SIZE>
        {
            using base_t = IAsyncQueueDispatcher;

            bool m_curlReady = false;
            int m_curlInitResult = 0;           // a CURLcode, kept so `process_request` can report why init failed
            logger_opt_smart_ptr m_logger = nullptr;      // only read by the worker thread inside `process_request`

        public:
            CAsyncQueue() : base_t() { start(); }

            // Must be called before the first request is queued, the queue is what orders it before the worker's reads
            inline void setLogger(core::smart_refctd_ptr<ILogger>&& logger) { m_logger = logger_opt_smart_ptr(std::move(logger)); }
            ~CAsyncQueue() { this->shutdown(); }

            void init();
            void exit();

            void process_request(future_base_t* future, SRequest_DOWNLOAD& req);
        } m_queue;

    public:

        CHttpDownloader() : m_queue() {}
        ~CHttpDownloader() = default;

        // Optional, errors get logged from the download worker thread. Call before the first `downloadFile`.
        inline void setLogger(core::smart_refctd_ptr<ILogger>&& logger) { m_queue.setLogger(std::move(logger)); }

        template<typename T>
        using future_t = IAsyncQueueDispatcher<CAsyncQueue, SRequest_DOWNLOAD, CIRCULAR_QUEUE_SIZE>::future_t<T>;

        void downloadFile(future_t<core::smart_refctd_ptr<IFile>>& future, const std::string& url);

    private:
        static size_t writeToBuffer(void* buffer, size_t size, size_t numMembers, void* userData);
        static core::smart_refctd_ptr<IFile> createFile(path&& filePath, std::vector<uint8_t>& data);

    };
}
