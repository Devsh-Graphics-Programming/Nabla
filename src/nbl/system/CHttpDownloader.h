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
    constexpr long LOW_SPEED_LIMIT_BYTES_PER_SECOND = 1024;
    constexpr long LOW_SPEED_TIME_SECONDS = 30;
    constexpr long TOTAL_TIMEOUT_SECONDS = 600;
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
            int m_curlInitResult = 0;
            logger_opt_smart_ptr m_logger = nullptr;

        public:
            CAsyncQueue() : base_t() { start(); }

            inline void setLogger(core::smart_refctd_ptr<ILogger>&& logger) { m_logger = logger_opt_smart_ptr(std::move(logger)); }
            ~CAsyncQueue() { this->shutdown(); }

            void init();
            void exit();

            void process_request(future_base_t* future, SRequest_DOWNLOAD& req);
        } m_queue;

    public:

        CHttpDownloader() : m_queue() {}
        ~CHttpDownloader() = default;

        inline void setLogger(core::smart_refctd_ptr<ILogger>&& logger) { m_queue.setLogger(std::move(logger)); }

        template<typename T>
        using future_t = IAsyncQueueDispatcher<CAsyncQueue, SRequest_DOWNLOAD, CIRCULAR_QUEUE_SIZE>::future_t<T>;

        void downloadFile(future_t<core::smart_refctd_ptr<IFile>>& future, const std::string& url);

    private:
        static size_t writeToBuffer(void* buffer, size_t size, size_t numMembers, void* userData);
        static core::smart_refctd_ptr<IFile> createFile(path&& filePath, std::vector<uint8_t>& data);

    };
}
