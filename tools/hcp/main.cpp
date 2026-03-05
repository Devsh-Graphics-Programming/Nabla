// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nabla.h"
#include "nbl/system/IApplicationFramework.h"
#include "nbl/system/CStdoutLogger.h"

#include "nbl/asset/interchange/SFileIOPolicy.h"
#include "nbl/asset/interchange/SGeometryContentHashCommon.h"
#include "nbl/core/hash/blake.h"
#include "argparse/argparse.hpp"

#include <chrono>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

using namespace nbl;
using namespace nbl::asset;
using namespace nbl::system;

constexpr size_t kMinBufferBytes = 2ull * 1024ull * 1024ull;
constexpr uint64_t kDefaultSeed = 0x6a09e667f3bcc909ull;

enum class RuntimeMode : uint8_t
{
    Sequential,
    Heuristic,
    Hybrid
};

struct Options
{
    RuntimeMode mode = RuntimeMode::Heuristic;
    size_t bufferBytes = kMinBufferBytes;
    uint64_t seed = kDefaultSeed;
};

static const char* modeName(RuntimeMode mode)
{
    if (mode == RuntimeMode::Sequential)
        return "sequential";
    if (mode == RuntimeMode::Hybrid)
        return "hybrid";
    return "heuristic";
}

static SFileIOPolicy makePolicy(RuntimeMode mode)
{
    SFileIOPolicy policy = {};
    if (mode == RuntimeMode::Sequential)
        policy.runtimeTuning.mode = SFileIOPolicy::SRuntimeTuning::Mode::Sequential;
    else if (mode == RuntimeMode::Hybrid)
        policy.runtimeTuning.mode = SFileIOPolicy::SRuntimeTuning::Mode::Hybrid;
    else
        policy.runtimeTuning.mode = SFileIOPolicy::SRuntimeTuning::Mode::Heuristic;
    return policy;
}

static uint64_t nextRand(uint64_t& state)
{
    state ^= state >> 12u;
    state ^= state << 25u;
    state ^= state >> 27u;
    return state * 2685821657736338717ull;
}

static std::optional<Options> parseOptions(const core::vector<std::string>& args)
{
    argparse::ArgumentParser parser("hcp");
    parser.add_argument("--runtime-tuning").default_value(std::string("heuristic"));
    parser.add_argument("--buffer-bytes").default_value(std::to_string(kMinBufferBytes));
    parser.add_argument("--seed").default_value(std::to_string(kDefaultSeed));

    try
    {
        parser.parse_args({ args.data(), args.data() + args.size() });
    }
    catch (const std::exception&)
    {
        return std::nullopt;
    }

    auto parseU64 = [](const std::string& v) -> std::optional<uint64_t>
    {
        try { return std::stoull(v, nullptr, 10); } catch (...) { return std::nullopt; }
    };
    auto parseSize = [](const std::string& v) -> std::optional<size_t>
    {
        try
        {
            const auto x = std::stoull(v, nullptr, 10);
            if (x > static_cast<unsigned long long>(std::numeric_limits<size_t>::max()))
                return std::nullopt;
            return static_cast<size_t>(x);
        }
        catch (...)
        {
            return std::nullopt;
        }
    };

    Options options = {};
    const auto mode = parser.get<std::string>("--runtime-tuning");
    if (mode == "sequential" || mode == "none")
        options.mode = RuntimeMode::Sequential;
    else if (mode == "heuristic")
        options.mode = RuntimeMode::Heuristic;
    else if (mode == "hybrid")
        options.mode = RuntimeMode::Hybrid;
    else
        return std::nullopt;

    const auto bytes = parseSize(parser.get<std::string>("--buffer-bytes"));
    const auto seed = parseU64(parser.get<std::string>("--seed"));
    if (!bytes.has_value() || !seed.has_value() || *bytes < kMinBufferBytes)
        return std::nullopt;

    options.bufferBytes = *bytes;
    options.seed = *seed;
    return options;
}

static core::smart_refctd_ptr<ICPUPolygonGeometry> createGeometry(const Options& options)
{
    constexpr E_FORMAT positionFormat = EF_R32G32B32_SFLOAT;
    constexpr E_FORMAT normalFormat = EF_R32G32B32_SFLOAT;
    constexpr E_FORMAT indexFormat = EF_R32_UINT;
    constexpr E_FORMAT colorFormat = EF_R8G8B8A8_UNORM;

    const uint32_t positionStride = getTexelOrBlockBytesize(positionFormat);
    const uint32_t normalStride = getTexelOrBlockBytesize(normalFormat);
    const uint32_t indexStride = getTexelOrBlockBytesize(indexFormat);
    const uint32_t colorStride = getTexelOrBlockBytesize(colorFormat);
    const auto alignDown = [&](uint32_t stride) -> size_t { return options.bufferBytes - (options.bufferBytes % stride); };

    auto makeBuffer = [&](size_t bytes, core::bitflag<IBuffer::E_USAGE_FLAGS> usage, uint64_t stream) -> core::smart_refctd_ptr<ICPUBuffer>
    {
        std::vector<uint8_t> data(bytes);
        uint64_t state = options.seed ^ (stream * 0x9e3779b97f4a7c15ull);
        if (state == 0ull)
            state = kDefaultSeed ^ stream;
        for (auto& b : data)
            b = static_cast<uint8_t>(nextRand(state) & 0xffull);

        ICPUBuffer::SCreationParams params = {};
        params.size = data.size();
        params.usage = usage;
        params.data = data.data();
        return ICPUBuffer::create(std::move(params));
    };

    auto makeView = [](const core::smart_refctd_ptr<ICPUBuffer>& buffer, E_FORMAT format, uint32_t stride) -> ICPUPolygonGeometry::SDataView
    {
        ICPUPolygonGeometry::SDataView view = {};
        view.composed.format = format;
        view.composed.stride = stride;
        view.composed.rangeFormat = IGeometryBase::getMatchingAABBFormat(format);
        view.composed.resetRange();
        view.src.offset = 0ull;
        view.src.size = buffer ? buffer->getSize() : 0ull;
        view.src.buffer = buffer;
        return view;
    };

    auto positionBuffer = makeBuffer(alignDown(positionStride), IBuffer::EUF_VERTEX_BUFFER_BIT, 1ull);
    auto normalBuffer = makeBuffer(alignDown(normalStride), IBuffer::EUF_VERTEX_BUFFER_BIT, 2ull);
    auto indexBuffer = makeBuffer(alignDown(indexStride), IBuffer::EUF_INDEX_BUFFER_BIT, 3ull);
    auto colorBuffer = makeBuffer(alignDown(colorStride), IBuffer::EUF_VERTEX_BUFFER_BIT, 4ull);
    if (!positionBuffer || !normalBuffer || !indexBuffer || !colorBuffer)
        return nullptr;

    auto geometry = core::make_smart_refctd_ptr<ICPUPolygonGeometry>();
    geometry->setIndexing(IPolygonGeometryBase::TriangleList());
    geometry->setPositionView(makeView(positionBuffer, positionFormat, positionStride));
    geometry->setNormalView(makeView(normalBuffer, normalFormat, normalStride));
    geometry->setIndexView(makeView(indexBuffer, indexFormat, indexStride));
    geometry->getAuxAttributeViews()->push_back(makeView(colorBuffer, colorFormat, colorStride));
    geometry->getAuxAttributeViews()->push_back(makeView(colorBuffer, colorFormat, colorStride));
    return geometry;
}

static bool runParityCheck(const Options& options, ILogger* logger)
{
    using clock_t = std::chrono::high_resolution_clock;
    auto toMs = [](clock_t::duration d) { return std::chrono::duration<double, std::milli>(d).count(); };
    auto toMiB = [](size_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0); };
    auto throughput = [&](size_t bytes, double ms) { return ms > 0.0 ? toMiB(bytes) * 1000.0 / ms : 0.0; };

    auto geometry = createGeometry(options);
    if (!geometry)
    {
        logger->log("Failed to create dummy geometry.", ILogger::ELL_ERROR);
        return false;
    }

    core::vector<core::smart_refctd_ptr<ICPUBuffer>> buffers;
    SPolygonGeometryContentHash::collectBuffers(geometry.get(), buffers);
    if (buffers.empty())
    {
        logger->log("No buffers collected from geometry.", ILogger::ELL_ERROR);
        return false;
    }

    size_t totalBytes = 0ull;
    for (const auto& buffer : buffers)
        totalBytes += buffer ? buffer->getSize() : 0ull;
    if (totalBytes == 0ull)
    {
        logger->log("Collected zero-sized buffers.", ILogger::ELL_ERROR);
        return false;
    }

    const auto legacyPolicy = makePolicy(RuntimeMode::Sequential);
    SPolygonGeometryContentHash::reset(geometry.get());
    const auto legacyStart = clock_t::now();
    const auto legacyHash = SPolygonGeometryContentHash::recompute(geometry.get(), legacyPolicy);
    const double legacyMs = toMs(clock_t::now() - legacyStart);

    SPolygonGeometryContentHash::reset(geometry.get());
    const auto recomputeStart = clock_t::now();
    const auto recomputeHash = SPolygonGeometryContentHash::recompute(geometry.get(), makePolicy(options.mode));
    const double recomputeMs = toMs(clock_t::now() - recomputeStart);
    if (recomputeHash != legacyHash)
    {
        logger->log("recompute hash mismatch.", ILogger::ELL_ERROR);
        return false;
    }

    if (!buffers[0])
    {
        logger->log("First geometry buffer is null.", ILogger::ELL_ERROR);
        return false;
    }
    const auto preservedHash = buffers[0]->getContentHash();
    const size_t missingBytes = totalBytes - buffers[0]->getSize();
    SPolygonGeometryContentHash::reset(geometry.get());
    buffers[0]->setContentHash(preservedHash);
    const auto missingStart = clock_t::now();
    const auto missingHash = SPolygonGeometryContentHash::computeMissing(geometry.get(), makePolicy(options.mode));
    const double missingMs = toMs(clock_t::now() - missingStart);
    if (buffers[0]->getContentHash() != preservedHash)
    {
        logger->log("computeMissing overwrote pre-set hash.", ILogger::ELL_ERROR);
        return false;
    }
    if (missingHash != legacyHash)
    {
        logger->log("computeMissing hash mismatch.", ILogger::ELL_ERROR);
        return false;
    }

    logger->log("HCP mode=%s buffers=%llu total_mib=%.3f", ILogger::ELL_INFO, modeName(options.mode), static_cast<unsigned long long>(buffers.size()), toMiB(totalBytes));
    logger->log("HCP legacy ms=%.3f mib_s=%.3f", ILogger::ELL_INFO, legacyMs, throughput(totalBytes, legacyMs));
    logger->log("HCP recompute ms=%.3f mib_s=%.3f", ILogger::ELL_INFO, recomputeMs, throughput(totalBytes, recomputeMs));
    logger->log("HCP computeMissing ms=%.3f mib_s=%.3f missing_mib=%.3f", ILogger::ELL_INFO, missingMs, throughput(missingBytes, missingMs), toMiB(missingBytes));
    return true;
}

class HashContentParityApp final : public IApplicationFramework
{
public:
    using IApplicationFramework::IApplicationFramework;

    bool onAppInitialized(core::smart_refctd_ptr<ISystem>&&) override
    {
        m_logger = core::make_smart_refctd_ptr<CStdoutLogger>(ILogger::DefaultLogMask());
        if (!isAPILoaded())
        {
            m_logger->log("Could not load Nabla API.", ILogger::ELL_ERROR);
            return false;
        }

        const auto options = parseOptions(argv);
        if (!options.has_value())
        {
            m_logger->log("Usage: hcp [--runtime-tuning sequential|heuristic|hybrid] [--buffer-bytes N] [--seed U64]", ILogger::ELL_ERROR);
            m_logger->log("Constraint: --buffer-bytes must be >= %llu", ILogger::ELL_ERROR, static_cast<unsigned long long>(kMinBufferBytes));
            return false;
        }

        if (!runParityCheck(*options, m_logger.get()))
            return false;
        m_logger->log("OK", ILogger::ELL_INFO);
        return true;
    }

    void workLoopBody() override {}
    bool keepRunning() override { return false; }

private:
    core::smart_refctd_ptr<ILogger> m_logger;
};

NBL_MAIN_FUNC(HashContentParityApp)
