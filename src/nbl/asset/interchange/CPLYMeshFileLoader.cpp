#ifdef _NBL_COMPILE_WITH_PLY_LOADER_
// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors
#include "CPLYMeshFileLoader.h"
#include "impl/SBinaryData.h"
#include "impl/SFileAccess.h"
#include "impl/STextParse.h"
#include "nbl/asset/IAssetManager.h"
#include "nbl/asset/interchange/SGeometryContentHash.h"
#include "nbl/asset/interchange/SGeometryLoaderCommon.h"
#include "nbl/asset/interchange/SInterchangeIO.h"
#include "nbl/asset/interchange/SLoaderRuntimeTuning.h"
#include "nbl/asset/metadata/CPLYMetadata.h"
#include "nbl/builtin/hlsl/array_accessors.hlsl"
#include "nbl/builtin/hlsl/shapes/AABBAccumulator.hlsl"
#include "nbl/builtin/hlsl/vector_utils/vector_traits.hlsl"
#include "nbl/core/hash/blake.h"
#include "nbl/system/IFile.h"
#include "nbl/system/ISystem.h"
#include <chrono>
#include <thread>
namespace nbl::asset
{
namespace
{
struct Parse
{
    static constexpr uint32_t UV0 = 0u;
    using Binary = impl::BinaryData;
	using Common = impl::TextParse;
	struct ContentHashBuild
	{
		bool enabled = false;
		bool inlineHash = false;
		core::vector<core::smart_refctd_ptr<ICPUBuffer>> hashedBuffers = {};
		std::jthread deferredThread = {};
		static inline ContentHashBuild create(const bool enabled, const bool inlineHash) { return {.enabled = enabled, .inlineHash = inlineHash}; }
		inline bool hashesInline() const { return enabled && inlineHash; }
		inline bool hashesDeferred() const { return enabled && !inlineHash; }
		inline void hashNow(ICPUBuffer* const buffer)
		{
			if (!hashesInline() || !buffer || buffer->getContentHash() != IPreHashed::INVALID_HASH)
				return;
			for (const auto& hashed : hashedBuffers)
				if (hashed.get() == buffer)
					return;
			buffer->setContentHash(buffer->computeContentHash());
			hashedBuffers.push_back(core::smart_refctd_ptr<ICPUBuffer>(buffer));
		}
		inline void tryDefer(ICPUBuffer* const buffer)
		{
			if (!hashesDeferred() || !buffer || deferredThread.joinable() || buffer->getContentHash() != IPreHashed::INVALID_HASH)
				return;
			auto keepAlive = core::smart_refctd_ptr<ICPUBuffer>(buffer);
			deferredThread = std::jthread([buffer = std::move(keepAlive)]() mutable {buffer->setContentHash(buffer->computeContentHash());});
		}
		inline void wait() { if (deferredThread.joinable()) deferredThread.join(); }
	};
	static std::string_view toStringView(const char* text)
	{
		return text ? std::string_view{text} : std::string_view{};
	}
	template<size_t N>
	static E_FORMAT selectStructuredFormat(const std::array<E_FORMAT, N>& formats, const uint32_t componentCount)
	{
		return componentCount > 0u && componentCount <= N ? formats[componentCount - 1u] : EF_UNKNOWN;
	}
	static E_FORMAT expandStructuredFormat(const E_FORMAT componentFormat, const uint32_t componentCount)
	{
		switch (componentFormat)
		{
			case EF_R8_SINT: return selectStructuredFormat(std::to_array<E_FORMAT>({EF_R8_SINT, EF_R8G8_SINT, EF_R8G8B8_SINT, EF_R8G8B8A8_SINT}), componentCount);
			case EF_R8_UINT: return selectStructuredFormat(std::to_array<E_FORMAT>({EF_R8_UINT, EF_R8G8_UINT, EF_R8G8B8_UINT, EF_R8G8B8A8_UINT}), componentCount);
			case EF_R16_SINT: return selectStructuredFormat(std::to_array<E_FORMAT>({EF_R16_SINT, EF_R16G16_SINT, EF_R16G16B16_SINT, EF_R16G16B16A16_SINT}), componentCount);
			case EF_R16_UINT: return selectStructuredFormat(std::to_array<E_FORMAT>({EF_R16_UINT, EF_R16G16_UINT, EF_R16G16B16_UINT, EF_R16G16B16A16_UINT}), componentCount);
			case EF_R32_SINT: return selectStructuredFormat(std::to_array<E_FORMAT>({EF_R32_SINT, EF_R32G32_SINT, EF_R32G32B32_SINT, EF_R32G32B32A32_SINT}), componentCount);
			case EF_R32_UINT: return selectStructuredFormat(std::to_array<E_FORMAT>({EF_R32_UINT, EF_R32G32_UINT, EF_R32G32B32_UINT, EF_R32G32B32A32_UINT}), componentCount);
			case EF_R32_SFLOAT: return selectStructuredFormat(std::to_array<E_FORMAT>({EF_R32_SFLOAT, EF_R32G32_SFLOAT, EF_R32G32B32_SFLOAT, EF_R32G32B32A32_SFLOAT}), componentCount);
			case EF_R64_SFLOAT: return selectStructuredFormat(std::to_array<E_FORMAT>({EF_R64_SFLOAT, EF_R64G64_SFLOAT, EF_R64G64B64_SFLOAT, EF_R64G64B64A64_SFLOAT}), componentCount);
			default: return EF_UNKNOWN;
		}
	}
	struct Context
	{
		static constexpr uint64_t ReadWindowPaddingBytes = 1ull;
		struct SProperty
		{
			static E_FORMAT getType(const char* typeString)
			{
				struct STypeAlias
				{
					std::string_view name;
					E_FORMAT format;
				};
				constexpr std::array<STypeAlias, 16> typeAliases = {{
					{"char", EF_R8_SINT},
					{"int8", EF_R8_SINT},
					{"uchar", EF_R8_UINT},
					{"uint8", EF_R8_UINT},
					{"short", EF_R16_SINT},
					{"int16", EF_R16_SINT},
					{"ushort", EF_R16_UINT},
					{"uint16", EF_R16_UINT},
					{"long", EF_R32_SINT},
					{"int", EF_R32_SINT},
					{"int32", EF_R32_SINT},
					{"ulong", EF_R32_UINT},
					{"uint", EF_R32_UINT},
					{"uint32", EF_R32_UINT},
					{"float", EF_R32_SFLOAT},
					{"float32", EF_R32_SFLOAT}
				}};
				const std::string_view typeName = Parse::toStringView(typeString);
				for (const auto& alias : typeAliases)
				{
					if (alias.name == typeName)
						return alias.format;
				}
				if (typeName == "double" || typeName == "float64")
					return EF_R64_SFLOAT;
				return EF_UNKNOWN;
			}
			bool isList() const
			{
				return type == EF_UNKNOWN && asset::isIntegerFormat(list.countType) && asset::isIntegerFormat(list.itemType);
			}
			void skip(Context& _ctx) const
			{
				if (isList())
				{
					int32_t count = _ctx.getInt(list.countType);
					for (decltype(count) i = 0; i < count; ++i)
						_ctx.getInt(list.itemType);
				}
				else if (_ctx.IsBinaryFile)
					_ctx.moveForward(getTexelOrBlockBytesize(type));
				else
					_ctx.getNextWord();
			}
			std::string Name;
			E_FORMAT type;
			struct SListTypes
			{
				E_FORMAT countType;
				E_FORMAT itemType;
			} list;
		};
		struct SElement
		{
			void skipElement(Context& _ctx) const
			{
				if (_ctx.IsBinaryFile)
				{
					if (KnownSize)
						_ctx.moveForward(KnownSize);
					else
						for (auto i = 0u; i < Properties.size(); ++i)
							Properties[i].skip(_ctx);
				}
				else
					_ctx.getNextLine();
			}
			std::string Name; // name of the element. We only want "vertex" and "face" elements
			// but we have to parse the others anyway.
			core::vector<SProperty> Properties; // Properties of this element
			size_t Count; // The number of elements in the file
			uint32_t KnownSize; // known size in bytes, 0 if unknown
		};
		static constexpr size_t DefaultIoReadWindowBytes = 50ull << 10;
		void init(size_t _ioReadWindowSize = DefaultIoReadWindowBytes)
		{
			ioReadWindowSize = std::max<size_t>(_ioReadWindowSize, DefaultIoReadWindowBytes);
			Buffer.resize(ioReadWindowSize + ReadWindowPaddingBytes, '\0');
			EndPointer = StartPointer = Buffer.data();
			LineEndPointer = EndPointer - 1;
			fillBuffer();
		}
		void fillBuffer() // gets more data from the file
		{
			if (EndOfFile)
				return;
			if (fileOffset >= inner.mainFile->getSize())
			{
				EndOfFile = true;
				return;
			}
			const auto length = std::distance(StartPointer, EndPointer);
			auto newStart = Buffer.data();
			// copy the remaining data to the start of the buffer
			if (length && StartPointer != newStart)
				memmove(newStart, StartPointer, length);
			// reset start position
			StartPointer = newStart;
			EndPointer = newStart + length;
			const size_t usableBufferSize = Buffer.size() > 0ull ? Buffer.size() - ReadWindowPaddingBytes : 0ull;
			if (usableBufferSize <= length)
			{
				EndOfFile = true;
				return;
			}
			const size_t requestSize = usableBufferSize - length;
			system::IFile::success_t success;
			// read data from the file
			inner.mainFile->read(success, EndPointer, fileOffset, requestSize);
			const size_t bytesRead = success.getBytesProcessed();
			++readCallCount;
			readBytesTotal += bytesRead;
			if (bytesRead < readMinBytes)
				readMinBytes = bytesRead;
			fileOffset += bytesRead;
			EndPointer += bytesRead;
			// if we didn't completely fill the buffer
			if (bytesRead != requestSize)
			{
				// cauterize the string
				*EndPointer = 0;
				EndOfFile = true;
			}
		}
		const char* getNextLine() // Split the string data into a line in place by terminating it instead of copying.
		{
			// move the start pointer along
			StartPointer = LineEndPointer + 1;
			// crlf split across buffer move
			if (*StartPointer == '\n')
				*(StartPointer++) = '\0';
			// begin at the start of the next line
			const std::array<const char, 3> Terminators = {'\0', '\r', '\n'};
			auto terminator = std::find_first_of(StartPointer, EndPointer, Terminators.begin(), Terminators.end());
			if (terminator != EndPointer)
				*(terminator++) = '\0';
			// we have reached the end of the buffer
			if (terminator == EndPointer)
			{
				if (EndOfFile)
				{
					StartPointer = EndPointer - 1;
					*StartPointer = '\0';
					return StartPointer;
				}
				// get data from the file
				fillBuffer();
				// reset line end pointer
				LineEndPointer = StartPointer - 1;
				return StartPointer != EndPointer ? getNextLine() : StartPointer;
			}
			LineEndPointer = terminator - 1;
			WordLength = -1;
			// return pointer to the start of the line
			return StartPointer;
		}
		const char* getNextWord() // null terminate the next word on the previous line and move the next word pointer along since we already have a full line in the buffer, we never need to retrieve more data
		{
			// move the start pointer along
			StartPointer += WordLength + 1;
			if (StartPointer >= EndPointer)
			{
				if (EndOfFile)
				{
					WordLength = -1;
					return EndPointer;
				}
				getNextLine();
			}
			if (StartPointer < EndPointer && !*StartPointer)
				getNextLine();
			if (StartPointer >= LineEndPointer)
			{
				WordLength = -1;
				return StartPointer;
			}
			assert(LineEndPointer <= EndPointer);
			// process the next word
			const std::array<const char, 3> WhiteSpace = {'\0', ' ', '\t'};
			auto wordEnd = std::find_first_of(StartPointer, LineEndPointer, WhiteSpace.begin(), WhiteSpace.end());
			// null terminate the next word
			if (wordEnd != LineEndPointer)
				*(wordEnd++) = '\0';
			// find next word
			auto nextWord = std::find_if(wordEnd, LineEndPointer, [WhiteSpace](const char c) -> bool { return std::find(WhiteSpace.begin(), WhiteSpace.end(), c) == WhiteSpace.end(); });
			WordLength = std::distance(StartPointer, nextWord) - 1;
			// return pointer to the start of current word
			return StartPointer;
		}
		size_t getAbsoluteOffset(const char* ptr) const
		{
			if (!ptr || ptr > EndPointer)
				return fileOffset;
			const size_t trailingBytes = static_cast<size_t>(EndPointer - ptr);
			return fileOffset >= trailingBytes ? (fileOffset - trailingBytes) : 0ull;
		}
		void useMappedBinaryWindow(const char* data, const size_t sizeBytes)
		{
			if (!data)
				return;
			StartPointer = const_cast<char*>(data);
			EndPointer = StartPointer + sizeBytes;
			LineEndPointer = StartPointer - 1;
			WordLength = -1;
			EndOfFile = true;
			fileOffset = inner.mainFile ? inner.mainFile->getSize() : fileOffset;
		}
		void moveForward(const size_t bytes) // skips x bytes in the file, getting more data if required
		{
			assert(IsBinaryFile);
			size_t remaining = bytes;
			if (remaining == 0ull)
				return;
			const size_t availableInitially = EndPointer > StartPointer ? static_cast<size_t>(EndPointer - StartPointer) : 0ull;
			if (remaining > availableInitially)
			{
				remaining -= availableInitially;
				StartPointer = EndPointer;
				if (remaining > ioReadWindowSize)
				{
					const size_t fileSize = inner.mainFile->getSize();
					const size_t fileRemaining = fileSize > fileOffset ? (fileSize - fileOffset) : 0ull;
					const size_t directSkip = std::min(remaining, fileRemaining);
					fileOffset += directSkip;
					remaining -= directSkip;
				}
			}
			while (remaining)
			{
				if (StartPointer >= EndPointer)
				{
					fillBuffer();
					if (StartPointer >= EndPointer)
						return;
				}
				const size_t available = static_cast<size_t>(EndPointer - StartPointer);
				const size_t step = std::min(available, remaining);
				StartPointer += step;
				remaining -= step;
			}
		}
		using widest_int_t = uint32_t;
		const char* getCurrentWordEnd(const char* word) const
		{
			const size_t tokenLen = WordLength >= 0 ? static_cast<size_t>(WordLength + 1) : std::char_traits<char>::length(word);
			return word + tokenLen;
		}
		inline bool ensureBytes(const size_t bytes)
		{
			if (StartPointer + bytes > EndPointer)
				fillBuffer();
			return StartPointer + bytes <= EndPointer;
		}
		template<typename T>
		inline T loadBinaryScalar()
		{
			if (!ensureBytes(sizeof(T)))
				return T{};
			const auto retval = Binary::loadUnaligned<T>(StartPointer, IsWrongEndian);
			StartPointer += sizeof(T);
			return retval;
		}
		template<typename T>
		inline T parseCurrentWordValue()
		{
			const char* word = getNextWord();
			if (!word)
				return T{};
			const char* const wordEnd = getCurrentWordEnd(word);
			if (word == wordEnd)
				return T{};
			T value = {};
			auto ptr = word;
			if (Common::parseNumber(ptr, wordEnd, value) && ptr == wordEnd)
				return value;
			return ptr != word ? value : T{};
		}
		widest_int_t getInt(const E_FORMAT f) // read the next int from the file and move the start pointer along
		{
			assert(!isFloatingPointFormat(f));
			if (IsBinaryFile)
			{
				switch (getTexelOrBlockBytesize(f))
				{
					case 1:
						if (ensureBytes(sizeof(int8_t)))
							return *(StartPointer++);
						break;
					case 2: return static_cast<widest_int_t>(loadBinaryScalar<int16_t>());
					case 4: return static_cast<widest_int_t>(loadBinaryScalar<int32_t>());
					default:
						assert(false);
						break;
				}
				return 0u;
			}
			return isSignedFormat(f) ? static_cast<widest_int_t>(parseCurrentWordValue<int64_t>()) : static_cast<widest_int_t>(parseCurrentWordValue<uint64_t>());
		}
		hlsl::float64_t getFloat(const E_FORMAT f) // read the next float from the file and move the start pointer along
		{
			assert(isFloatingPointFormat(f));
			if (IsBinaryFile)
			{
				switch (getTexelOrBlockBytesize(f))
				{
					case 4: return loadBinaryScalar<hlsl::float32_t>();
					case 8: return loadBinaryScalar<hlsl::float64_t>();
					default:
						assert(false);
						break;
				}
				return 0.0;
			}
			return parseCurrentWordValue<hlsl::float64_t>();
		}
		void getData(void* dst, const E_FORMAT f) // read the next thing from the file and move the start pointer along
		{
			const auto size = getTexelOrBlockBytesize(f);
			if (!ensureBytes(size))
				return;
			if (IsWrongEndian)
				std::reverse_copy(StartPointer, StartPointer + size, reinterpret_cast<char*>(dst));
			else
				memcpy(dst, StartPointer, size);
			StartPointer += size;
		}
        struct SVertAttrIt {
            uint8_t* ptr;
            uint32_t stride;
            E_FORMAT dstFmt;
        };
        enum class EFastVertexReadResult : uint8_t {
            NotApplicable,
            Success,
            Error
        };
        EFastVertexReadResult readVertexElementFast(
            const SElement& el,
            hlsl::shapes::util::AABBAccumulator3<float>* parsedAABB) {
            if (!IsBinaryFile || el.Name != "vertex")
                return EFastVertexReadResult::NotApplicable;
            enum class ELayoutKind : uint8_t { XYZ, XYZ_N, XYZ_N_UV };
            auto allF32 = [&el]()->bool {
                for (const auto& prop : el.Properties)
                    if (prop.type != EF_R32_SFLOAT)
                        return false;
                return true;
            };
            if (!allF32())
                return EFastVertexReadResult::NotApplicable;
            auto matchNames = [&el](std::initializer_list<const char*> names)->bool {
                if (el.Properties.size() != names.size())
                    return false;
                size_t i = 0ull;
                for (const auto* name : names)
                {
                    if (el.Properties[i].Name != name)
                        return false;
                    ++i;
                }
                return true;
            };
            ELayoutKind layout = ELayoutKind::XYZ;
            if (matchNames({"x", "y", "z"}))
                layout = ELayoutKind::XYZ;
            else if (matchNames({"x", "y", "z", "nx", "ny", "nz"}))
                layout = ELayoutKind::XYZ_N;
            else if (matchNames({"x", "y", "z", "nx", "ny", "nz", "u", "v"}) ||
                     matchNames({"x", "y", "z", "nx", "ny", "nz", "s", "t"}))
                layout = ELayoutKind::XYZ_N_UV;
            else
                return EFastVertexReadResult::NotApplicable;
            const size_t floatBytes = sizeof(hlsl::float32_t);
            auto validateTuple = [&](const size_t beginIx, const size_t componentCount, uint32_t& outStride, uint8_t*& outBase)->bool {
                if (beginIx + componentCount > vertAttrIts.size())
                    return false;
                auto& first = vertAttrIts[beginIx];
                if (!first.ptr || first.dstFmt != EF_R32_SFLOAT)
                    return false;
                outStride = first.stride;
                outBase = first.ptr;
                for (size_t c = 1ull; c < componentCount; ++c)
                {
                    auto& it = vertAttrIts[beginIx + c];
                    if (!it.ptr || it.dstFmt != EF_R32_SFLOAT)
                        return false;
                    if (it.stride != outStride)
                        return false;
                    if (it.ptr != outBase + c * floatBytes)
                        return false;
                }
                return true;
            };
            uint32_t posStride = 0u, normalStride = 0u, uvStride = 0u;
            uint8_t* posBase = nullptr;
            uint8_t* normalBase = nullptr;
            uint8_t* uvBase = nullptr;
            switch (layout)
            {
                case ELayoutKind::XYZ:
                    if (vertAttrIts.size() != 3u || !validateTuple(0u, 3u, posStride, posBase))
                        return EFastVertexReadResult::NotApplicable;
                    break;
                case ELayoutKind::XYZ_N:
                    if (vertAttrIts.size() != 6u)
                        return EFastVertexReadResult::NotApplicable;
                    if (!validateTuple(0u, 3u, posStride, posBase) || !validateTuple(3u, 3u, normalStride, normalBase))
                        return EFastVertexReadResult::NotApplicable;
                    break;
                case ELayoutKind::XYZ_N_UV:
                    if (vertAttrIts.size() != 8u)
                        return EFastVertexReadResult::NotApplicable;
                    if (!validateTuple(0u, 3u, posStride, posBase) || !validateTuple(3u, 3u, normalStride, normalBase) || !validateTuple(6u, 2u, uvStride, uvBase))
                        return EFastVertexReadResult::NotApplicable;
                    break;
            }
            const size_t srcBytesPerVertex = [layout]()->size_t {
                switch (layout)
                {
                    case ELayoutKind::XYZ: return sizeof(hlsl::float32_t) * 3ull;
                    case ELayoutKind::XYZ_N: return sizeof(hlsl::float32_t) * 6ull;
                    case ELayoutKind::XYZ_N_UV: return sizeof(hlsl::float32_t) * 8ull;
                    default: return 0ull;
                }
            }();
            if (srcBytesPerVertex == 0ull || el.Count > (std::numeric_limits<size_t>::max() / srcBytesPerVertex))
                return EFastVertexReadResult::Error;
            const bool trackAABB = parsedAABB != nullptr;
            const bool needsByteSwap = IsWrongEndian;
            auto decodeF32 = [needsByteSwap](const uint8_t* src)->float {
                uint32_t bits = 0u;
                std::memcpy(&bits, src, sizeof(bits));
                if (needsByteSwap)
                    bits = Binary::byteswap(bits);
                float value = 0.f;
                std::memcpy(&value, &bits, sizeof(value));
                return value;
            };
            size_t remainingVertices = el.Count;
            while (remainingVertices > 0ull)
            {
                if (StartPointer + srcBytesPerVertex > EndPointer)
                    fillBuffer();
                const size_t available = EndPointer > StartPointer ? static_cast<size_t>(EndPointer - StartPointer) : 0ull;
                if (available < srcBytesPerVertex)
                    return EFastVertexReadResult::Error;
                const size_t batchVertices = std::min(remainingVertices, available / srcBytesPerVertex);
                const uint8_t* src = reinterpret_cast<const uint8_t*>(StartPointer);
                switch (layout)
                {
                    case ELayoutKind::XYZ:
                    {
                        if (posStride == 3ull * floatBytes)
                        {
                            const size_t batchBytes = batchVertices * 3ull * floatBytes;
                            if (trackAABB && batchVertices >= (1ull << 20))
                            {
                                const size_t hw = SLoaderRuntimeTuner::resolveHardwareThreads();
                                const size_t hardMaxWorkers = SLoaderRuntimeTuner::resolveHardMaxWorkers(hw, inner.params.ioPolicy.runtimeTuning.workerHeadroom);
                                SLoaderRuntimeTuningRequest vertexTuningRequest = {};
                                vertexTuningRequest.inputBytes = batchBytes;
                                vertexTuningRequest.totalWorkUnits = batchVertices;
                                vertexTuningRequest.minBytesPerWorker = 3ull * floatBytes;
                                vertexTuningRequest.hardwareThreads = static_cast<uint32_t>(hw);
                                vertexTuningRequest.hardMaxWorkers = static_cast<uint32_t>(hardMaxWorkers);
                                vertexTuningRequest.targetChunksPerWorker = inner.params.ioPolicy.runtimeTuning.targetChunksPerWorker;
                                vertexTuningRequest.sampleData = reinterpret_cast<const uint8_t*>(src);
                                vertexTuningRequest.sampleBytes = SLoaderRuntimeTuner::resolveSampleBytes(inner.params.ioPolicy, batchBytes);
                                const auto vertexTuning = SLoaderRuntimeTuner::tune(inner.params.ioPolicy, vertexTuningRequest);
                                const size_t workerCount = std::min(vertexTuning.workerCount, batchVertices);
                                if (workerCount > 1ull)
                                {
                                    struct SAABBRange { float minX = std::numeric_limits<float>::max(); float minY = std::numeric_limits<float>::max(); float minZ = std::numeric_limits<float>::max(); float maxX = std::numeric_limits<float>::lowest(); float maxY = std::numeric_limits<float>::lowest(); float maxZ = std::numeric_limits<float>::lowest(); };
                                    std::vector<SAABBRange> workerRanges(workerCount);
                                    uint8_t* dstBase = posBase;
                                    SLoaderRuntimeTuner::dispatchWorkers(workerCount, [&](const size_t workerIx) {
                                        const size_t begin = (batchVertices * workerIx) / workerCount;
                                        const size_t end = (batchVertices * (workerIx + 1ull)) / workerCount;
                                        const size_t count = end - begin;
                                        if (count == 0ull)
                                            return;
                                        auto& range = workerRanges[workerIx];
                                        const uint8_t* inBytes = src + begin * 3ull * floatBytes;
                                        float* outFloats = reinterpret_cast<float*>(dstBase + begin * 3ull * floatBytes);
                                        if (!needsByteSwap)
                                        {
                                            std::memcpy(outFloats, inBytes, count * 3ull * floatBytes);
                                            const float* xyz = reinterpret_cast<const float*>(inBytes);
                                            for (size_t v = 0ull; v < count; ++v)
                                            {
                                                const float x = xyz[v * 3ull + 0ull];
                                                const float y = xyz[v * 3ull + 1ull];
                                                const float z = xyz[v * 3ull + 2ull];
                                                if (x < range.minX) range.minX = x;
                                                if (y < range.minY) range.minY = y;
                                                if (z < range.minZ) range.minZ = z;
                                                if (x > range.maxX) range.maxX = x;
                                                if (y > range.maxY) range.maxY = y;
                                                if (z > range.maxZ) range.maxZ = z;
                                            }
                                        }
                                        else
                                        {
                                            for (size_t v = 0ull; v < count; ++v)
                                            {
                                                uint32_t xb = 0u, yb = 0u, zb = 0u;
                                                std::memcpy(&xb, inBytes + 0ull * floatBytes, sizeof(xb));
                                                std::memcpy(&yb, inBytes + 1ull * floatBytes, sizeof(yb));
                                                std::memcpy(&zb, inBytes + 2ull * floatBytes, sizeof(zb));
                                                xb = Binary::byteswap(xb);
                                                yb = Binary::byteswap(yb);
                                                zb = Binary::byteswap(zb);
                                                float x = 0.f, y = 0.f, z = 0.f;
                                                std::memcpy(&x, &xb, sizeof(x));
                                                std::memcpy(&y, &yb, sizeof(y));
                                                std::memcpy(&z, &zb, sizeof(z));
                                                outFloats[0] = x;
                                                outFloats[1] = y;
                                                outFloats[2] = z;
                                                if (x < range.minX) range.minX = x;
                                                if (y < range.minY) range.minY = y;
                                                if (z < range.minZ) range.minZ = z;
                                                if (x > range.maxX) range.maxX = x;
                                                if (y > range.maxY) range.maxY = y;
                                                if (z > range.maxZ) range.maxZ = z;
                                                inBytes += 3ull * floatBytes;
                                                outFloats += 3ull;
                                            }
                                        }
                                    });
                                    auto& aabb = parsedAABB->value;
                                    for (const auto& range : workerRanges)
                                    {
                                        if (range.minX < aabb.minVx.x) aabb.minVx.x = range.minX;
                                        if (range.minY < aabb.minVx.y) aabb.minVx.y = range.minY;
                                        if (range.minZ < aabb.minVx.z) aabb.minVx.z = range.minZ;
                                        if (range.maxX > aabb.maxVx.x) aabb.maxVx.x = range.maxX;
                                        if (range.maxY > aabb.maxVx.y) aabb.maxVx.y = range.maxY;
                                        if (range.maxZ > aabb.maxVx.z) aabb.maxVx.z = range.maxZ;
                                    }
                                    src += batchBytes;
                                    posBase += batchBytes;
                                    break;
                                }
                            }
                            if (!needsByteSwap)
                            {
                                std::memcpy(posBase, src, batchBytes);
                                if (trackAABB)
                                {
                                    const float* xyz = reinterpret_cast<const float*>(src);
                                    auto& aabb = parsedAABB->value;
                                    for (size_t v = 0ull; v < batchVertices; ++v)
                                    {
                                        const float x = xyz[v * 3ull + 0ull];
                                        const float y = xyz[v * 3ull + 1ull];
                                        const float z = xyz[v * 3ull + 2ull];
                                        if (x < aabb.minVx.x) aabb.minVx.x = x;
                                        if (y < aabb.minVx.y) aabb.minVx.y = y;
                                        if (z < aabb.minVx.z) aabb.minVx.z = z;
                                        if (x > aabb.maxVx.x) aabb.maxVx.x = x;
                                        if (y > aabb.maxVx.y) aabb.maxVx.y = y;
                                        if (z > aabb.maxVx.z) aabb.maxVx.z = z;
                                    }
                                }
                                src += batchBytes;
                                posBase += batchBytes;
                            }
                            else
                            {
                                for (size_t v = 0ull; v < batchVertices; ++v)
                                {
                                    const float x = decodeF32(src + 0ull * floatBytes);
                                    const float y = decodeF32(src + 1ull * floatBytes);
                                    const float z = decodeF32(src + 2ull * floatBytes);
                                    reinterpret_cast<float*>(posBase)[0] = x;
                                    reinterpret_cast<float*>(posBase)[1] = y;
                                    reinterpret_cast<float*>(posBase)[2] = z;
                                    if (trackAABB)
                                        hlsl::shapes::util::extendAABBAccumulator(*parsedAABB, x, y, z);
                                    src += 3ull * floatBytes;
                                    posBase += posStride;
                                }
                            }
                        }
                        else
                        {
                            for (size_t v = 0ull; v < batchVertices; ++v)
                            {
                                const float x = decodeF32(src + 0ull * floatBytes);
                                const float y = decodeF32(src + 1ull * floatBytes);
                                const float z = decodeF32(src + 2ull * floatBytes);
                                reinterpret_cast<float*>(posBase)[0] = x;
                                reinterpret_cast<float*>(posBase)[1] = y;
                                reinterpret_cast<float*>(posBase)[2] = z;
                                if (trackAABB)
                                    hlsl::shapes::util::extendAABBAccumulator(*parsedAABB, x, y, z);
                                src += 3ull * floatBytes;
                                posBase += posStride;
                            }
                        }
                    }
                    break;
                    case ELayoutKind::XYZ_N:
                    {
                        for (size_t v = 0ull; v < batchVertices; ++v)
                        {
                            const float x = decodeF32(src + 0ull * floatBytes);
                            const float y = decodeF32(src + 1ull * floatBytes);
                            const float z = decodeF32(src + 2ull * floatBytes);
                            reinterpret_cast<float*>(posBase)[0] = x;
                            reinterpret_cast<float*>(posBase)[1] = y;
                            reinterpret_cast<float*>(posBase)[2] = z;
                            if (trackAABB)
                                hlsl::shapes::util::extendAABBAccumulator(*parsedAABB, hlsl::float32_t3(x, y, z));
                            src += 3ull * floatBytes;
                            posBase += posStride;
                            reinterpret_cast<float*>(normalBase)[0] = decodeF32(src + 0ull * floatBytes);
                            reinterpret_cast<float*>(normalBase)[1] = decodeF32(src + 1ull * floatBytes);
                            reinterpret_cast<float*>(normalBase)[2] = decodeF32(src + 2ull * floatBytes);
                            src += 3ull * floatBytes;
                            normalBase += normalStride;
                        }
                    }
                    break;
                    case ELayoutKind::XYZ_N_UV:
                    {
                        for (size_t v = 0ull; v < batchVertices; ++v)
                        {
                            const float x = decodeF32(src + 0ull * floatBytes);
                            const float y = decodeF32(src + 1ull * floatBytes);
                            const float z = decodeF32(src + 2ull * floatBytes);
                            reinterpret_cast<float*>(posBase)[0] = x;
                            reinterpret_cast<float*>(posBase)[1] = y;
                            reinterpret_cast<float*>(posBase)[2] = z;
                            if (trackAABB)
                                hlsl::shapes::util::extendAABBAccumulator(*parsedAABB, hlsl::float32_t3(x, y, z));
                            src += 3ull * floatBytes;
                            posBase += posStride;
                            reinterpret_cast<float*>(normalBase)[0] = decodeF32(src + 0ull * floatBytes);
                            reinterpret_cast<float*>(normalBase)[1] = decodeF32(src + 1ull * floatBytes);
                            reinterpret_cast<float*>(normalBase)[2] = decodeF32(src + 2ull * floatBytes);
                            src += 3ull * floatBytes;
                            normalBase += normalStride;
                            reinterpret_cast<float*>(uvBase)[0] = decodeF32(src + 0ull * floatBytes);
                            reinterpret_cast<float*>(uvBase)[1] = decodeF32(src + 1ull * floatBytes);
                            src += 2ull * floatBytes;
                            uvBase += uvStride;
                        }
                    }
                }
                const size_t consumed = batchVertices * srcBytesPerVertex;
                StartPointer += consumed;
                remainingVertices -= batchVertices;
            }
            const size_t posAdvance = el.Count * posStride;
            vertAttrIts[0].ptr += posAdvance;
            vertAttrIts[1].ptr += posAdvance;
            vertAttrIts[2].ptr += posAdvance;
            if (layout == ELayoutKind::XYZ_N || layout == ELayoutKind::XYZ_N_UV)
            {
                const size_t normalAdvance = el.Count * normalStride;
                vertAttrIts[3].ptr += normalAdvance;
                vertAttrIts[4].ptr += normalAdvance;
                vertAttrIts[5].ptr += normalAdvance;
            }
            if (layout == ELayoutKind::XYZ_N_UV)
            {
                const size_t uvAdvance = el.Count * uvStride;
                vertAttrIts[6].ptr += uvAdvance;
                vertAttrIts[7].ptr += uvAdvance;
            }
            return EFastVertexReadResult::Success;
        }
        void readVertex(const IAssetLoader::SAssetLoadParams& _params,
                        const SElement& el) {
            assert(el.Name == "vertex");
            assert(el.Properties.size() == vertAttrIts.size());
            if (!IsBinaryFile)
                getNextLine();
            for (size_t j = 0; j < el.Count; ++j)
                for (auto i = 0u; i < vertAttrIts.size(); i++) {
                    const auto& prop = el.Properties[i];
                    auto& it = vertAttrIts[i];
                    if (!it.ptr) {
                        prop.skip(*this);
                        continue;
                    }
                    if (!IsBinaryFile) {
                        if (isIntegerFormat(prop.type)) {
                            uint64_t tmp = getInt(prop.type);
                            encodePixels(it.dstFmt, it.ptr, &tmp);
                        } else {
                            hlsl::float64_t tmp = getFloat(prop.type);
                            encodePixels(it.dstFmt, it.ptr, &tmp);
                        }
                    } else if (it.dstFmt != prop.type) {
                        assert(isIntegerFormat(it.dstFmt) == isIntegerFormat(prop.type));
                        if (isIntegerFormat(it.dstFmt)) {
                            uint64_t tmp = getInt(prop.type);
                            encodePixels(it.dstFmt, it.ptr, &tmp);
                        } else {
                            hlsl::float64_t tmp = getFloat(prop.type);
                            encodePixels(it.dstFmt, it.ptr, &tmp);
                        }
                    } else
                        getData(it.ptr, prop.type);
                    //
                    it.ptr += it.stride;
                }
        }
        bool readFace(const SElement& Element, core::vector<uint32_t>& _outIndices,
                      uint32_t& _maxIndex, const uint32_t vertexCount) {
            if (!IsBinaryFile)
                getNextLine();
            const bool hasVertexCount = vertexCount != 0u;
            for (const auto& prop : Element.Properties) {
                if (prop.isList() &&
                    (prop.Name == "vertex_indices" || prop.Name == "vertex_index")) {
                    const uint32_t count = getInt(prop.list.countType);
                    const auto srcIndexFmt = prop.list.itemType;
                    if (count < 3u) {
                        for (uint32_t j = 0u; j < count; ++j)
                            getInt(srcIndexFmt);
                        continue;
                    }
                    if (count > 3u)
                        _outIndices.reserve(_outIndices.size() +
                                            static_cast<size_t>(count - 2u) * 3ull);
                    auto emitFan = [&_outIndices, &_maxIndex, hasVertexCount,
                                    vertexCount](auto&& readIndex,
                                                 const uint32_t faceVertexCount) -> bool {
                        uint32_t i0 = readIndex();
                        uint32_t i1 = readIndex();
                        uint32_t i2 = readIndex();
                        if (hasVertexCount) {
                            if (i0 >= vertexCount || i1 >= vertexCount || i2 >= vertexCount)
                                return false;
                        } else {
                            _maxIndex = std::max(_maxIndex, std::max(i0, std::max(i1, i2)));
                        }
                        _outIndices.push_back(i0);
                        _outIndices.push_back(i1);
                        _outIndices.push_back(i2);
                        uint32_t prev = i2;
                        for (uint32_t j = 3u; j < faceVertexCount; ++j) {
                            const uint32_t idx = readIndex();
                            if (hasVertexCount) {
                                if (idx >= vertexCount)
                                    return false;
                            } else {
                                _maxIndex = std::max(_maxIndex, idx);
                            }
                            _outIndices.push_back(i0);
                            _outIndices.push_back(prev);
						_outIndices.push_back(idx);
						prev = idx;
					}
					return true;
				};
				auto tryReadContiguousFan = [&]<typename T>() -> bool {
					const size_t bytesNeeded = static_cast<size_t>(count) * sizeof(T);
					if (!ensureBytes(bytesNeeded))
						return false;
					const uint8_t* ptr = reinterpret_cast<const uint8_t*>(StartPointer);
					auto readIndex = [&ptr]() -> uint32_t {
						T v = {};
						std::memcpy(&v, ptr, sizeof(v));
						ptr += sizeof(v);
						return static_cast<uint32_t>(v);
					};
					if (!emitFan(readIndex, count))
						return false;
					StartPointer = reinterpret_cast<char*>(const_cast<uint8_t*>(ptr));
					return true;
				};
				if (IsBinaryFile && !IsWrongEndian && srcIndexFmt == EF_R32_UINT && tryReadContiguousFan.template operator()<uint32_t>())
					continue;
				if (IsBinaryFile && !IsWrongEndian && srcIndexFmt == EF_R16_UINT && tryReadContiguousFan.template operator()<uint16_t>())
					continue;
				auto readIndex = [&]() -> uint32_t {
					return static_cast<uint32_t>(getInt(srcIndexFmt));
				};
                    if (!emitFan(readIndex, count))
                        return false;
                } else if (prop.Name == "intensity") {
                    // todo: face intensity
                    prop.skip(*this);
                } else
                    prop.skip(*this);
            }
            return true;
        }
        enum class EFastFaceReadResult : uint8_t { NotApplicable,
                                                   Success,
                                                   Error };
        EFastFaceReadResult readFaceElementFast(
            const SElement& element, core::vector<uint32_t>& _outIndices,
            uint32_t& _maxIndex, uint64_t& _faceCount, const uint32_t vertexCount,
            const bool computeIndexHash, core::blake3_hash_t& outIndexHash) {
            if (!IsBinaryFile)
                return EFastFaceReadResult::NotApplicable;
            if (element.Properties.size() != 1u)
                return EFastFaceReadResult::NotApplicable;
            const auto& prop = element.Properties[0];
            if (!prop.isList() ||
                (prop.Name != "vertex_indices" && prop.Name != "vertex_index"))
                return EFastFaceReadResult::NotApplicable;
            if (prop.list.countType != EF_R8_UINT)
                return EFastFaceReadResult::NotApplicable;
            const E_FORMAT srcIndexFmt = prop.list.itemType;
            const bool isSrcU32 = srcIndexFmt == EF_R32_UINT;
            const bool isSrcS32 = srcIndexFmt == EF_R32_SINT;
            const bool isSrcU16 = srcIndexFmt == EF_R16_UINT;
            const bool isSrcS16 = srcIndexFmt == EF_R16_SINT;
            if (!isSrcU32 && !isSrcS32 && !isSrcU16 && !isSrcS16)
                return EFastFaceReadResult::NotApplicable;
            const bool is32Bit = isSrcU32 || isSrcS32;
            const bool needEndianSwap = IsWrongEndian;
            const size_t indexSize = is32Bit ? sizeof(uint32_t) : sizeof(uint16_t);
            const bool hasVertexCount = vertexCount != 0u;
            const bool trackMaxIndex = !hasVertexCount;
            outIndexHash = IPreHashed::INVALID_HASH;
            const size_t minTriangleRecordSize = sizeof(uint8_t) + indexSize * 3u;
            if (element.Count >
                (std::numeric_limits<size_t>::max() / minTriangleRecordSize))
                return EFastFaceReadResult::Error;
            const size_t minBytesNeeded = element.Count * minTriangleRecordSize;
            if (StartPointer + minBytesNeeded <= EndPointer) {
                if (element.Count > (std::numeric_limits<size_t>::max() / 3u))
                    return EFastFaceReadResult::Error;
                const size_t triIndices = element.Count * 3u;
                if (_outIndices.size() >
                    (std::numeric_limits<size_t>::max() - triIndices))
                    return EFastFaceReadResult::Error;
                const size_t oldSize = _outIndices.size();
                const uint32_t oldMaxIndex = _maxIndex;
                _outIndices.resize(oldSize + triIndices);
                uint32_t* out = _outIndices.data() + oldSize;
                const uint8_t* ptr = reinterpret_cast<const uint8_t*>(StartPointer);
                auto readU32 = [needEndianSwap](const uint8_t* src) -> uint32_t {
                    uint32_t value = 0u;
                    std::memcpy(&value, src, sizeof(value));
                    if (needEndianSwap)
                        value = Binary::byteswap(value);
                    return value;
                };
                auto readU16 = [needEndianSwap](const uint8_t* src) -> uint16_t {
                    uint16_t value = 0u;
                    std::memcpy(&value, src, sizeof(value));
                    if (needEndianSwap)
                        value = Binary::byteswap(value);
                    return value;
                };
                if (is32Bit) {
                    const size_t hw = SLoaderRuntimeTuner::resolveHardwareThreads();
                    const size_t hardMaxWorkers =
                        SLoaderRuntimeTuner::resolveHardMaxWorkers(
                            hw, inner.params.ioPolicy.runtimeTuning.workerHeadroom);
                    const size_t recordBytes = sizeof(uint8_t) + 3ull * sizeof(uint32_t);
                    SLoaderRuntimeTuningRequest faceTuningRequest = {};
                    faceTuningRequest.inputBytes = minBytesNeeded;
                    faceTuningRequest.totalWorkUnits = element.Count;
                    faceTuningRequest.minBytesPerWorker = recordBytes;
                    faceTuningRequest.hardwareThreads = static_cast<uint32_t>(hw);
                    faceTuningRequest.hardMaxWorkers =
                        static_cast<uint32_t>(hardMaxWorkers);
                    faceTuningRequest.targetChunksPerWorker =
                        inner.params.ioPolicy.runtimeTuning.targetChunksPerWorker;
                    faceTuningRequest.sampleData = ptr;
                    faceTuningRequest.sampleBytes =
                        SLoaderRuntimeTuner::resolveSampleBytes(inner.params.ioPolicy,
                                                                minBytesNeeded);
                    const auto faceTuning = SLoaderRuntimeTuner::tune(
                        inner.params.ioPolicy, faceTuningRequest);
                    size_t workerCount = std::min(faceTuning.workerCount, element.Count);
                    if (workerCount > 1ull) {
                        const bool needMax = trackMaxIndex;
                        const bool validateAgainstVertexCount = hasVertexCount;
                        std::vector<uint8_t> workerNonTriangle(workerCount, 0u);
                        std::vector<uint8_t> workerInvalid(workerCount, 0u);
                        std::vector<uint32_t> workerMax(needMax ? workerCount : 0ull, 0u);
                        const bool hashInParsePipeline = computeIndexHash;
                        std::vector<uint8_t> workerReady(
                            hashInParsePipeline ? workerCount : 0ull, 0u);
                        std::vector<uint8_t> workerHashable(
                            hashInParsePipeline ? workerCount : 0ull, 1u);
                        std::atomic_bool hashPipelineOk = true;
                        core::blake3_hash_t parsedIndexHash = IPreHashed::INVALID_HASH;
                        std::jthread hashThread;
                        if (hashInParsePipeline) {
                            hashThread = std::jthread([&]() {
                                try {
                                    core::blake3_hasher hasher;
                                    for (size_t workerIx = 0ull; workerIx < workerCount;
                                         ++workerIx) {
                                        auto ready =
                                            std::atomic_ref<uint8_t>(workerReady[workerIx]);
                                        while (ready.load(std::memory_order_acquire) == 0u)
                                            ready.wait(0u, std::memory_order_acquire);
                                        if (workerHashable[workerIx] == 0u) {
                                            hashPipelineOk.store(false, std::memory_order_relaxed);
                                            return;
                                        }
                                        const size_t begin =
                                            (element.Count * workerIx) / workerCount;
                                        const size_t end =
                                            (element.Count * (workerIx + 1ull)) / workerCount;
                                        const size_t faceCount = end - begin;
                                        hasher.update(out + begin * 3ull,
                                                      faceCount * 3ull * sizeof(uint32_t));
                                    }
                                    parsedIndexHash = static_cast<core::blake3_hash_t>(hasher);
                                } catch (...) {
                                    hashPipelineOk.store(false, std::memory_order_relaxed);
                                }
                            });
                        }
                        auto parseChunk = [&](const size_t workerIx, const size_t beginFace,
                                              const size_t endFace) -> void {
                            const uint8_t* in = ptr + beginFace * recordBytes;
                            uint32_t* outLocal = out + beginFace * 3ull;
                            uint32_t localMax = 0u;
                            for (size_t faceIx = beginFace; faceIx < endFace; ++faceIx) {
                                if (*in != 3u) {
                                    workerNonTriangle[workerIx] = 1u;
                                    if (hashInParsePipeline)
                                        workerHashable[workerIx] = 0u;
                                    break;
                                }
                                ++in;
                                const uint32_t i0 = readU32(in + 0ull * sizeof(uint32_t));
                                const uint32_t i1 = readU32(in + 1ull * sizeof(uint32_t));
                                const uint32_t i2 = readU32(in + 2ull * sizeof(uint32_t));
                                outLocal[0] = i0;
                                outLocal[1] = i1;
                                outLocal[2] = i2;
                                const uint32_t triOr = i0 | i1 | i2;
                                if (isSrcS32 && (triOr & 0x80000000u)) {
                                    workerInvalid[workerIx] = 1u;
                                    if (hashInParsePipeline)
                                        workerHashable[workerIx] = 0u;
                                    break;
                                }
                                if (validateAgainstVertexCount) {
                                    if (i0 >= vertexCount || i1 >= vertexCount || i2 >= vertexCount) {
                                        workerInvalid[workerIx] = 1u;
                                        if (hashInParsePipeline)
                                            workerHashable[workerIx] = 0u;
                                        break;
                                    }
                                } else if (needMax) {
                                    if (i0 > localMax) localMax = i0;
                                    if (i1 > localMax) localMax = i1;
                                    if (i2 > localMax) localMax = i2;
                                }
                                in += 3ull * sizeof(uint32_t);
                                outLocal += 3ull;
                            }
                            if (needMax)
                                workerMax[workerIx] = localMax;
                            if (hashInParsePipeline) {
                                auto ready = std::atomic_ref<uint8_t>(workerReady[workerIx]);
                                ready.store(1u, std::memory_order_release);
                                ready.notify_one();
                            }
                        };
                        SLoaderRuntimeTuner::dispatchWorkers(
                            workerCount, [&](const size_t workerIx) {
                                const size_t begin = (element.Count * workerIx) / workerCount;
                                const size_t end =
                                    (element.Count * (workerIx + 1ull)) / workerCount;
                                parseChunk(workerIx, begin, end);
                            });
                        if (hashThread.joinable())
                            hashThread.join();
                        const bool anyNonTriangle =
                            std::any_of(workerNonTriangle.begin(), workerNonTriangle.end(),
                                        [](const uint8_t v) { return v != 0u; });
                        if (anyNonTriangle) {
                            _outIndices.resize(oldSize);
                            _maxIndex = oldMaxIndex;
                            return EFastFaceReadResult::NotApplicable;
                        }
                        const bool anyInvalid =
                            std::any_of(workerInvalid.begin(), workerInvalid.end(),
                                        [](const uint8_t v) { return v != 0u; });
                        if (anyInvalid) {
                            _outIndices.resize(oldSize);
                            _maxIndex = oldMaxIndex;
                            return EFastFaceReadResult::Error;
                        }
                        if (trackMaxIndex) {
                            for (const uint32_t local : workerMax)
                                if (local > _maxIndex)
                                    _maxIndex = local;
                        }
                        if (hashInParsePipeline &&
                            hashPipelineOk.load(std::memory_order_relaxed))
                            outIndexHash = parsedIndexHash;
                        StartPointer = reinterpret_cast<char*>(
                            const_cast<uint8_t*>(ptr + element.Count * recordBytes));
                        _faceCount += element.Count;
                        return EFastFaceReadResult::Success;
                    }
                }
                if (is32Bit)
                {
                    if (isSrcU32)
                    {
                        if (trackMaxIndex)
                        {
                            for (size_t j = 0u; j < element.Count; ++j)
                            {
                                const uint8_t c = *ptr++;
                                if (c != 3u)
                                    return EFastFaceReadResult::NotApplicable;
                                out[0] = readU32(ptr + 0ull * sizeof(uint32_t));
                                out[1] = readU32(ptr + 1ull * sizeof(uint32_t));
                                out[2] = readU32(ptr + 2ull * sizeof(uint32_t));
                                ptr += 3ull * sizeof(uint32_t);
                                if (out[0] > _maxIndex) _maxIndex = out[0];
                                if (out[1] > _maxIndex) _maxIndex = out[1];
                                if (out[2] > _maxIndex) _maxIndex = out[2];
                                out += 3u;
                            }
                        }
                        else
                        {
                            for (size_t j = 0u; j < element.Count; ++j)
                            {
                                const uint8_t c = *ptr++;
                                if (c != 3u)
                                    return EFastFaceReadResult::NotApplicable;
                                out[0] = readU32(ptr + 0ull * sizeof(uint32_t));
                                out[1] = readU32(ptr + 1ull * sizeof(uint32_t));
                                out[2] = readU32(ptr + 2ull * sizeof(uint32_t));
                                ptr += 3ull * sizeof(uint32_t);
                                if (out[0] >= vertexCount || out[1] >= vertexCount || out[2] >= vertexCount)
                                    return EFastFaceReadResult::Error;
                                out += 3u;
                            }
                        }
                    }
                    else
                    {
                        for (size_t j = 0u; j < element.Count; ++j)
                        {
                            const uint8_t c = *ptr++;
                            if (c != 3u)
                                return EFastFaceReadResult::NotApplicable;
                            out[0] = readU32(ptr + 0ull * sizeof(uint32_t));
                            out[1] = readU32(ptr + 1ull * sizeof(uint32_t));
                            out[2] = readU32(ptr + 2ull * sizeof(uint32_t));
                            ptr += 3ull * sizeof(uint32_t);
                            if ((out[0] | out[1] | out[2]) & 0x80000000u)
                                return EFastFaceReadResult::Error;
                            if (trackMaxIndex)
                            {
                                if (out[0] > _maxIndex) _maxIndex = out[0];
                                if (out[1] > _maxIndex) _maxIndex = out[1];
                                if (out[2] > _maxIndex) _maxIndex = out[2];
                            }
                            else if (out[0] >= vertexCount || out[1] >= vertexCount || out[2] >= vertexCount)
                                return EFastFaceReadResult::Error;
                            out += 3u;
                        }
                    }
                }
                else
                {
                    if (isSrcU16)
                    {
                        if (trackMaxIndex)
                        {
                            for (size_t j = 0u; j < element.Count; ++j)
                            {
                                const uint8_t c = *ptr++;
                                if (c != 3u)
                                    return EFastFaceReadResult::NotApplicable;
                                out[0] = readU16(ptr + 0ull * sizeof(uint16_t));
                                out[1] = readU16(ptr + 1ull * sizeof(uint16_t));
                                out[2] = readU16(ptr + 2ull * sizeof(uint16_t));
                                ptr += 3ull * sizeof(uint16_t);
                                if (out[0] > _maxIndex) _maxIndex = out[0];
                                if (out[1] > _maxIndex) _maxIndex = out[1];
                                if (out[2] > _maxIndex) _maxIndex = out[2];
                                out += 3u;
                            }
                        }
                        else
                        {
                            for (size_t j = 0u; j < element.Count; ++j)
                            {
                                const uint8_t c = *ptr++;
                                if (c != 3u)
                                    return EFastFaceReadResult::NotApplicable;
                                out[0] = readU16(ptr + 0ull * sizeof(uint16_t));
                                out[1] = readU16(ptr + 1ull * sizeof(uint16_t));
                                out[2] = readU16(ptr + 2ull * sizeof(uint16_t));
                                ptr += 3ull * sizeof(uint16_t);
                                if (out[0] >= vertexCount || out[1] >= vertexCount || out[2] >= vertexCount)
                                    return EFastFaceReadResult::Error;
                                out += 3u;
                            }
                        }
                    }
                    else
                    {
                        for (size_t j = 0u; j < element.Count; ++j)
                        {
                            const uint8_t c = *ptr++;
                            if (c != 3u)
                                return EFastFaceReadResult::NotApplicable;
                            out[0] = readU16(ptr + 0ull * sizeof(uint16_t));
                            out[1] = readU16(ptr + 1ull * sizeof(uint16_t));
                            out[2] = readU16(ptr + 2ull * sizeof(uint16_t));
                            ptr += 3ull * sizeof(uint16_t);
                            if ((out[0] | out[1] | out[2]) & 0x8000u)
                                return EFastFaceReadResult::Error;
                            if (trackMaxIndex)
                            {
                                if (out[0] > _maxIndex) _maxIndex = out[0];
                                if (out[1] > _maxIndex) _maxIndex = out[1];
                                if (out[2] > _maxIndex) _maxIndex = out[2];
                            }
                            else if (out[0] >= vertexCount || out[1] >= vertexCount || out[2] >= vertexCount)
                                return EFastFaceReadResult::Error;
                            out += 3u;
                        }
                    }
                }
                StartPointer = reinterpret_cast<char*>(const_cast<uint8_t*>(ptr));
                _faceCount += element.Count;
                return EFastFaceReadResult::Success;
            }
            if (element.Count > (std::numeric_limits<size_t>::max() / 3u))
                return EFastFaceReadResult::Error;
            const size_t reserveCount = element.Count * 3u;
            if (_outIndices.size() >
                (std::numeric_limits<size_t>::max() - reserveCount))
                return EFastFaceReadResult::Error;
            const size_t oldSize = _outIndices.size();
            _outIndices.resize(oldSize + reserveCount);
            uint32_t* out = _outIndices.data() + oldSize;
            size_t written = 0ull;
            auto ensureBytes = [this](const size_t bytes) -> bool {
                if (StartPointer + bytes > EndPointer)
                    fillBuffer();
                return StartPointer + bytes <= EndPointer;
            };
            auto readCount = [&ensureBytes, this](int32_t& outCount) -> bool {
                if (!ensureBytes(sizeof(uint8_t)))
                    return false;
                outCount = static_cast<uint8_t>(*StartPointer++);
                return true;
            };
            auto readIndex = [&ensureBytes, this, is32Bit, isSrcU32, isSrcU16,
                              needEndianSwap](uint32_t& out) -> bool {
                if (is32Bit) {
                    if (!ensureBytes(sizeof(uint32_t)))
                        return false;
                    if (isSrcU32) {
                        std::memcpy(&out, StartPointer, sizeof(uint32_t));
                        if (needEndianSwap)
                            out = Binary::byteswap(out);
                    } else {
                        int32_t v = 0;
                        std::memcpy(&v, StartPointer, sizeof(v));
                        if (needEndianSwap)
                            v = Binary::byteswap(v);
                        if (v < 0)
                            return false;
                        out = static_cast<uint32_t>(v);
                    }
                    StartPointer += sizeof(uint32_t);
                    return true;
                }
                if (!ensureBytes(sizeof(uint16_t)))
                    return false;
                if (isSrcU16) {
                    uint16_t v = 0u;
                    std::memcpy(&v, StartPointer, sizeof(uint16_t));
                    if (needEndianSwap)
                        v = Binary::byteswap(v);
                    out = v;
                } else {
                    int16_t v = 0;
                    std::memcpy(&v, StartPointer, sizeof(int16_t));
                    if (needEndianSwap)
                        v = Binary::byteswap(v);
                    if (v < 0)
                        return false;
                    out = static_cast<uint32_t>(v);
                }
                StartPointer += sizeof(uint16_t);
                return true;
            };
            auto readPackedU32 = [needEndianSwap](const uint8_t* src) -> uint32_t {
                uint32_t value = 0u;
                std::memcpy(&value, src, sizeof(value));
                if (needEndianSwap)
                    value = Binary::byteswap(value);
                return value;
            };
            auto readPackedU16 = [needEndianSwap](const uint8_t* src) -> uint32_t {
                uint16_t value = 0u;
                std::memcpy(&value, src, sizeof(value));
                if (needEndianSwap)
                    value = Binary::byteswap(value);
                return value;
            };
            for (size_t j = 0u; j < element.Count; ++j) {
                if (is32Bit && ensureBytes(sizeof(uint8_t) + sizeof(uint32_t) * 3ull) && static_cast<uint8_t>(*StartPointer) == 3u)
                {
                    ++StartPointer;
                    const uint32_t i0 = readPackedU32(reinterpret_cast<const uint8_t*>(StartPointer) + 0ull * sizeof(uint32_t));
                    const uint32_t i1 = readPackedU32(reinterpret_cast<const uint8_t*>(StartPointer) + 1ull * sizeof(uint32_t));
                    const uint32_t i2 = readPackedU32(reinterpret_cast<const uint8_t*>(StartPointer) + 2ull * sizeof(uint32_t));
                    StartPointer += 3ull * sizeof(uint32_t);
                    if (isSrcS32 && ((i0 | i1 | i2) & 0x80000000u))
                        return EFastFaceReadResult::Error;
                    if (trackMaxIndex)
                    {
                        if (i0 > _maxIndex) _maxIndex = i0;
                        if (i1 > _maxIndex) _maxIndex = i1;
                        if (i2 > _maxIndex) _maxIndex = i2;
                    }
                    else if (i0 >= vertexCount || i1 >= vertexCount || i2 >= vertexCount)
                        return EFastFaceReadResult::Error;
                    out[0] = i0;
                    out[1] = i1;
                    out[2] = i2;
                    out += 3u;
                    written += 3ull;
                    ++_faceCount;
                    continue;
                }
                if (!is32Bit && ensureBytes(sizeof(uint8_t) + sizeof(uint16_t) * 3ull) && static_cast<uint8_t>(*StartPointer) == 3u)
                {
                    ++StartPointer;
                    const uint32_t i0 = readPackedU16(reinterpret_cast<const uint8_t*>(StartPointer) + 0ull * sizeof(uint16_t));
                    const uint32_t i1 = readPackedU16(reinterpret_cast<const uint8_t*>(StartPointer) + 1ull * sizeof(uint16_t));
                    const uint32_t i2 = readPackedU16(reinterpret_cast<const uint8_t*>(StartPointer) + 2ull * sizeof(uint16_t));
                    StartPointer += 3ull * sizeof(uint16_t);
                    if (isSrcS16 && ((i0 | i1 | i2) & 0x8000u))
                        return EFastFaceReadResult::Error;
                    if (trackMaxIndex)
                    {
                        if (i0 > _maxIndex) _maxIndex = i0;
                        if (i1 > _maxIndex) _maxIndex = i1;
                        if (i2 > _maxIndex) _maxIndex = i2;
                    }
                    else if (i0 >= vertexCount || i1 >= vertexCount || i2 >= vertexCount)
                        return EFastFaceReadResult::Error;
                    out[0] = i0;
                    out[1] = i1;
                    out[2] = i2;
                    out += 3u;
                    written += 3ull;
                    ++_faceCount;
                    continue;
                }
                int32_t countSigned = 0;
                if (!readCount(countSigned))
                    return EFastFaceReadResult::Error;
                const uint32_t count = static_cast<uint32_t>(countSigned);
                if (count < 3u) {
                    uint32_t dummy = 0u;
                    for (uint32_t k = 0u; k < count; ++k) {
                        if (!readIndex(dummy))
                            return EFastFaceReadResult::Error;
                    }
                    ++_faceCount;
                    continue;
                }
                uint32_t i0 = 0u;
                uint32_t i1 = 0u;
                uint32_t i2 = 0u;
                if (!readIndex(i0) || !readIndex(i1) || !readIndex(i2))
                    return EFastFaceReadResult::Error;
                if (trackMaxIndex) {
                    _maxIndex = std::max(_maxIndex, std::max(i0, std::max(i1, i2)));
                } else if (i0 >= vertexCount || i1 >= vertexCount ||
                           i2 >= vertexCount) {
                    return EFastFaceReadResult::Error;
                }
                out[0] = i0;
                out[1] = i1;
                out[2] = i2;
                out += 3u;
                written += 3ull;
                uint32_t prev = i2;
                for (uint32_t k = 3u; k < count; ++k) {
                    uint32_t idx = 0u;
                    if (!readIndex(idx))
                        return EFastFaceReadResult::Error;
                    if (trackMaxIndex) {
                        _maxIndex = std::max(_maxIndex, idx);
                    } else if (idx >= vertexCount) {
                        return EFastFaceReadResult::Error;
                    }
                    if (_outIndices.size() < oldSize + written + 3ull)
                    {
                        const size_t outOffset = static_cast<size_t>(out - _outIndices.data());
                        _outIndices.resize(oldSize + written + 3ull);
                        out = _outIndices.data() + outOffset;
                    }
                    out[0] = i0;
                    out[1] = prev;
                    out[2] = idx;
                    out += 3u;
                    written += 3ull;
                    prev = idx;
                }
                ++_faceCount;
            }
            _outIndices.resize(oldSize + written);
            return EFastFaceReadResult::Success;
        }
        IAssetLoader::SAssetLoadContext inner;
        uint32_t topHierarchyLevel;
        IAssetLoader::IAssetLoaderOverride* loaderOverride;
        core::vector<char> Buffer; // input buffer must be at least twice as long as the longest line in the file
        size_t ioReadWindowSize = DefaultIoReadWindowBytes;
        core::vector<SElement> ElementList = {};
        char *StartPointer = nullptr, *EndPointer = nullptr,
             *LineEndPointer = nullptr;
        int32_t LineLength = 0;
        int32_t WordLength = -1; // this variable is a misnomer, its really the offset to next word minus one
        bool IsBinaryFile = false, IsWrongEndian = false, EndOfFile = false;
        size_t fileOffset = {};
        uint64_t readCallCount = 0ull;
        uint64_t readBytesTotal = 0ull;
        uint64_t readMinBytes = std::numeric_limits<uint64_t>::max();
        core::vector<SVertAttrIt> vertAttrIts;
    };
};
}
CPLYMeshFileLoader::CPLYMeshFileLoader() = default;
const char** CPLYMeshFileLoader::getAssociatedFileExtensions() const
{
	static const char* ext[] = { "ply", nullptr };
	return ext;
}
bool CPLYMeshFileLoader::isALoadableFileFormat(system::IFile* _file, const system::logger_opt_ptr) const {
    std::array<char, 128> buf = {};
    system::IFile::success_t success;
    _file->read(success, buf.data(), 0, buf.size());
    if (!success)
        return false;
    const std::string_view fileHeader(buf.data(), success.getBytesProcessed());
    size_t lineStart = 0ull;
    const size_t firstLineEnd = fileHeader.find('\n');
    std::string_view firstLine = fileHeader.substr(0ull, firstLineEnd);
    firstLine = Parse::Common::trimWhitespace(firstLine);
    if (firstLine != "ply")
        return false;
    if (firstLineEnd == std::string_view::npos)
        return false;
    lineStart = firstLineEnd + 1ull;
    constexpr std::array<std::string_view, 3> headers = {
        "format ascii 1.0", "format binary_little_endian 1.0",
        "format binary_big_endian 1.0"};
    while (lineStart < fileHeader.size()) {
        size_t lineEnd = fileHeader.find('\n', lineStart);
        if (lineEnd == std::string_view::npos)
            lineEnd = fileHeader.size();
        std::string_view line = Parse::Common::trimWhitespace(fileHeader.substr(lineStart, lineEnd - lineStart));
        if (line.starts_with("format "))
            return std::find(headers.begin(), headers.end(), line) != headers.end();
        lineStart = lineEnd + 1ull;
    }
    return false;
}
//! creates/loads an animated mesh from the file.
SAssetBundle CPLYMeshFileLoader::loadAsset(
    system::IFile* _file, const IAssetLoader::SAssetLoadParams& _params,
    IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel) {
    using namespace nbl::core;
    using clock_t = std::chrono::high_resolution_clock;
    if (!_file)
        return {};
    const bool computeContentHashes = !_params.loaderFlags.hasAnyFlag(
        IAssetLoader::ELPF_DONT_COMPUTE_CONTENT_HASHES);
    uint64_t faceCount = 0u;
    uint64_t fastFaceElementCount = 0u;
    uint64_t fastVertexElementCount = 0u;
    uint32_t maxIndexRead = 0u;
    core::blake3_hash_t precomputedIndexHash = IPreHashed::INVALID_HASH;
    const uint64_t fileSize = _file->getSize();
    const bool hashInBuild =
        computeContentHashes &&
        SLoaderRuntimeTuner::shouldInlineHashBuild(_params.ioPolicy, fileSize);
    impl::SLoadSession loadSession = {};
    if (!impl::SLoadSession::begin(_params.logger, "PLY loader", _file, _params.ioPolicy, fileSize, true, loadSession))
        return {};
    Parse::Context ctx = {asset::IAssetLoader::SAssetLoadContext{_params, _file},
                          _hierarchyLevel, _override};
    uint64_t desiredReadWindow =
        loadSession.isWholeFile()
            ? (fileSize + Parse::Context::ReadWindowPaddingBytes)
            : loadSession.ioPlan.chunkSizeBytes();
    if (loadSession.isWholeFile()) {
        const bool mappedInput = loadSession.mappedPointer() != nullptr;
        if (mappedInput &&
            fileSize > (Parse::Context::DefaultIoReadWindowBytes * 2ull))
            desiredReadWindow = Parse::Context::DefaultIoReadWindowBytes;
    }
    const uint64_t safeReadWindow = std::min<uint64_t>(desiredReadWindow, static_cast<uint64_t>(std::numeric_limits<size_t>::max() - Parse::Context::ReadWindowPaddingBytes));
    ctx.init(static_cast<size_t>(safeReadWindow));
    // start with empty mesh
    auto geometry = make_smart_refctd_ptr<ICPUPolygonGeometry>();
    hlsl::shapes::util::AABBAccumulator3<float> parsedAABB = hlsl::shapes::util::createAABBAccumulator<float>();
    uint32_t vertCount = 0;
    Parse::ContentHashBuild contentHashBuild = Parse::ContentHashBuild::create(computeContentHashes, hashInBuild);
    double headerMs = 0.0, vertexMs = 0.0, faceMs = 0.0, finalizeMs = 0.0;
    auto visitVertexAttributeViews = [&](auto&& visitor) -> void {
        visitor(geometry->getPositionView());
        visitor(geometry->getNormalView());
        for (const auto& view : *geometry->getAuxAttributeViews())
            visitor(view);
    };
    auto visitGeometryViews = [&](auto&& visitor) -> void {
        visitVertexAttributeViews(visitor);
        visitor(geometry->getIndexView());
        for (const auto& view : *geometry->getJointWeightViews()) {
            visitor(view.indices);
            visitor(view.weights);
        }
        if (const auto jointObb = geometry->getJointOBBView(); jointObb)
            visitor(*jointObb);
    };
    auto hashViewBufferIfNeeded = [&](const IGeometry<ICPUBuffer>::SDataView& view) -> void {
        if (!view || !view.src.buffer)
            return;
        contentHashBuild.hashNow(view.src.buffer.get());
    };
    auto hashRemainingGeometryBuffers = [&]() -> void {
        if (contentHashBuild.hashesInline())
            visitGeometryViews(hashViewBufferIfNeeded);
    };
    auto tryLaunchDeferredHash = [&](const IGeometry<ICPUBuffer>::SDataView& view) -> void {
        if (!view || !view.src.buffer)
            return;
        contentHashBuild.tryDefer(view.src.buffer.get());
    };
    // Currently only supports ASCII or binary meshes
    if (Parse::toStringView(ctx.getNextLine()) != "ply") {
        _params.logger.log("Not a valid PLY file %s", system::ILogger::ELL_ERROR,
                           ctx.inner.mainFile->getFileName().string().c_str());
        return {};
    }
    // cut the next line out
    ctx.getNextLine();
    // grab the word from this line
    const char* word = ctx.getNextWord();
    // ignore comments
    for (; Parse::toStringView(word) == "comment"; ctx.getNextLine())
        word = ctx.getNextWord();
    bool readingHeader = true;
    bool continueReading = true;
    ctx.IsBinaryFile = false;
    ctx.IsWrongEndian = false;
    const auto headerStart = clock_t::now();
    do {
        const std::string_view wordView = Parse::toStringView(word);
        if (wordView == "property") {
            word = ctx.getNextWord();
            if (ctx.ElementList.empty()) {
                _params.logger.log("PLY property token found before element %s",
                                   system::ILogger::ELL_WARNING, word);
            } else {
                // get element
                auto& el = ctx.ElementList.back();
                // fill property struct
                auto& prop = el.Properties.emplace_back();
                prop.type = prop.getType(word);
                if (prop.type == EF_UNKNOWN) {
                    el.KnownSize = false;
                    word = ctx.getNextWord();
                    prop.list.countType = prop.getType(word);
                    if (ctx.IsBinaryFile && !isIntegerFormat(prop.list.countType)) {
                        _params.logger.log("Cannot read binary PLY file containing data "
                                           "types of unknown or non integer length %s",
                                           system::ILogger::ELL_WARNING, word);
                        continueReading = false;
                    } else {
                        word = ctx.getNextWord();
                        prop.list.itemType = prop.getType(word);
                        if (ctx.IsBinaryFile && !isIntegerFormat(prop.list.itemType)) {
                            _params.logger.log("Cannot read binary PLY file containing data "
                                               "types of unknown or non integer length %s",
                                               system::ILogger::ELL_ERROR, word);
                            continueReading = false;
                        }
                    }
                } else if (ctx.IsBinaryFile && prop.type == EF_UNKNOWN) {
                    _params.logger.log("Cannot read binary PLY file containing data "
                                       "types of unknown length %s",
                                       system::ILogger::ELL_ERROR, word);
                    continueReading = false;
                } else
                    el.KnownSize += getTexelOrBlockBytesize(prop.type);
                prop.Name = ctx.getNextWord();
            }
        } else if (wordView == "element") {
            auto& el = ctx.ElementList.emplace_back();
            el.Name = ctx.getNextWord();
            const char* const countWord = ctx.getNextWord();
            uint64_t parsedCount = 0ull;
            const std::string_view countWordView = Parse::toStringView(countWord);
            if (!countWordView.empty()) {
                if (!Parse::Common::parseExactNumber(countWordView, parsedCount))
                    parsedCount = 0ull;
            }
            el.Count = static_cast<size_t>(parsedCount);
            el.KnownSize = 0;
            if (el.Name == "vertex")
                vertCount = el.Count;
        } else if (wordView == "comment") {
            // ignore line
        } else if (wordView == "format") {
            // must be `format {binary_little_endian|binary_big_endian|ascii} 1.0`
            word = ctx.getNextWord();
            const std::string_view formatView = Parse::toStringView(word);
            if (formatView == "binary_little_endian") {
                ctx.IsBinaryFile = true;
            } else if (formatView == "binary_big_endian") {
                ctx.IsBinaryFile = true;
                ctx.IsWrongEndian = true;
            } else if (formatView == "ascii") {
            } else {
                // abort if this isn't an ascii or a binary mesh
                _params.logger.log("Unsupported PLY mesh format %s",
                                   system::ILogger::ELL_ERROR, word);
                continueReading = false;
            }
            if (continueReading) {
                word = ctx.getNextWord();
                if (Parse::toStringView(word) != "1.0") {
                    _params.logger.log("Unsupported PLY mesh version %s",
                                       system::ILogger::ELL_WARNING, word);
                }
            }
        } else if (wordView == "end_header") {
            readingHeader = false;
            if (ctx.IsBinaryFile) {
                char* const binaryStartInBuffer = ctx.LineEndPointer + 1;
                const auto* const mappedBase = reinterpret_cast<const char*>(loadSession.mappedPointer());
                if (mappedBase) {
                    const size_t binaryOffset =
                        ctx.getAbsoluteOffset(binaryStartInBuffer);
                    const size_t remainingBytes = static_cast<size_t>(
                        binaryOffset < fileSize ? (fileSize - binaryOffset) : 0ull);
                    ctx.useMappedBinaryWindow(mappedBase + binaryOffset, remainingBytes);
                } else {
                    ctx.StartPointer = binaryStartInBuffer;
                }
            }
        } else {
            _params.logger.log("Unknown item in PLY file %s",
                               system::ILogger::ELL_WARNING, word);
        }
        if (readingHeader && continueReading) {
            ctx.getNextLine();
            word = ctx.getNextWord();
        }
    } while (readingHeader && continueReading);
    headerMs = std::chrono::duration<double, std::milli>(clock_t::now() - headerStart).count();
    if (!continueReading)
        return {};
    // now to read the actual data from the file
    using index_t = uint32_t;
    core::vector<index_t> indices = {};
    bool verticesProcessed = false;
    const std::string fileName = _file->getFileName().string();
    auto logMalformedElement = [&](const char* const elementName) -> void {
        _params.logger.log("PLY %s fast path failed on malformed data for %s", system::ILogger::ELL_ERROR, elementName, fileName.c_str());
    };
    auto skipUnknownElement = [&](const Parse::Context::SElement& el) -> bool {
        if (ctx.IsBinaryFile && el.KnownSize) {
            const uint64_t bytesToSkip64 = static_cast<uint64_t>(el.KnownSize) * static_cast<uint64_t>(el.Count);
            if (bytesToSkip64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
                return false;
            ctx.moveForward(static_cast<size_t>(bytesToSkip64));
        } else {
            for (size_t j = 0; j < el.Count; ++j)
                el.skipElement(ctx);
        }
        return true;
    };
    auto readFaceElement = [&](const Parse::Context::SElement& el) -> bool {
        const uint32_t vertexCount32 = vertCount <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ? static_cast<uint32_t>(vertCount) : 0u;
        const auto fastFaceResult = ctx.readFaceElementFast(el, indices, maxIndexRead, faceCount, vertexCount32, contentHashBuild.hashesDeferred(), precomputedIndexHash);
        if (fastFaceResult == Parse::Context::EFastFaceReadResult::Success) {
            ++fastFaceElementCount;
            return true;
        }
        if (fastFaceResult == Parse::Context::EFastFaceReadResult::NotApplicable) {
            indices.reserve(indices.size() + el.Count * 3u);
            for (size_t j = 0; j < el.Count; ++j) {
                if (!ctx.readFace(el, indices, maxIndexRead, vertexCount32))
                    return false;
                ++faceCount;
            }
            return true;
        }
        logMalformedElement("face");
        return false;
    };
    // loop through each of the elements
    for (uint32_t i = 0; i < ctx.ElementList.size(); ++i) {
        auto& el = ctx.ElementList[i];
        if (el.Name == "vertex") {
            const auto vertexStart = clock_t::now();
            if (verticesProcessed) {
                // multiple vertex elements are currently treated as unsupported
                _params.logger.log("Multiple `vertex` elements not supported!",
                                   system::ILogger::ELL_ERROR);
                return {};
            }
            ICPUPolygonGeometry::SDataViewBase posView = {}, normalView = {},
                                               uvView = {};
            core::vector<ICPUPolygonGeometry::SDataView> extraViews;
            for (auto& vertexProperty : el.Properties) {
                const auto& propertyName = vertexProperty.Name;
                // only positions and normals need to be structured/canonicalized in any way
                auto negotiateFormat = [&vertexProperty](ICPUPolygonGeometry::SDataViewBase& view, const uint8_t component) -> void {
                    assert(getFormatChannelCount(vertexProperty.type) != 0);
                    if (getTexelOrBlockBytesize(vertexProperty.type) > getTexelOrBlockBytesize(view.format))
                        view.format = vertexProperty.type;
                    view.stride = hlsl::max<uint32_t>(view.stride, component);
                };
                if (propertyName == "x")
                    negotiateFormat(posView, 0);
                else if (propertyName == "y")
                    negotiateFormat(posView, 1);
                else if (propertyName == "z")
                    negotiateFormat(posView, 2);
                else if (propertyName == "nx")
                    negotiateFormat(normalView, 0);
                else if (propertyName == "ny")
                    negotiateFormat(normalView, 1);
                else if (propertyName == "nz")
                    negotiateFormat(normalView, 2);
                else if (propertyName == "u" || propertyName == "s")
                    negotiateFormat(uvView, 0);
                else if (propertyName == "v" || propertyName == "t")
                    negotiateFormat(uvView, 1);
                else
                    // property names for extra channels are currently not persisted in metadata
                    extraViews.push_back(createView(vertexProperty.type, el.Count));
            }
            auto setFinalFormat = [&ctx](ICPUPolygonGeometry::SDataViewBase& view) -> void {
                const auto componentFormat = view.format;
                const auto componentCount = view.stride + 1;
                view.format = Parse::expandStructuredFormat(view.format, componentCount);
                view.stride = getTexelOrBlockBytesize(view.format);
                for (auto c = 0u; c < componentCount; c++) {
                    size_t offset = getTexelOrBlockBytesize(componentFormat) * c;
                    ctx.vertAttrIts.push_back({.ptr = reinterpret_cast<uint8_t*>(offset),
                                               .stride = view.stride,
                                               .dstFmt = componentFormat});
                }
            };
            auto attachStructuredView = [&](ICPUPolygonGeometry::SDataViewBase& baseView, auto&& setter) -> void {
                if (baseView.format == EF_UNKNOWN)
                    return;
                auto beginIx = ctx.vertAttrIts.size();
                setFinalFormat(baseView);
                auto view = createView(baseView.format, el.Count);
                for (const auto size = ctx.vertAttrIts.size(); beginIx != size; ++beginIx)
                    ctx.vertAttrIts[beginIx].ptr += ptrdiff_t(view.src.buffer->getPointer()) + view.src.offset;
                setter(std::move(view));
            };
            attachStructuredView(posView, [&](auto view) { geometry->setPositionView(std::move(view)); });
            attachStructuredView(normalView, [&](auto view) { geometry->setNormalView(std::move(view)); });
            attachStructuredView(uvView, [&](auto view) {
                auto* const auxViews = geometry->getAuxAttributeViews();
                auxViews->resize(Parse::UV0 + 1u);
                auxViews->operator[](Parse::UV0) = std::move(view);
            });
            for (auto& view : extraViews)
                ctx.vertAttrIts.push_back({.ptr = reinterpret_cast<uint8_t*>(view.src.buffer->getPointer()) + view.src.offset,
                                           .stride = getTexelOrBlockBytesize(view.composed.format),
                                           .dstFmt = view.composed.format});
            for (auto& view : extraViews)
                geometry->getAuxAttributeViews()->push_back(std::move(view));
            // loop through vertex properties
            const auto fastVertexResult = ctx.readVertexElementFast(el, &parsedAABB);
            if (fastVertexResult == Parse::Context::EFastVertexReadResult::Success) {
                ++fastVertexElementCount;
            } else if (fastVertexResult ==
                       Parse::Context::EFastVertexReadResult::NotApplicable) {
                ctx.readVertex(_params, el);
            } else {
                logMalformedElement("vertex");
                return {};
            }
            visitVertexAttributeViews(hashViewBufferIfNeeded);
            tryLaunchDeferredHash(geometry->getPositionView());
            verticesProcessed = true;
            vertexMs += std::chrono::duration<double, std::milli>(clock_t::now() - vertexStart).count();
        } else if (el.Name == "face") {
            const auto faceStart = clock_t::now();
            if (!readFaceElement(el))
                return {};
            faceMs += std::chrono::duration<double, std::milli>(clock_t::now() - faceStart).count();
        } else {
            if (!skipUnknownElement(el))
                return {};
        }
    }
    if (!parsedAABB.empty())
        geometry->applyAABB(parsedAABB.value);
    else
        CPolygonGeometryManipulator::recomputeAABB(geometry.get());
    const uint64_t indexCount = static_cast<uint64_t>(indices.size());
    if (indices.empty()) {
        // no index buffer means point cloud
        geometry->setIndexing(IPolygonGeometryBase::PointList());
    } else {
        if (vertCount != 0u && maxIndexRead >= vertCount) {
            _params.logger.log("PLY indices out of range for %s",
                               system::ILogger::ELL_ERROR,
                               _file->getFileName().string().c_str());
            return {};
        }
        geometry->setIndexing(IPolygonGeometryBase::TriangleList());
        const bool canUseU16 =
            (vertCount != 0u)
                ? (vertCount <= std::numeric_limits<uint16_t>::max())
                : (maxIndexRead <= std::numeric_limits<uint16_t>::max());
        if (canUseU16) {
            core::vector<uint16_t> indices16(indices.size());
            for (size_t i = 0u; i < indices.size(); ++i)
                indices16[i] = static_cast<uint16_t>(indices[i]);
            auto view = SGeometryLoaderCommon::createAdoptedView<EF_R16_UINT>(
                std::move(indices16));
            if (!view)
                return {};
            geometry->setIndexView(std::move(view));
            hashViewBufferIfNeeded(geometry->getIndexView());
        } else {
            auto view = SGeometryLoaderCommon::createAdoptedView<EF_R32_UINT>(
                std::move(indices));
            if (!view)
                return {};
            if (precomputedIndexHash != IPreHashed::INVALID_HASH)
                view.src.buffer->setContentHash(precomputedIndexHash);
            geometry->setIndexView(std::move(view));
            hashViewBufferIfNeeded(geometry->getIndexView());
        }
    }
    const auto finalizeStart = clock_t::now();
    if (contentHashBuild.hashesDeferred()) {
        contentHashBuild.wait();
        SPolygonGeometryContentHash::computeMissing(geometry.get(),
                                                    _params.ioPolicy);
    } else {
        hashRemainingGeometryBuffers();
    }
    finalizeMs = std::chrono::duration<double, std::milli>(clock_t::now() - finalizeStart).count();
    const uint64_t ioMinRead = ctx.readCallCount ? ctx.readMinBytes : 0ull;
    const uint64_t ioAvgRead =
        ctx.readCallCount ? (ctx.readBytesTotal / ctx.readCallCount) : 0ull;
    const SFileReadTelemetry ioTelemetry = {.callCount = ctx.readCallCount,
                                            .totalBytes = ctx.readBytesTotal,
                                            .minBytes = ctx.readMinBytes};
    loadSession.logTinyIO(_params.logger, ioTelemetry);
    _params.logger.log(
        "PLY loader stats: file=%s binary=%d verts=%llu faces=%llu idx=%llu "
        "vertex_fast=%llu face_fast=%llu io_reads=%llu io_min_read=%llu "
        "io_avg_read=%llu io_req=%s io_eff=%s io_chunk=%llu io_reason=%s",
        system::ILogger::ELL_PERFORMANCE, _file->getFileName().string().c_str(),
        ctx.IsBinaryFile ? 1 : 0, static_cast<unsigned long long>(vertCount),
        static_cast<unsigned long long>(faceCount),
        static_cast<unsigned long long>(indexCount),
        static_cast<unsigned long long>(fastVertexElementCount),
        static_cast<unsigned long long>(fastFaceElementCount),
        static_cast<unsigned long long>(ctx.readCallCount),
        static_cast<unsigned long long>(ioMinRead),
        static_cast<unsigned long long>(ioAvgRead),
        system::to_string(_params.ioPolicy.strategy).c_str(),
        system::to_string(loadSession.ioPlan.strategy).c_str(),
        static_cast<unsigned long long>(loadSession.ioPlan.chunkSizeBytes()), loadSession.ioPlan.reason);
    _params.logger.log("PLY loader stages: file=%s header=%.3f ms vertex=%.3f ms face=%.3f ms finalize=%.3f ms", system::ILogger::ELL_PERFORMANCE, _file->getFileName().string().c_str(), headerMs, vertexMs, faceMs, finalizeMs);
    auto meta = core::make_smart_refctd_ptr<CPLYMetadata>();
    return SAssetBundle(std::move(meta), {std::move(geometry)});
}
}
#endif // _NBL_COMPILE_WITH_PLY_LOADER_
