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
			std::string Name;
			core::vector<SProperty> Properties;
			size_t Count;
			uint32_t KnownSize;
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
		void fillBuffer()
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
			if (length && StartPointer != newStart)
				memmove(newStart, StartPointer, length);
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
			inner.mainFile->read(success, EndPointer, fileOffset, requestSize);
			const size_t bytesRead = success.getBytesProcessed();
			++readCallCount;
			readBytesTotal += bytesRead;
			if (bytesRead < readMinBytes)
				readMinBytes = bytesRead;
			fileOffset += bytesRead;
			EndPointer += bytesRead;
			if (bytesRead != requestSize)
			{
				*EndPointer = 0;
				EndOfFile = true;
			}
		}
		const char* getNextLine()
		{
			StartPointer = LineEndPointer + 1;
			if (*StartPointer == '\n')
				*(StartPointer++) = '\0';
			const std::array<const char, 3> Terminators = {'\0', '\r', '\n'};
			auto terminator = std::find_first_of(StartPointer, EndPointer, Terminators.begin(), Terminators.end());
			if (terminator != EndPointer)
				*(terminator++) = '\0';
			if (terminator == EndPointer)
			{
				if (EndOfFile)
				{
					StartPointer = EndPointer - 1;
					*StartPointer = '\0';
					return StartPointer;
				}
				fillBuffer();
				LineEndPointer = StartPointer - 1;
				return StartPointer != EndPointer ? getNextLine() : StartPointer;
			}
			LineEndPointer = terminator - 1;
			WordLength = -1;
			return StartPointer;
		}
		const char* getNextWord()
		{
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
			const std::array<const char, 3> WhiteSpace = {'\0', ' ', '\t'};
			auto wordEnd = std::find_first_of(StartPointer, LineEndPointer, WhiteSpace.begin(), WhiteSpace.end());
			if (wordEnd != LineEndPointer)
				*(wordEnd++) = '\0';
			auto nextWord = std::find_if(wordEnd, LineEndPointer, [WhiteSpace](const char c) -> bool { return std::find(WhiteSpace.begin(), WhiteSpace.end(), c) == WhiteSpace.end(); });
			WordLength = std::distance(StartPointer, nextWord) - 1;
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
		void moveForward(const size_t bytes)
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
		widest_int_t getInt(const E_FORMAT f)
		{
			assert(!isFloatingPointFormat(f));
			if (IsBinaryFile)
			{
				if (StartPointer + sizeof(widest_int_t) > EndPointer)
					fillBuffer();
				switch (getTexelOrBlockBytesize(f))
				{
					case 1:
						if (StartPointer + sizeof(int8_t) <= EndPointer)
							return *(StartPointer++);
						break;
					case 2:
						if (StartPointer + sizeof(int16_t) <= EndPointer)
						{
							const auto retval = Binary::loadUnaligned<int16_t>(StartPointer, IsWrongEndian);
							StartPointer += sizeof(int16_t);
							return retval;
						}
						break;
					case 4:
						if (StartPointer + sizeof(int32_t) <= EndPointer)
						{
							const auto retval = Binary::loadUnaligned<int32_t>(StartPointer, IsWrongEndian);
							StartPointer += sizeof(int32_t);
							return retval;
						}
						break;
					default:
						assert(false);
						break;
				}
				return 0u;
			}
			const char* word = getNextWord();
			if (!word)
				return 0u;
			const char* const wordEnd = getCurrentWordEnd(word);
			if (word == wordEnd)
				return 0u;
			auto parseInt = [&](auto& value) -> widest_int_t
			{
				auto ptr = word;
				if (Common::parseNumber(ptr, wordEnd, value) && ptr == wordEnd)
					return static_cast<widest_int_t>(value);
				return ptr != word ? static_cast<widest_int_t>(value) : 0u;
			};
			if (isSignedFormat(f))
			{
				int64_t value = 0;
				return parseInt(value);
			}
			uint64_t value = 0u;
			return parseInt(value);
		}
		hlsl::float64_t getFloat(const E_FORMAT f)
		{
			assert(isFloatingPointFormat(f));
			if (IsBinaryFile)
			{
				if (StartPointer + sizeof(hlsl::float64_t) > EndPointer)
					fillBuffer();
				switch (getTexelOrBlockBytesize(f))
				{
					case 4:
						if (StartPointer + sizeof(hlsl::float32_t) <= EndPointer)
						{
							const auto retval = Binary::loadUnaligned<hlsl::float32_t>(StartPointer, IsWrongEndian);
							StartPointer += sizeof(hlsl::float32_t);
							return retval;
						}
						break;
					case 8:
						if (StartPointer + sizeof(hlsl::float64_t) <= EndPointer)
						{
							const auto retval = Binary::loadUnaligned<hlsl::float64_t>(StartPointer, IsWrongEndian);
							StartPointer += sizeof(hlsl::float64_t);
							return retval;
						}
						break;
					default:
						assert(false);
						break;
				}
				return 0.0;
			}
			const char* word = getNextWord();
			if (!word)
				return 0.0;
			const char* const wordEnd = getCurrentWordEnd(word);
			if (word == wordEnd)
				return 0.0;
			hlsl::float64_t value = 0.0;
			auto ptr = word;
			if (Common::parseNumber(ptr, wordEnd, value) && ptr == wordEnd)
				return value;
			return ptr != word ? value : 0.0;
		}
		void getData(void* dst, const E_FORMAT f)
		{
			const auto size = getTexelOrBlockBytesize(f);
			if (StartPointer + size > EndPointer)
			{
				fillBuffer();
				if (StartPointer + size > EndPointer)
					return;
			}
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
            struct SLayoutDesc {
                uint32_t propertyCount;
                uint32_t srcBytesPerVertex;
                bool hasNormals;
                bool hasUVs;
            };
            auto allF32 = [&el]() -> bool {
                for (const auto& prop : el.Properties) {
                    if (prop.type != EF_R32_SFLOAT)
                        return false;
                }
                return true;
            };
            if (!allF32())
                return EFastVertexReadResult::NotApplicable;
            auto matchNames =
                [&el](std::initializer_list<const char*> names) -> bool {
                if (el.Properties.size() != names.size())
                    return false;
                size_t i = 0ull;
                for (const auto* name : names) {
                    if (el.Properties[i].Name != name)
                        return false;
                    ++i;
                }
                return true;
            };
            static constexpr SLayoutDesc xyz = {3u, sizeof(hlsl::float32_t) * 3u,
                                                false, false};
            static constexpr SLayoutDesc xyz_n = {6u, sizeof(hlsl::float32_t) * 6u,
                                                  true, false};
            static constexpr SLayoutDesc xyz_n_uv = {8u, sizeof(hlsl::float32_t) * 8u,
                                                     true, true};
            const SLayoutDesc* layout = nullptr;
            if (matchNames({"x", "y", "z"}))
                layout = &xyz;
            else if (matchNames({"x", "y", "z", "nx", "ny", "nz"}))
                layout = &xyz_n;
            else if (matchNames({"x", "y", "z", "nx", "ny", "nz", "u", "v"}) ||
                     matchNames({"x", "y", "z", "nx", "ny", "nz", "s", "t"}))
                layout = &xyz_n_uv;
            if (!layout)
                return EFastVertexReadResult::NotApplicable;
            const size_t floatBytes = sizeof(hlsl::float32_t);
            struct STupleDesc {
                uint32_t beginIx;
                uint32_t componentCount;
                uint32_t stride = 0u;
                uint8_t* base = nullptr;
            };
            std::array<STupleDesc, 3> tuples = {STupleDesc{0u, 3u},
                                                STupleDesc{3u, 3u},
                                                STupleDesc{6u, 2u}};
            const uint32_t tupleCount =
                1u + static_cast<uint32_t>(layout->hasNormals) +
                static_cast<uint32_t>(layout->hasUVs);
            auto validateTuple = [&](STupleDesc& tuple) -> bool {
                if (tuple.beginIx + tuple.componentCount > vertAttrIts.size())
                    return false;
                auto& first = vertAttrIts[tuple.beginIx];
                if (!first.ptr || first.dstFmt != EF_R32_SFLOAT)
                    return false;
                tuple.stride = first.stride;
                tuple.base = first.ptr;
                for (uint32_t c = 1u; c < tuple.componentCount; ++c) {
                    auto& it = vertAttrIts[tuple.beginIx + c];
                    if (!it.ptr || it.dstFmt != EF_R32_SFLOAT)
                        return false;
                    if (it.stride != tuple.stride)
                        return false;
                    if (it.ptr != tuple.base + c * floatBytes)
                        return false;
                }
                return true;
            };
            auto commitTuple = [&](const STupleDesc& tuple) -> void {
                for (uint32_t c = 0u; c < tuple.componentCount; ++c)
                    vertAttrIts[tuple.beginIx + c].ptr = tuple.base + c * floatBytes;
            };
            if (vertAttrIts.size() != layout->propertyCount)
                return EFastVertexReadResult::NotApplicable;
            for (uint32_t tupleIx = 0u; tupleIx < tupleCount; ++tupleIx)
                if (!validateTuple(tuples[tupleIx]))
                    return EFastVertexReadResult::NotApplicable;
            if (el.Count >
                (std::numeric_limits<size_t>::max() / layout->srcBytesPerVertex))
                return EFastVertexReadResult::Error;
            const bool trackAABB = parsedAABB != nullptr;
            const bool needsByteSwap = IsWrongEndian;
            auto decodeF32 = [needsByteSwap](const uint8_t* src) -> float {
                uint32_t bits = 0u;
                std::memcpy(&bits, src, sizeof(bits));
                if (needsByteSwap)
                    bits = Binary::byteswap(bits);
                float value = 0.f;
                std::memcpy(&value, &bits, sizeof(value));
                return value;
            };
            auto decodeVector = [&]<typename Vec>(const uint8_t* src) -> Vec {
                constexpr uint32_t N = hlsl::vector_traits<Vec>::Dimension;
                Vec value{};
                hlsl::array_set<Vec, float> setter;
                for (uint32_t i = 0u; i < N; ++i)
                    setter(value, i,
                           decodeF32(src + static_cast<size_t>(i) * floatBytes));
                return value;
            };
            auto storeVector = []<typename Vec>(uint8_t* dst,
                                                const Vec& value) -> void {
                constexpr uint32_t N = hlsl::vector_traits<Vec>::Dimension;
                hlsl::array_get<Vec, float> getter;
                auto* const out = reinterpret_cast<float*>(dst);
                for (uint32_t i = 0u; i < N; ++i)
                    out[i] = getter(value, i);
            };
            auto decodeStore = [&]<typename Vec>(STupleDesc& tuple,
                                                 const uint8_t*& src) -> Vec {
                Vec value = decodeVector.operator()<Vec>(src);
                storeVector.operator()<Vec>(tuple.base, value);
                src += static_cast<size_t>(hlsl::vector_traits<Vec>::Dimension) *
                       floatBytes;
                tuple.base += tuple.stride;
                return value;
            };
            size_t remainingVertices = el.Count;
            while (remainingVertices > 0ull) {
                if (StartPointer + layout->srcBytesPerVertex > EndPointer)
                    fillBuffer();
                const size_t available =
                    EndPointer > StartPointer
                        ? static_cast<size_t>(EndPointer - StartPointer)
                        : 0ull;
                if (available < layout->srcBytesPerVertex)
                    return EFastVertexReadResult::Error;
                const size_t batchVertices =
                    std::min(remainingVertices, available / layout->srcBytesPerVertex);
                const uint8_t* src = reinterpret_cast<const uint8_t*>(StartPointer);
                if (!layout->hasNormals && !layout->hasUVs &&
                    tuples[0].stride == 3ull * floatBytes && !needsByteSwap &&
                    !trackAABB) {
                    const size_t batchBytes = batchVertices * 3ull * floatBytes;
                    std::memcpy(tuples[0].base, src, batchBytes);
                    src += batchBytes;
                    tuples[0].base += batchBytes;
                } else {
                    for (size_t v = 0ull; v < batchVertices; ++v) {
                        const hlsl::float32_t3 position =
                            decodeStore.operator()<hlsl::float32_t3>(tuples[0], src);
                        if (trackAABB)
                            hlsl::shapes::util::extendAABBAccumulator(*parsedAABB, position);
                        if (layout->hasNormals) {
                            decodeStore.operator()<hlsl::float32_t3>(tuples[1], src);
                        }
                        if (layout->hasUVs) {
                            decodeStore.operator()<hlsl::float32_t2>(tuples[2], src);
                        }
                    }
                }
                const size_t consumed = batchVertices * layout->srcBytesPerVertex;
                StartPointer += consumed;
                remainingVertices -= batchVertices;
            }
            for (uint32_t tupleIx = 0u; tupleIx < tupleCount; ++tupleIx)
                commitTuple(tuples[tupleIx]);
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
                    if (IsBinaryFile && !IsWrongEndian && srcIndexFmt == EF_R32_UINT) {
                        const size_t bytesNeeded =
                            static_cast<size_t>(count) * sizeof(uint32_t);
                        if (StartPointer + bytesNeeded > EndPointer)
                            fillBuffer();
                        if (StartPointer + bytesNeeded <= EndPointer) {
                            const uint8_t* ptr =
                                reinterpret_cast<const uint8_t*>(StartPointer);
                            auto readIndex = [&ptr]() -> uint32_t {
                                uint32_t v = 0u;
                                std::memcpy(&v, ptr, sizeof(v));
                                ptr += sizeof(v);
                                return v;
                            };
                            if (!emitFan(readIndex, count))
                                return false;
                            StartPointer =
                                reinterpret_cast<char*>(const_cast<uint8_t*>(ptr));
                            continue;
                        }
                    } else if (IsBinaryFile && !IsWrongEndian &&
                               srcIndexFmt == EF_R16_UINT) {
                        const size_t bytesNeeded =
                            static_cast<size_t>(count) * sizeof(uint16_t);
                        if (StartPointer + bytesNeeded > EndPointer)
                            fillBuffer();
                        if (StartPointer + bytesNeeded <= EndPointer) {
                            const uint8_t* ptr =
                                reinterpret_cast<const uint8_t*>(StartPointer);
                            auto readIndex = [&ptr]() -> uint32_t {
                                uint16_t v = 0u;
                                std::memcpy(&v, ptr, sizeof(v));
                                ptr += sizeof(v);
                                return static_cast<uint32_t>(v);
                            };
                            if (!emitFan(readIndex, count))
                                return false;
                            StartPointer =
                                reinterpret_cast<char*>(const_cast<uint8_t*>(ptr));
                            continue;
                        }
                    }
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
            const hlsl::uint32_t3 vertexLimit(vertexCount);
            const auto triExceedsVertexLimit =
                [&vertexLimit](const hlsl::uint32_t3& tri) -> bool {
                return hlsl::any(glm::greaterThanEqual(tri, vertexLimit));
            };
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
                bool fallbackToGeneric = false;
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
                                const hlsl::uint32_t3 tri(
                                    readU32(in + 0ull * sizeof(uint32_t)),
                                    readU32(in + 1ull * sizeof(uint32_t)),
                                    readU32(in + 2ull * sizeof(uint32_t)));
                                outLocal[0] = tri.x;
                                outLocal[1] = tri.y;
                                outLocal[2] = tri.z;
                                const uint32_t triOr = tri.x | tri.y | tri.z;
                                if (isSrcS32 && (triOr & 0x80000000u)) {
                                    workerInvalid[workerIx] = 1u;
                                    if (hashInParsePipeline)
                                        workerHashable[workerIx] = 0u;
                                    break;
                                }
                                if (validateAgainstVertexCount) {
                                    if (triExceedsVertexLimit(tri)) {
                                        workerInvalid[workerIx] = 1u;
                                        if (hashInParsePipeline)
                                            workerHashable[workerIx] = 0u;
                                        break;
                                    }
                                } else if (needMax) {
                                    const uint32_t triMax = std::max({tri.x, tri.y, tri.z});
                                    if (triMax > localMax)
                                        localMax = triMax;
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
                auto consumeTriangles = [&](const size_t indexBytes, const uint32_t signedMask, auto readTri) -> EFastFaceReadResult {
                    for (size_t j = 0u; j < element.Count; ++j) {
                        if (*ptr++ != 3u) {
                            fallbackToGeneric = true;
                            return EFastFaceReadResult::NotApplicable;
                        }
                        const hlsl::uint32_t3 tri = readTri(ptr);
                        ptr += 3ull * indexBytes;
                        const uint32_t triOr = tri.x | tri.y | tri.z;
                        if (signedMask && (triOr & signedMask))
                            return EFastFaceReadResult::Error;
                        out[0] = tri.x;
                        out[1] = tri.y;
                        out[2] = tri.z;
                        if (trackMaxIndex) {
                            const uint32_t triMax = std::max({tri.x, tri.y, tri.z});
                            if (triMax > _maxIndex)
                                _maxIndex = triMax;
                        } else if (triExceedsVertexLimit(tri))
                            return EFastFaceReadResult::Error;
                        out += 3u;
                    }
                    return EFastFaceReadResult::Success;
                };
                const auto fastReadResult = is32Bit ?
                    consumeTriangles(sizeof(uint32_t), isSrcS32 ? 0x80000000u : 0u,
                                     [&](const uint8_t* const src) -> hlsl::uint32_t3 {
                                         return hlsl::uint32_t3(readU32(src + 0ull * sizeof(uint32_t)),
                                                                readU32(src + 1ull * sizeof(uint32_t)),
                                                                readU32(src + 2ull * sizeof(uint32_t)));
                                     }) :
                    consumeTriangles(sizeof(uint16_t), isSrcS16 ? 0x8000u : 0u,
                                     [&](const uint8_t* const src) -> hlsl::uint32_t3 {
                                         return hlsl::uint32_t3(readU16(src + 0ull * sizeof(uint16_t)),
                                                                readU16(src + 1ull * sizeof(uint16_t)),
                                                                readU16(src + 2ull * sizeof(uint16_t)));
                                     });
                if (fastReadResult == EFastFaceReadResult::Error)
                    return EFastFaceReadResult::Error;
                if (!fallbackToGeneric) {
                    StartPointer = reinterpret_cast<char*>(const_cast<uint8_t*>(ptr));
                    _faceCount += element.Count;
                    return EFastFaceReadResult::Success;
                }
                _outIndices.resize(oldSize);
                _maxIndex = oldMaxIndex;
            }
            if (element.Count > (std::numeric_limits<size_t>::max() / 3u))
                return EFastFaceReadResult::Error;
            const size_t reserveCount = element.Count * 3u;
            if (_outIndices.size() >
                (std::numeric_limits<size_t>::max() - reserveCount))
                return EFastFaceReadResult::Error;
            _outIndices.reserve(_outIndices.size() + reserveCount);
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
            for (size_t j = 0u; j < element.Count; ++j) {
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
                _outIndices.push_back(i0);
                _outIndices.push_back(i1);
                _outIndices.push_back(i2);
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
                    _outIndices.push_back(i0);
                    _outIndices.push_back(prev);
                    _outIndices.push_back(idx);
                    prev = idx;
                }
                ++_faceCount;
            }
            return EFastFaceReadResult::Success;
        }
        IAssetLoader::SAssetLoadContext inner;
        uint32_t topHierarchyLevel;
        IAssetLoader::IAssetLoaderOverride* loaderOverride;
        core::vector<char> Buffer;
        size_t ioReadWindowSize = DefaultIoReadWindowBytes;
        core::vector<SElement> ElementList = {};
        char *StartPointer = nullptr, *EndPointer = nullptr,
             *LineEndPointer = nullptr;
        int32_t LineLength = 0;
        int32_t WordLength = -1;
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
    auto geometry = make_smart_refctd_ptr<ICPUPolygonGeometry>();
    hlsl::shapes::util::AABBAccumulator3<float> parsedAABB = hlsl::shapes::util::createAABBAccumulator<float>();
    uint32_t vertCount = 0;
    Parse::ContentHashBuild contentHashBuild = Parse::ContentHashBuild::create(computeContentHashes, hashInBuild);
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
    if (Parse::toStringView(ctx.getNextLine()) != "ply") {
        _params.logger.log("Not a valid PLY file %s", system::ILogger::ELL_ERROR,
                           ctx.inner.mainFile->getFileName().string().c_str());
        return {};
    }
    ctx.getNextLine();
    const char* word = ctx.getNextWord();
    for (; Parse::toStringView(word) == "comment"; ctx.getNextLine())
        word = ctx.getNextWord();
    bool readingHeader = true;
    bool continueReading = true;
    ctx.IsBinaryFile = false;
    ctx.IsWrongEndian = false;
    do {
        const std::string_view wordView = Parse::toStringView(word);
        if (wordView == "property") {
            word = ctx.getNextWord();
            if (ctx.ElementList.empty()) {
                _params.logger.log("PLY property token found before element %s",
                                   system::ILogger::ELL_WARNING, word);
            } else {
                auto& el = ctx.ElementList.back();
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
        } else if (wordView == "format") {
            word = ctx.getNextWord();
            const std::string_view formatView = Parse::toStringView(word);
            if (formatView == "binary_little_endian") {
                ctx.IsBinaryFile = true;
            } else if (formatView == "binary_big_endian") {
                ctx.IsBinaryFile = true;
                ctx.IsWrongEndian = true;
            } else if (formatView == "ascii") {
            } else {
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
    if (!continueReading)
        return {};
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
    for (uint32_t i = 0; i < ctx.ElementList.size(); ++i) {
        auto& el = ctx.ElementList[i];
        if (el.Name == "vertex") {
            if (verticesProcessed) {
                _params.logger.log("Multiple `vertex` elements not supported!",
                                   system::ILogger::ELL_ERROR);
                return {};
            }
            ICPUPolygonGeometry::SDataViewBase posView = {}, normalView = {},
                                               uvView = {};
            core::vector<ICPUPolygonGeometry::SDataView> extraViews;
            for (auto& vertexProperty : el.Properties) {
                const auto& propertyName = vertexProperty.Name;
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
        } else if (el.Name == "face") {
            if (!readFaceElement(el))
                return {};
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
    if (contentHashBuild.hashesDeferred()) {
        contentHashBuild.wait();
        SPolygonGeometryContentHash::computeMissing(geometry.get(),
                                                    _params.ioPolicy);
    } else {
        hashRemainingGeometryBuffers();
    }
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
    auto meta = core::make_smart_refctd_ptr<CPLYMetadata>();
    return SAssetBundle(std::move(meta), {std::move(geometry)});
}
}
#endif // _NBL_COMPILE_WITH_PLY_LOADER_
