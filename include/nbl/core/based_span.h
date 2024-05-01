#ifndef _NBL_CORE_BASED_SPAN_INCLUDED_
#define _NBL_CORE_BASED_SPAN_INCLUDED_

namespace nbl::core
{
	template<typename T, size_t Extent = std::dynamic_extent>
	struct based_span
	{
		constexpr static inline bool IsConst = std::is_const_v<T>;

	public:
		using element_type = T;

		constexpr based_span()
		{
			static_assert(sizeof(based_span<T, Extent>) == sizeof(std::span<T, Extent>));
		}

		constexpr explicit based_span(T* basePtr, std::span<T, Extent> span) : m_byteOffset(ptrdiff_t(span.data()) - ptrdiff_t(basePtr)), m_size(span.size()) {}
		constexpr based_span(size_t byteOffset, size_t size) : m_byteOffset(byteOffset), m_size(size) {}

		inline bool empty() const { return m_size == 0ull; }

		inline std::span<T> operator()(std::conditional_t<IsConst, const void*, void*> newBase) const
		{
			std::conditional_t<IsConst, const uint8_t, uint8_t>* retval = nullptr;
			if (!empty())
				retval = reinterpret_cast<decltype(retval)>(newBase) + m_byteOffset;
			return { reinterpret_cast<T*>(retval),m_size };
		}

		inline auto byte_offset() const { return m_byteOffset; }

	private:
		size_t m_byteOffset = ~0ull;
		size_t m_size = 0ull;
	};
}

#endif