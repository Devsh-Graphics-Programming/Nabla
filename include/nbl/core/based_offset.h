#ifndef _NBL_CORE_BASED_OFFET_INCLUDED_
#define _NBL_CORE_BASED_OFFET_INCLUDED_

namespace nbl::core
{
	template<typename T>
	struct based_offset
	{
		constexpr static inline bool IsConst = std::is_const_v<T>;

	public:
		using element_type = T;

		constexpr based_offset() {}

		constexpr explicit based_offset(T* basePtr, T* ptr) : m_byteOffset(ptrdiff_t(ptr) - ptrdiff_t(basePtr)) {}
		constexpr based_offset(const size_t _byteOffset) : m_byteOffset(_byteOffset) {}

		inline explicit operator bool() const { return m_byteOffset != InvalidOffset; }
		inline T* operator()(std::conditional_t<IsConst, const void*, void*> newBase) const
		{
			std::conditional_t<IsConst, const uint8_t, uint8_t>* retval = nullptr;
			if (bool(*this))
				retval = reinterpret_cast<decltype(retval)>(newBase) + m_byteOffset;
			return reinterpret_cast<T*>(retval);
		}

		inline based_offset<T> operator+(const size_t extraOff) const { return { sizeof(T) * extraOff + m_byteOffset }; }

		inline auto byte_offset() const { return m_byteOffset; }

	private:
		constexpr static inline size_t InvalidOffset = ~0ull;
		size_t m_byteOffset = InvalidOffset;
	};
}

#endif