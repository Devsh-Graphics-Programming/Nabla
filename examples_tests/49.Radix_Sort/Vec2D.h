#ifndef _VEC2D_H_INCLUDED__
#define _VEC2D_H_INCLUDED__

#include <cstdint>
#include <memory>
#include <utility>
#include <algorithm>

namespace nbl
{

	class Vec2D
	{
	private:
		uint32_t Key;
		uint32_t Data;
	protected:
	
	public:
		Vec2D() = default;
		Vec2D(const uint32_t Key, const uint32_t Data);
		
		Vec2D(const Vec2D& Object) = default;
		Vec2D(Vec2D&& Object) = delete;

		

		Vec2D& operator=(const Vec2D& Object) = default;
		Vec2D& operator=(Vec2D&& Object) = delete;

		/////////////////////////////////////////////////////////////////
		
		[[nodiscard]] inline uint32_t Get_Key() const
		{
			return Key;
		}
		
		[[nodiscard]] inline uint32_t Get_Data() const
		{
			return Data;
		}
		
		~Vec2D() = default;
	};
	
}

#endif