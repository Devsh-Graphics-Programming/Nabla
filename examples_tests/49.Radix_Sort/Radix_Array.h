#ifndef _RADIX_ARRAY_H_INCLUDED__
#define _RADIX_ARRAY_H_INCLUDED__

#include <cstdint>

namespace Radix_Sort
{
	class Vec2D;	//Forward Declaration
	
	class Radix_Array
	{
	private:
		std::size_t Data_Size;
		Vec2D* Data_Vec2D;

		void Deallocate_Resource() noexcept;
	public:
		Radix_Array();
		Radix_Array(const std::size_t Array_Size);

		Radix_Array(const Radix_Array& Object);
		Radix_Array(Radix_Array&& Object) noexcept;

		//[[nodiscard]] Vec2D* begin() const noexcept;
		//[[nodiscard]] Vec2D* end() const noexcept;

		Radix_Array& operator=(const Radix_Array& Object);
		Radix_Array& operator=(Radix_Array&& Object) noexcept;

		[[nodiscard]] inline std::size_t Get_Data_Size() const noexcept
		{
			return Data_Size;
		}
		
		~Radix_Array();
	};
	
}

#endif /* _RADIX_ARRAY_H_INCLUDED__ */