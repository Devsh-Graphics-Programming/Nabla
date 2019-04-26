#ifndef _SIMD_SWIZZLE_H_
#define _SIMD_SWIZZLE_H_


template <class T, class X>
class IRR_FORCE_EBO SIMD_32bitSwizzleAble
{
    template<int mask>
    inline X shuffleFunc(X reg) const;
    public:
        inline T xxxx() const {return shuffleFunc<_MM_SHUFFLE(0,0,0,0)>(((const T*)this)->getAsRegister());}
        inline T xxxy() const {return shuffleFunc<_MM_SHUFFLE(1,0,0,0)>(((const T*)this)->getAsRegister());}
        inline T xxxz() const {return shuffleFunc<_MM_SHUFFLE(2,0,0,0)>(((const T*)this)->getAsRegister());}
        inline T xxxw() const {return shuffleFunc<_MM_SHUFFLE(3,0,0,0)>(((const T*)this)->getAsRegister());}
        inline T xxyx() const {return shuffleFunc<_MM_SHUFFLE(0,1,0,0)>(((const T*)this)->getAsRegister());}
        inline T xxyy() const {return shuffleFunc<_MM_SHUFFLE(1,1,0,0)>(((const T*)this)->getAsRegister());}
        inline T xxyz() const {return shuffleFunc<_MM_SHUFFLE(2,1,0,0)>(((const T*)this)->getAsRegister());}
        inline T xxyw() const {return shuffleFunc<_MM_SHUFFLE(3,1,0,0)>(((const T*)this)->getAsRegister());}
        inline T xxzx() const {return shuffleFunc<_MM_SHUFFLE(0,2,0,0)>(((const T*)this)->getAsRegister());}
        inline T xxzy() const {return shuffleFunc<_MM_SHUFFLE(1,2,0,0)>(((const T*)this)->getAsRegister());}
        inline T xxzz() const {return shuffleFunc<_MM_SHUFFLE(2,2,0,0)>(((const T*)this)->getAsRegister());}
        inline T xxzw() const {return shuffleFunc<_MM_SHUFFLE(3,2,0,0)>(((const T*)this)->getAsRegister());}
        inline T xxwx() const {return shuffleFunc<_MM_SHUFFLE(0,3,0,0)>(((const T*)this)->getAsRegister());}
        inline T xxwy() const {return shuffleFunc<_MM_SHUFFLE(1,3,0,0)>(((const T*)this)->getAsRegister());}
        inline T xxwz() const {return shuffleFunc<_MM_SHUFFLE(2,3,0,0)>(((const T*)this)->getAsRegister());}
        inline T xxww() const {return shuffleFunc<_MM_SHUFFLE(3,3,0,0)>(((const T*)this)->getAsRegister());}
        inline T xyxx() const {return shuffleFunc<_MM_SHUFFLE(0,0,1,0)>(((const T*)this)->getAsRegister());}
        inline T xyxy() const {return shuffleFunc<_MM_SHUFFLE(1,0,1,0)>(((const T*)this)->getAsRegister());}
        inline T xyxz() const {return shuffleFunc<_MM_SHUFFLE(2,0,1,0)>(((const T*)this)->getAsRegister());}
        inline T xyxw() const {return shuffleFunc<_MM_SHUFFLE(3,0,1,0)>(((const T*)this)->getAsRegister());}
        inline T xyyx() const {return shuffleFunc<_MM_SHUFFLE(0,1,1,0)>(((const T*)this)->getAsRegister());}
        inline T xyyy() const {return shuffleFunc<_MM_SHUFFLE(1,1,1,0)>(((const T*)this)->getAsRegister());}
        inline T xyyz() const {return shuffleFunc<_MM_SHUFFLE(2,1,1,0)>(((const T*)this)->getAsRegister());}
        inline T xyyw() const {return shuffleFunc<_MM_SHUFFLE(3,1,1,0)>(((const T*)this)->getAsRegister());}
        inline T xyzx() const {return shuffleFunc<_MM_SHUFFLE(0,2,1,0)>(((const T*)this)->getAsRegister());}
        inline T xyzy() const {return shuffleFunc<_MM_SHUFFLE(1,2,1,0)>(((const T*)this)->getAsRegister());}
        inline T xyzz() const {return shuffleFunc<_MM_SHUFFLE(2,2,1,0)>(((const T*)this)->getAsRegister());}
        inline T xyzw() const {return shuffleFunc<_MM_SHUFFLE(3,2,1,0)>(((const T*)this)->getAsRegister());}
        inline T xywx() const {return shuffleFunc<_MM_SHUFFLE(0,3,1,0)>(((const T*)this)->getAsRegister());}
        inline T xywy() const {return shuffleFunc<_MM_SHUFFLE(1,3,1,0)>(((const T*)this)->getAsRegister());}
        inline T xywz() const {return shuffleFunc<_MM_SHUFFLE(2,3,1,0)>(((const T*)this)->getAsRegister());}
        inline T xyww() const {return shuffleFunc<_MM_SHUFFLE(3,3,1,0)>(((const T*)this)->getAsRegister());}
        inline T xzxx() const {return shuffleFunc<_MM_SHUFFLE(0,0,2,0)>(((const T*)this)->getAsRegister());}
        inline T xzxy() const {return shuffleFunc<_MM_SHUFFLE(1,0,2,0)>(((const T*)this)->getAsRegister());}
        inline T xzxz() const {return shuffleFunc<_MM_SHUFFLE(2,0,2,0)>(((const T*)this)->getAsRegister());}
        inline T xzxw() const {return shuffleFunc<_MM_SHUFFLE(3,0,2,0)>(((const T*)this)->getAsRegister());}
        inline T xzyx() const {return shuffleFunc<_MM_SHUFFLE(0,1,2,0)>(((const T*)this)->getAsRegister());}
        inline T xzyy() const {return shuffleFunc<_MM_SHUFFLE(1,1,2,0)>(((const T*)this)->getAsRegister());}
        inline T xzyz() const {return shuffleFunc<_MM_SHUFFLE(2,1,2,0)>(((const T*)this)->getAsRegister());}
        inline T xzyw() const {return shuffleFunc<_MM_SHUFFLE(3,1,2,0)>(((const T*)this)->getAsRegister());}
        inline T xzzx() const {return shuffleFunc<_MM_SHUFFLE(0,2,2,0)>(((const T*)this)->getAsRegister());}
        inline T xzzy() const {return shuffleFunc<_MM_SHUFFLE(1,2,2,0)>(((const T*)this)->getAsRegister());}
        inline T xzzz() const {return shuffleFunc<_MM_SHUFFLE(2,2,2,0)>(((const T*)this)->getAsRegister());}
        inline T xzzw() const {return shuffleFunc<_MM_SHUFFLE(3,2,2,0)>(((const T*)this)->getAsRegister());}
        inline T xzwx() const {return shuffleFunc<_MM_SHUFFLE(0,3,2,0)>(((const T*)this)->getAsRegister());}
        inline T xzwy() const {return shuffleFunc<_MM_SHUFFLE(1,3,2,0)>(((const T*)this)->getAsRegister());}
        inline T xzwz() const {return shuffleFunc<_MM_SHUFFLE(2,3,2,0)>(((const T*)this)->getAsRegister());}
        inline T xzww() const {return shuffleFunc<_MM_SHUFFLE(3,3,2,0)>(((const T*)this)->getAsRegister());}
        inline T xwxx() const {return shuffleFunc<_MM_SHUFFLE(0,0,3,0)>(((const T*)this)->getAsRegister());}
        inline T xwxy() const {return shuffleFunc<_MM_SHUFFLE(1,0,3,0)>(((const T*)this)->getAsRegister());}
        inline T xwxz() const {return shuffleFunc<_MM_SHUFFLE(2,0,3,0)>(((const T*)this)->getAsRegister());}
        inline T xwxw() const {return shuffleFunc<_MM_SHUFFLE(3,0,3,0)>(((const T*)this)->getAsRegister());}
        inline T xwyx() const {return shuffleFunc<_MM_SHUFFLE(0,1,3,0)>(((const T*)this)->getAsRegister());}
        inline T xwyy() const {return shuffleFunc<_MM_SHUFFLE(1,1,3,0)>(((const T*)this)->getAsRegister());}
        inline T xwyz() const {return shuffleFunc<_MM_SHUFFLE(2,1,3,0)>(((const T*)this)->getAsRegister());}
        inline T xwyw() const {return shuffleFunc<_MM_SHUFFLE(3,1,3,0)>(((const T*)this)->getAsRegister());}
        inline T xwzx() const {return shuffleFunc<_MM_SHUFFLE(0,2,3,0)>(((const T*)this)->getAsRegister());}
        inline T xwzy() const {return shuffleFunc<_MM_SHUFFLE(1,2,3,0)>(((const T*)this)->getAsRegister());}
        inline T xwzz() const {return shuffleFunc<_MM_SHUFFLE(2,2,3,0)>(((const T*)this)->getAsRegister());}
        inline T xwzw() const {return shuffleFunc<_MM_SHUFFLE(3,2,3,0)>(((const T*)this)->getAsRegister());}
        inline T xwwx() const {return shuffleFunc<_MM_SHUFFLE(0,3,3,0)>(((const T*)this)->getAsRegister());}
        inline T xwwy() const {return shuffleFunc<_MM_SHUFFLE(1,3,3,0)>(((const T*)this)->getAsRegister());}
        inline T xwwz() const {return shuffleFunc<_MM_SHUFFLE(2,3,3,0)>(((const T*)this)->getAsRegister());}
        inline T xwww() const {return shuffleFunc<_MM_SHUFFLE(3,3,3,0)>(((const T*)this)->getAsRegister());}
        inline T yxxx() const {return shuffleFunc<_MM_SHUFFLE(0,0,0,0)>(((const T*)this)->getAsRegister());}
        inline T yxxy() const {return shuffleFunc<_MM_SHUFFLE(1,0,0,1)>(((const T*)this)->getAsRegister());}
        inline T yxxz() const {return shuffleFunc<_MM_SHUFFLE(2,0,0,1)>(((const T*)this)->getAsRegister());}
        inline T yxxw() const {return shuffleFunc<_MM_SHUFFLE(3,0,0,1)>(((const T*)this)->getAsRegister());}
        inline T yxyx() const {return shuffleFunc<_MM_SHUFFLE(0,1,0,1)>(((const T*)this)->getAsRegister());}
        inline T yxyy() const {return shuffleFunc<_MM_SHUFFLE(1,1,0,1)>(((const T*)this)->getAsRegister());}
        inline T yxyz() const {return shuffleFunc<_MM_SHUFFLE(2,1,0,1)>(((const T*)this)->getAsRegister());}
        inline T yxyw() const {return shuffleFunc<_MM_SHUFFLE(3,1,0,1)>(((const T*)this)->getAsRegister());}
        inline T yxzx() const {return shuffleFunc<_MM_SHUFFLE(0,2,0,1)>(((const T*)this)->getAsRegister());}
        inline T yxzy() const {return shuffleFunc<_MM_SHUFFLE(1,2,0,1)>(((const T*)this)->getAsRegister());}
        inline T yxzz() const {return shuffleFunc<_MM_SHUFFLE(2,2,0,1)>(((const T*)this)->getAsRegister());}
        inline T yxzw() const {return shuffleFunc<_MM_SHUFFLE(3,2,0,1)>(((const T*)this)->getAsRegister());}
        inline T yxwx() const {return shuffleFunc<_MM_SHUFFLE(0,3,0,1)>(((const T*)this)->getAsRegister());}
        inline T yxwy() const {return shuffleFunc<_MM_SHUFFLE(1,3,0,1)>(((const T*)this)->getAsRegister());}
        inline T yxwz() const {return shuffleFunc<_MM_SHUFFLE(2,3,0,1)>(((const T*)this)->getAsRegister());}
        inline T yxww() const {return shuffleFunc<_MM_SHUFFLE(3,3,0,1)>(((const T*)this)->getAsRegister());}
        inline T yyxx() const {return shuffleFunc<_MM_SHUFFLE(0,0,1,1)>(((const T*)this)->getAsRegister());}
        inline T yyxy() const {return shuffleFunc<_MM_SHUFFLE(1,0,1,1)>(((const T*)this)->getAsRegister());}
        inline T yyxz() const {return shuffleFunc<_MM_SHUFFLE(2,0,1,1)>(((const T*)this)->getAsRegister());}
        inline T yyxw() const {return shuffleFunc<_MM_SHUFFLE(3,0,1,1)>(((const T*)this)->getAsRegister());}
        inline T yyyx() const {return shuffleFunc<_MM_SHUFFLE(0,1,1,1)>(((const T*)this)->getAsRegister());}
        inline T yyyy() const {return shuffleFunc<_MM_SHUFFLE(1,1,1,1)>(((const T*)this)->getAsRegister());}
        inline T yyyz() const {return shuffleFunc<_MM_SHUFFLE(2,1,1,1)>(((const T*)this)->getAsRegister());}
        inline T yyyw() const {return shuffleFunc<_MM_SHUFFLE(3,1,1,1)>(((const T*)this)->getAsRegister());}
        inline T yyzx() const {return shuffleFunc<_MM_SHUFFLE(0,2,1,1)>(((const T*)this)->getAsRegister());}
        inline T yyzy() const {return shuffleFunc<_MM_SHUFFLE(1,2,1,1)>(((const T*)this)->getAsRegister());}
        inline T yyzz() const {return shuffleFunc<_MM_SHUFFLE(2,2,1,1)>(((const T*)this)->getAsRegister());}
        inline T yyzw() const {return shuffleFunc<_MM_SHUFFLE(3,2,1,1)>(((const T*)this)->getAsRegister());}
        inline T yywx() const {return shuffleFunc<_MM_SHUFFLE(0,3,1,1)>(((const T*)this)->getAsRegister());}
        inline T yywy() const {return shuffleFunc<_MM_SHUFFLE(1,3,1,1)>(((const T*)this)->getAsRegister());}
        inline T yywz() const {return shuffleFunc<_MM_SHUFFLE(2,3,1,1)>(((const T*)this)->getAsRegister());}
        inline T yyww() const {return shuffleFunc<_MM_SHUFFLE(3,3,1,1)>(((const T*)this)->getAsRegister());}
        inline T yzxx() const {return shuffleFunc<_MM_SHUFFLE(0,0,2,1)>(((const T*)this)->getAsRegister());}
        inline T yzxy() const {return shuffleFunc<_MM_SHUFFLE(1,0,2,1)>(((const T*)this)->getAsRegister());}
        inline T yzxz() const {return shuffleFunc<_MM_SHUFFLE(2,0,2,1)>(((const T*)this)->getAsRegister());}
        inline T yzxw() const {return shuffleFunc<_MM_SHUFFLE(3,0,2,1)>(((const T*)this)->getAsRegister());}
        inline T yzyx() const {return shuffleFunc<_MM_SHUFFLE(0,1,2,1)>(((const T*)this)->getAsRegister());}
        inline T yzyy() const {return shuffleFunc<_MM_SHUFFLE(1,1,2,1)>(((const T*)this)->getAsRegister());}
        inline T yzyz() const {return shuffleFunc<_MM_SHUFFLE(2,1,2,1)>(((const T*)this)->getAsRegister());}
        inline T yzyw() const {return shuffleFunc<_MM_SHUFFLE(3,1,2,1)>(((const T*)this)->getAsRegister());}
        inline T yzzx() const {return shuffleFunc<_MM_SHUFFLE(0,2,2,1)>(((const T*)this)->getAsRegister());}
        inline T yzzy() const {return shuffleFunc<_MM_SHUFFLE(1,2,2,1)>(((const T*)this)->getAsRegister());}
        inline T yzzz() const {return shuffleFunc<_MM_SHUFFLE(2,2,2,1)>(((const T*)this)->getAsRegister());}
        inline T yzzw() const {return shuffleFunc<_MM_SHUFFLE(3,2,2,1)>(((const T*)this)->getAsRegister());}
        inline T yzwx() const {return shuffleFunc<_MM_SHUFFLE(0,3,2,1)>(((const T*)this)->getAsRegister());}
        inline T yzwy() const {return shuffleFunc<_MM_SHUFFLE(1,3,2,1)>(((const T*)this)->getAsRegister());}
        inline T yzwz() const {return shuffleFunc<_MM_SHUFFLE(2,3,2,1)>(((const T*)this)->getAsRegister());}
        inline T yzww() const {return shuffleFunc<_MM_SHUFFLE(3,3,2,1)>(((const T*)this)->getAsRegister());}
        inline T ywxx() const {return shuffleFunc<_MM_SHUFFLE(0,0,3,1)>(((const T*)this)->getAsRegister());}
        inline T ywxy() const {return shuffleFunc<_MM_SHUFFLE(1,0,3,1)>(((const T*)this)->getAsRegister());}
        inline T ywxz() const {return shuffleFunc<_MM_SHUFFLE(2,0,3,1)>(((const T*)this)->getAsRegister());}
        inline T ywxw() const {return shuffleFunc<_MM_SHUFFLE(3,0,3,1)>(((const T*)this)->getAsRegister());}
        inline T ywyx() const {return shuffleFunc<_MM_SHUFFLE(0,1,3,1)>(((const T*)this)->getAsRegister());}
        inline T ywyy() const {return shuffleFunc<_MM_SHUFFLE(1,1,3,1)>(((const T*)this)->getAsRegister());}
        inline T ywyz() const {return shuffleFunc<_MM_SHUFFLE(2,1,3,1)>(((const T*)this)->getAsRegister());}
        inline T ywyw() const {return shuffleFunc<_MM_SHUFFLE(3,1,3,1)>(((const T*)this)->getAsRegister());}
        inline T ywzx() const {return shuffleFunc<_MM_SHUFFLE(0,2,3,1)>(((const T*)this)->getAsRegister());}
        inline T ywzy() const {return shuffleFunc<_MM_SHUFFLE(1,2,3,1)>(((const T*)this)->getAsRegister());}
        inline T ywzz() const {return shuffleFunc<_MM_SHUFFLE(2,2,3,1)>(((const T*)this)->getAsRegister());}
        inline T ywzw() const {return shuffleFunc<_MM_SHUFFLE(3,2,3,1)>(((const T*)this)->getAsRegister());}
        inline T ywwx() const {return shuffleFunc<_MM_SHUFFLE(0,3,3,1)>(((const T*)this)->getAsRegister());}
        inline T ywwy() const {return shuffleFunc<_MM_SHUFFLE(1,3,3,1)>(((const T*)this)->getAsRegister());}
        inline T ywwz() const {return shuffleFunc<_MM_SHUFFLE(2,3,3,1)>(((const T*)this)->getAsRegister());}
        inline T ywww() const {return shuffleFunc<_MM_SHUFFLE(3,3,3,1)>(((const T*)this)->getAsRegister());}
        inline T zxxx() const {return shuffleFunc<_MM_SHUFFLE(0,0,0,2)>(((const T*)this)->getAsRegister());}
        inline T zxxy() const {return shuffleFunc<_MM_SHUFFLE(1,0,0,2)>(((const T*)this)->getAsRegister());}
        inline T zxxz() const {return shuffleFunc<_MM_SHUFFLE(2,0,0,2)>(((const T*)this)->getAsRegister());}
        inline T zxxw() const {return shuffleFunc<_MM_SHUFFLE(3,0,0,2)>(((const T*)this)->getAsRegister());}
        inline T zxyx() const {return shuffleFunc<_MM_SHUFFLE(0,1,0,2)>(((const T*)this)->getAsRegister());}
        inline T zxyy() const {return shuffleFunc<_MM_SHUFFLE(1,1,0,2)>(((const T*)this)->getAsRegister());}
        inline T zxyz() const {return shuffleFunc<_MM_SHUFFLE(2,1,0,2)>(((const T*)this)->getAsRegister());}
        inline T zxyw() const {return shuffleFunc<_MM_SHUFFLE(3,1,0,2)>(((const T*)this)->getAsRegister());}
        inline T zxzx() const {return shuffleFunc<_MM_SHUFFLE(0,2,0,2)>(((const T*)this)->getAsRegister());}
        inline T zxzy() const {return shuffleFunc<_MM_SHUFFLE(1,2,0,2)>(((const T*)this)->getAsRegister());}
        inline T zxzz() const {return shuffleFunc<_MM_SHUFFLE(2,2,0,2)>(((const T*)this)->getAsRegister());}
        inline T zxzw() const {return shuffleFunc<_MM_SHUFFLE(3,2,0,2)>(((const T*)this)->getAsRegister());}
        inline T zxwx() const {return shuffleFunc<_MM_SHUFFLE(0,3,0,2)>(((const T*)this)->getAsRegister());}
        inline T zxwy() const {return shuffleFunc<_MM_SHUFFLE(1,3,0,2)>(((const T*)this)->getAsRegister());}
        inline T zxwz() const {return shuffleFunc<_MM_SHUFFLE(2,3,0,2)>(((const T*)this)->getAsRegister());}
        inline T zxww() const {return shuffleFunc<_MM_SHUFFLE(3,3,0,2)>(((const T*)this)->getAsRegister());}
        inline T zyxx() const {return shuffleFunc<_MM_SHUFFLE(0,0,1,2)>(((const T*)this)->getAsRegister());}
        inline T zyxy() const {return shuffleFunc<_MM_SHUFFLE(1,0,1,2)>(((const T*)this)->getAsRegister());}
        inline T zyxz() const {return shuffleFunc<_MM_SHUFFLE(2,0,1,2)>(((const T*)this)->getAsRegister());}
        inline T zyxw() const {return shuffleFunc<_MM_SHUFFLE(3,0,1,2)>(((const T*)this)->getAsRegister());}
        inline T zyyx() const {return shuffleFunc<_MM_SHUFFLE(0,1,1,2)>(((const T*)this)->getAsRegister());}
        inline T zyyy() const {return shuffleFunc<_MM_SHUFFLE(1,1,1,2)>(((const T*)this)->getAsRegister());}
        inline T zyyz() const {return shuffleFunc<_MM_SHUFFLE(2,1,1,2)>(((const T*)this)->getAsRegister());}
        inline T zyyw() const {return shuffleFunc<_MM_SHUFFLE(3,1,1,2)>(((const T*)this)->getAsRegister());}
        inline T zyzx() const {return shuffleFunc<_MM_SHUFFLE(0,2,1,2)>(((const T*)this)->getAsRegister());}
        inline T zyzy() const {return shuffleFunc<_MM_SHUFFLE(1,2,1,2)>(((const T*)this)->getAsRegister());}
        inline T zyzz() const {return shuffleFunc<_MM_SHUFFLE(2,2,1,2)>(((const T*)this)->getAsRegister());}
        inline T zyzw() const {return shuffleFunc<_MM_SHUFFLE(3,2,1,2)>(((const T*)this)->getAsRegister());}
        inline T zywx() const {return shuffleFunc<_MM_SHUFFLE(0,3,1,2)>(((const T*)this)->getAsRegister());}
        inline T zywy() const {return shuffleFunc<_MM_SHUFFLE(1,3,1,2)>(((const T*)this)->getAsRegister());}
        inline T zywz() const {return shuffleFunc<_MM_SHUFFLE(2,3,1,2)>(((const T*)this)->getAsRegister());}
        inline T zyww() const {return shuffleFunc<_MM_SHUFFLE(3,3,1,2)>(((const T*)this)->getAsRegister());}
        inline T zzxx() const {return shuffleFunc<_MM_SHUFFLE(0,0,2,2)>(((const T*)this)->getAsRegister());}
        inline T zzxy() const {return shuffleFunc<_MM_SHUFFLE(1,0,2,2)>(((const T*)this)->getAsRegister());}
        inline T zzxz() const {return shuffleFunc<_MM_SHUFFLE(2,0,2,2)>(((const T*)this)->getAsRegister());}
        inline T zzxw() const {return shuffleFunc<_MM_SHUFFLE(3,0,2,2)>(((const T*)this)->getAsRegister());}
        inline T zzyx() const {return shuffleFunc<_MM_SHUFFLE(0,1,2,2)>(((const T*)this)->getAsRegister());}
        inline T zzyy() const {return shuffleFunc<_MM_SHUFFLE(1,1,2,2)>(((const T*)this)->getAsRegister());}
        inline T zzyz() const {return shuffleFunc<_MM_SHUFFLE(2,1,2,2)>(((const T*)this)->getAsRegister());}
        inline T zzyw() const {return shuffleFunc<_MM_SHUFFLE(3,1,2,2)>(((const T*)this)->getAsRegister());}
        inline T zzzx() const {return shuffleFunc<_MM_SHUFFLE(0,2,2,2)>(((const T*)this)->getAsRegister());}
        inline T zzzy() const {return shuffleFunc<_MM_SHUFFLE(1,2,2,2)>(((const T*)this)->getAsRegister());}
        inline T zzzz() const {return shuffleFunc<_MM_SHUFFLE(2,2,2,2)>(((const T*)this)->getAsRegister());}
        inline T zzzw() const {return shuffleFunc<_MM_SHUFFLE(3,2,2,2)>(((const T*)this)->getAsRegister());}
        inline T zzwx() const {return shuffleFunc<_MM_SHUFFLE(0,3,2,2)>(((const T*)this)->getAsRegister());}
        inline T zzwy() const {return shuffleFunc<_MM_SHUFFLE(1,3,2,2)>(((const T*)this)->getAsRegister());}
        inline T zzwz() const {return shuffleFunc<_MM_SHUFFLE(2,3,2,2)>(((const T*)this)->getAsRegister());}
        inline T zzww() const {return shuffleFunc<_MM_SHUFFLE(3,3,2,2)>(((const T*)this)->getAsRegister());}
        inline T zwxx() const {return shuffleFunc<_MM_SHUFFLE(0,0,3,2)>(((const T*)this)->getAsRegister());}
        inline T zwxy() const {return shuffleFunc<_MM_SHUFFLE(1,0,3,2)>(((const T*)this)->getAsRegister());}
        inline T zwxz() const {return shuffleFunc<_MM_SHUFFLE(2,0,3,2)>(((const T*)this)->getAsRegister());}
        inline T zwxw() const {return shuffleFunc<_MM_SHUFFLE(3,0,3,2)>(((const T*)this)->getAsRegister());}
        inline T zwyx() const {return shuffleFunc<_MM_SHUFFLE(0,1,3,2)>(((const T*)this)->getAsRegister());}
        inline T zwyy() const {return shuffleFunc<_MM_SHUFFLE(1,1,3,2)>(((const T*)this)->getAsRegister());}
        inline T zwyz() const {return shuffleFunc<_MM_SHUFFLE(2,1,3,2)>(((const T*)this)->getAsRegister());}
        inline T zwyw() const {return shuffleFunc<_MM_SHUFFLE(3,1,3,2)>(((const T*)this)->getAsRegister());}
        inline T zwzx() const {return shuffleFunc<_MM_SHUFFLE(0,2,3,2)>(((const T*)this)->getAsRegister());}
        inline T zwzy() const {return shuffleFunc<_MM_SHUFFLE(1,2,3,2)>(((const T*)this)->getAsRegister());}
        inline T zwzz() const {return shuffleFunc<_MM_SHUFFLE(2,2,3,2)>(((const T*)this)->getAsRegister());}
        inline T zwzw() const {return shuffleFunc<_MM_SHUFFLE(3,2,3,2)>(((const T*)this)->getAsRegister());}
        inline T zwwx() const {return shuffleFunc<_MM_SHUFFLE(0,3,3,2)>(((const T*)this)->getAsRegister());}
        inline T zwwy() const {return shuffleFunc<_MM_SHUFFLE(1,3,3,2)>(((const T*)this)->getAsRegister());}
        inline T zwwz() const {return shuffleFunc<_MM_SHUFFLE(2,3,3,2)>(((const T*)this)->getAsRegister());}
        inline T zwww() const {return shuffleFunc<_MM_SHUFFLE(3,3,3,2)>(((const T*)this)->getAsRegister());}
        inline T wxxx() const {return shuffleFunc<_MM_SHUFFLE(0,0,0,3)>(((const T*)this)->getAsRegister());}
        inline T wxxy() const {return shuffleFunc<_MM_SHUFFLE(1,0,0,3)>(((const T*)this)->getAsRegister());}
        inline T wxxz() const {return shuffleFunc<_MM_SHUFFLE(2,0,0,3)>(((const T*)this)->getAsRegister());}
        inline T wxxw() const {return shuffleFunc<_MM_SHUFFLE(3,0,0,3)>(((const T*)this)->getAsRegister());}
        inline T wxyx() const {return shuffleFunc<_MM_SHUFFLE(0,1,0,3)>(((const T*)this)->getAsRegister());}
        inline T wxyy() const {return shuffleFunc<_MM_SHUFFLE(1,1,0,3)>(((const T*)this)->getAsRegister());}
        inline T wxyz() const {return shuffleFunc<_MM_SHUFFLE(2,1,0,3)>(((const T*)this)->getAsRegister());}
        inline T wxyw() const {return shuffleFunc<_MM_SHUFFLE(3,1,0,3)>(((const T*)this)->getAsRegister());}
        inline T wxzx() const {return shuffleFunc<_MM_SHUFFLE(0,2,0,3)>(((const T*)this)->getAsRegister());}
        inline T wxzy() const {return shuffleFunc<_MM_SHUFFLE(1,2,0,3)>(((const T*)this)->getAsRegister());}
        inline T wxzz() const {return shuffleFunc<_MM_SHUFFLE(2,2,0,3)>(((const T*)this)->getAsRegister());}
        inline T wxzw() const {return shuffleFunc<_MM_SHUFFLE(3,2,0,3)>(((const T*)this)->getAsRegister());}
        inline T wxwx() const {return shuffleFunc<_MM_SHUFFLE(0,3,0,3)>(((const T*)this)->getAsRegister());}
        inline T wxwy() const {return shuffleFunc<_MM_SHUFFLE(1,3,0,3)>(((const T*)this)->getAsRegister());}
        inline T wxwz() const {return shuffleFunc<_MM_SHUFFLE(2,3,0,3)>(((const T*)this)->getAsRegister());}
        inline T wxww() const {return shuffleFunc<_MM_SHUFFLE(3,3,0,3)>(((const T*)this)->getAsRegister());}
        inline T wyxx() const {return shuffleFunc<_MM_SHUFFLE(0,0,1,3)>(((const T*)this)->getAsRegister());}
        inline T wyxy() const {return shuffleFunc<_MM_SHUFFLE(1,0,1,3)>(((const T*)this)->getAsRegister());}
        inline T wyxz() const {return shuffleFunc<_MM_SHUFFLE(2,0,1,3)>(((const T*)this)->getAsRegister());}
        inline T wyxw() const {return shuffleFunc<_MM_SHUFFLE(3,0,1,3)>(((const T*)this)->getAsRegister());}
        inline T wyyx() const {return shuffleFunc<_MM_SHUFFLE(0,1,1,3)>(((const T*)this)->getAsRegister());}
        inline T wyyy() const {return shuffleFunc<_MM_SHUFFLE(1,1,1,3)>(((const T*)this)->getAsRegister());}
        inline T wyyz() const {return shuffleFunc<_MM_SHUFFLE(2,1,1,3)>(((const T*)this)->getAsRegister());}
        inline T wyyw() const {return shuffleFunc<_MM_SHUFFLE(3,1,1,3)>(((const T*)this)->getAsRegister());}
        inline T wyzx() const {return shuffleFunc<_MM_SHUFFLE(0,2,1,3)>(((const T*)this)->getAsRegister());}
        inline T wyzy() const {return shuffleFunc<_MM_SHUFFLE(1,2,1,3)>(((const T*)this)->getAsRegister());}
        inline T wyzz() const {return shuffleFunc<_MM_SHUFFLE(2,2,1,3)>(((const T*)this)->getAsRegister());}
        inline T wyzw() const {return shuffleFunc<_MM_SHUFFLE(3,2,1,3)>(((const T*)this)->getAsRegister());}
        inline T wywx() const {return shuffleFunc<_MM_SHUFFLE(0,3,1,3)>(((const T*)this)->getAsRegister());}
        inline T wywy() const {return shuffleFunc<_MM_SHUFFLE(1,3,1,3)>(((const T*)this)->getAsRegister());}
        inline T wywz() const {return shuffleFunc<_MM_SHUFFLE(2,3,1,3)>(((const T*)this)->getAsRegister());}
        inline T wyww() const {return shuffleFunc<_MM_SHUFFLE(3,3,1,3)>(((const T*)this)->getAsRegister());}
        inline T wzxx() const {return shuffleFunc<_MM_SHUFFLE(0,0,2,3)>(((const T*)this)->getAsRegister());}
        inline T wzxy() const {return shuffleFunc<_MM_SHUFFLE(1,0,2,3)>(((const T*)this)->getAsRegister());}
        inline T wzxz() const {return shuffleFunc<_MM_SHUFFLE(2,0,2,3)>(((const T*)this)->getAsRegister());}
        inline T wzxw() const {return shuffleFunc<_MM_SHUFFLE(3,0,2,3)>(((const T*)this)->getAsRegister());}
        inline T wzyx() const {return shuffleFunc<_MM_SHUFFLE(0,1,2,3)>(((const T*)this)->getAsRegister());}
        inline T wzyy() const {return shuffleFunc<_MM_SHUFFLE(1,1,2,3)>(((const T*)this)->getAsRegister());}
        inline T wzyz() const {return shuffleFunc<_MM_SHUFFLE(2,1,2,3)>(((const T*)this)->getAsRegister());}
        inline T wzyw() const {return shuffleFunc<_MM_SHUFFLE(3,1,2,3)>(((const T*)this)->getAsRegister());}
        inline T wzzx() const {return shuffleFunc<_MM_SHUFFLE(0,2,2,3)>(((const T*)this)->getAsRegister());}
        inline T wzzy() const {return shuffleFunc<_MM_SHUFFLE(1,2,2,3)>(((const T*)this)->getAsRegister());}
        inline T wzzz() const {return shuffleFunc<_MM_SHUFFLE(2,2,2,3)>(((const T*)this)->getAsRegister());}
        inline T wzzw() const {return shuffleFunc<_MM_SHUFFLE(3,2,2,3)>(((const T*)this)->getAsRegister());}
        inline T wzwx() const {return shuffleFunc<_MM_SHUFFLE(0,3,2,3)>(((const T*)this)->getAsRegister());}
        inline T wzwy() const {return shuffleFunc<_MM_SHUFFLE(1,3,2,3)>(((const T*)this)->getAsRegister());}
        inline T wzwz() const {return shuffleFunc<_MM_SHUFFLE(2,3,2,3)>(((const T*)this)->getAsRegister());}
        inline T wzww() const {return shuffleFunc<_MM_SHUFFLE(3,3,2,3)>(((const T*)this)->getAsRegister());}
        inline T wwxx() const {return shuffleFunc<_MM_SHUFFLE(0,0,3,3)>(((const T*)this)->getAsRegister());}
        inline T wwxy() const {return shuffleFunc<_MM_SHUFFLE(1,0,3,3)>(((const T*)this)->getAsRegister());}
        inline T wwxz() const {return shuffleFunc<_MM_SHUFFLE(2,0,3,3)>(((const T*)this)->getAsRegister());}
        inline T wwxw() const {return shuffleFunc<_MM_SHUFFLE(3,0,3,3)>(((const T*)this)->getAsRegister());}
        inline T wwyx() const {return shuffleFunc<_MM_SHUFFLE(0,1,3,3)>(((const T*)this)->getAsRegister());}
        inline T wwyy() const {return shuffleFunc<_MM_SHUFFLE(1,1,3,3)>(((const T*)this)->getAsRegister());}
        inline T wwyz() const {return shuffleFunc<_MM_SHUFFLE(2,1,3,3)>(((const T*)this)->getAsRegister());}
        inline T wwyw() const {return shuffleFunc<_MM_SHUFFLE(3,1,3,3)>(((const T*)this)->getAsRegister());}
        inline T wwzx() const {return shuffleFunc<_MM_SHUFFLE(0,2,3,3)>(((const T*)this)->getAsRegister());}
        inline T wwzy() const {return shuffleFunc<_MM_SHUFFLE(1,2,3,3)>(((const T*)this)->getAsRegister());}
        inline T wwzz() const {return shuffleFunc<_MM_SHUFFLE(2,2,3,3)>(((const T*)this)->getAsRegister());}
        inline T wwzw() const {return shuffleFunc<_MM_SHUFFLE(3,2,3,3)>(((const T*)this)->getAsRegister());}
        inline T wwwx() const {return shuffleFunc<_MM_SHUFFLE(0,3,3,3)>(((const T*)this)->getAsRegister());}
        inline T wwwy() const {return shuffleFunc<_MM_SHUFFLE(1,3,3,3)>(((const T*)this)->getAsRegister());}
        inline T wwwz() const {return shuffleFunc<_MM_SHUFFLE(2,3,3,3)>(((const T*)this)->getAsRegister());}
        inline T wwww() const {return shuffleFunc<_MM_SHUFFLE(3,3,3,3)>(((const T*)this)->getAsRegister());}

		template<size_t A, size_t B, size_t C, size_t D>
		inline T swizzle() const
		{
			return shuffleFunc<_MM_SHUFFLE(D, C, B, A)>(((const T*)this)->getAsRegister());
		}
};

#define FAST_FLOAT_SHUFFLE(X,Y) _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(X),Y))

#ifdef __GNUC__
// warning: ignoring attributes on template argument ‘__m128i {aka __vector(2) long long int}’ [-Wignored-attributes] (etc...)
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

template <>
template <int mask>
inline __m128 SIMD_32bitSwizzleAble<vectorSIMDf,__m128>::shuffleFunc(__m128 reg) const
{
    return FAST_FLOAT_SHUFFLE(reg,mask);
}

template <>
template <int mask>
inline __m128i SIMD_32bitSwizzleAble<vectorSIMD_32<int32_t>,__m128i>::shuffleFunc(__m128i reg) const
{
    return _mm_shuffle_epi32(reg,mask);
}

template <>
template <int mask>
inline __m128i SIMD_32bitSwizzleAble<vectorSIMD_32<uint32_t>,__m128i>::shuffleFunc(__m128i reg) const
{
    return _mm_shuffle_epi32(reg,mask);
}

#ifdef __GNUC__
#   pragma GCC diagnostic pop
#endif


template <class T, class X>
class IRR_FORCE_EBO SIMD_8bitSwizzleAble
{
	template<size_t A, size_t B, size_t C, size_t D, size_t E, size_t F, size_t G, size_t H, size_t I, size_t J, size_t K, size_t L, size_t M, size_t N, size_t O, size_t P>
	inline T swizzle() const
	{
		__m128i mask = _mm_set_epi8(P, O, N, M, L, K, J, I, H, G, F, E, D, C, B, A);
		return T(_mm_shuffle_epi8(((const T*)this)->getAsRegister(), mask));
	}
};

template <class T, class X>
class IRR_FORCE_EBO SIMD_16bitSwizzleAble
{
	template<size_t A, size_t B, size_t C, size_t D, size_t E, size_t F, size_t G, size_t H>
	inline T swizzle() const
	{
		__m128i mask = _mm_setr_epi8(2*A, 2*A+1, 2*B, 2*B+1, 2*C, 2*C+1, 2*D, 2*D+1, 2*E, 2*E+1, 2*F, 2*F+1, 2*G, 2*G+1, 2*H, 2*H+1);
		return T(_mm_shuffle_epi8(((const T*)this)->getAsRegister(), mask));
	}
};

#endif
