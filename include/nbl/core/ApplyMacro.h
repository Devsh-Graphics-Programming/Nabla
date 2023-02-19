#define EXPAND(x) x

#define APPLY0(M, ...)
#define APPLY1(M, A, ...) EXPAND(M(A))
#define APPLY2(M, A, ...) EXPAND(M(A)), EXPAND(APPLY1(M, __VA_ARGS__))
#define APPLY3(M, A, ...) EXPAND(M(A)), EXPAND(APPLY2(M, __VA_ARGS__))
#define APPLY4(M, A, ...) EXPAND(M(A)), EXPAND(APPLY3(M, __VA_ARGS__))
#define APPLY5(M, A, ...) EXPAND(M(A)), EXPAND(APPLY4(M, __VA_ARGS__))
#define APPLY6(M, A, ...) EXPAND(M(A)), EXPAND(APPLY5(M, __VA_ARGS__))
#define APPLY7(M, A, ...) EXPAND(M(A)), EXPAND(APPLY6(M, __VA_ARGS__))
#define APPLY8(M, A, ...) EXPAND(M(A)), EXPAND(APPLY7(M, __VA_ARGS__))
#define APPLY9(M, A, ...) EXPAND(M(A)), EXPAND(APPLY8(M, __VA_ARGS__))
#define APPLY10(M, A, ...) EXPAND(M(A)), EXPAND(APPLY9(M, __VA_ARGS__))
#define APPLY11(M, A, ...) EXPAND(M(A)), EXPAND(APPLY10(M, __VA_ARGS__))
#define APPLY12(M, A, ...) EXPAND(M(A)), EXPAND(APPLY11(M, __VA_ARGS__))
#define APPLY_N__(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, X,    \
                  ...)                                                         \
  APPLY##X
#define APPLY(M, ...)                                                          \
  EXPAND(EXPAND(APPLY_N__(M, __VA_ARGS__, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2,  \
                          1, 0))(M, __VA_ARGS__)

#define APPLY2_0(M, C) FUCK
#define APPLY2_1(M, C, A, ...) EXPAND(M(C, A))
#define APPLY2_2(M, C, A, ...)                                                 \
  EXPAND(M(C, A)), EXPAND(APPLY2_1(M, C, __VA_ARGS__))
#define APPLY2_3(M, C, A, ...)                                                 \
  EXPAND(M(C, A)), EXPAND(APPLY2_2(M, C, __VA_ARGS__))
#define APPLY2_4(M, C, A, ...)                                                 \
  EXPAND(M(C, A)), EXPAND(APPLY2_3(M, C, __VA_ARGS__))
#define APPLY2_5(M, C, A, ...)                                                 \
  EXPAND(M(C, A)), EXPAND(APPLY2_4(M, C, __VA_ARGS__))
#define APPLY2_6(M, C, A, ...)                                                 \
  EXPAND(M(C, A)), EXPAND(APPLY2_5(M, C, __VA_ARGS__))
#define APPLY2_7(M, C, A, ...)                                                 \
  EXPAND(M(C, A)), EXPAND(APPLY2_6(M, C, __VA_ARGS__))
#define APPLY2_8(M, C, A, ...)                                                 \
  EXPAND(M(C, A)), EXPAND(APPLY2_7(M, C, __VA_ARGS__))
#define APPLY2_9(M, C, A, ...)                                                 \
  EXPAND(M(C, A)), EXPAND(APPLY2_8(M, C, __VA_ARGS__))
#define APPLY2_10(M, C, A, ...)                                                \
  EXPAND(M(C, A)), EXPAND(APPLY2_9(M, C, __VA_ARGS__))
#define APPLY2_11(M, C, A, ...)                                                \
  EXPAND(M(C, A)), EXPAND(APPLY2_10(M, C, __VA_ARGS__))
#define APPLY2_12(M, C, A, ...)                                                \
  EXPAND(M(C, A)), EXPAND(APPLY2_11(M, C, __VA_ARGS__))
#define APPLY2_N__(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, X,   \
                   ...)                                                        \
  APPLY2_##X
#define APPLY_2(M, C, ...)                                                     \
  EXPAND(EXPAND(APPLY2_N__(M, __VA_ARGS__, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, \
                           1, 0))(M, C, __VA_ARGS__))