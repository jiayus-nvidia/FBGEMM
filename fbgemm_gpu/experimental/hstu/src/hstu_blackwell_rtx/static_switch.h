// Minimal static dispatch helpers used by the Blackwell RTX FP8 forward path.

#pragma once

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#ifdef HSTU_DISABLE_CONTEXT
#define CONTEXT_SWITCH(COND, CONST_NAME, ...) \
  [&] {                                       \
    constexpr static bool CONST_NAME = false; \
    return __VA_ARGS__();                     \
  }()
#else
#define CONTEXT_SWITCH BOOL_SWITCH
#endif

#ifdef HSTU_DISABLE_TARGET
#define TARGET_SWITCH(COND, CONST_NAME, ...)  \
  [&] {                                       \
    constexpr static bool CONST_NAME = false; \
    return __VA_ARGS__();                     \
  }()
#else
#define TARGET_SWITCH BOOL_SWITCH
#endif

#ifdef HSTU_DISABLE_RAB
#define RAB_SWITCH(COND, CONST_NAME, ...)     \
  [&] {                                       \
    constexpr static bool CONST_NAME = false; \
    return __VA_ARGS__();                     \
  }()
#else
#define RAB_SWITCH BOOL_SWITCH
#endif
