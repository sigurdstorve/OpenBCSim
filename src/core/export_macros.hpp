// This makes DLL_PUBLIC and DLL_PRIVATE available when building with Visual Studio and GCC.
#if defined(_WIN32) || (_WIN64)
#   ifdef _DLL
#       define DLL_PUBLIC __declspec(dllexport)
#   else
#       define DLL_PUBLIC __declspec(dllimport)
#   endif
#   define DLL_PRIVATE
#else
#   define DLL_PUBLIC  __attribute__ ((visibility("default")))
#   define DLL_PRIVATE __attribute__ ((visibility("hidden")))
#endif
