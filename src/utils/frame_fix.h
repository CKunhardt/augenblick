#pragma once

#ifdef __CUDACC__
// Forward declarations for Python C API structs to avoid incomplete type errors
#ifndef PyFrameObject_DEFINED
#define PyFrameObject_DEFINED
struct _frame;
typedef struct _frame PyFrameObject;
#endif

#ifndef PyThreadState_DEFINED
#define PyThreadState_DEFINED
struct _ts;
typedef struct _ts PyThreadState;
#endif
#endif
