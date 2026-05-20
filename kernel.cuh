#pragma once

// #define EXPL_POP

#include "deviceFunctions.cuh"
#ifdef EXPL_POP
#include "irbcPop.cuh"
#include "kernelPop.cuh"
#else
#include "irbcAccum.cuh"
#include "kernelAccum.cuh"
#endif
