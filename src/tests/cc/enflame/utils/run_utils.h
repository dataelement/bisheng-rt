#pragma once
#include <vector>

#include "TopsInference/TopsInferRuntime.h"
#include "tops_stream_struct.h"

int engine_run_async_d2d(
    TopsInference::IEngine* engine, TopsStreamStruct* stream_struct,
    std::vector<void*>& inputs, std::vector<void*>& outputs, bool use_api2);
