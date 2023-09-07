/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include "tensorflow/lite/micro/examples/network_tester/ifm.h"
#include "tensorflow/lite/micro/examples/network_tester/model.h"

#ifdef __GNUC__
#include <unistd.h> 
#else
#include <direct.h>
#endif

#ifndef MODEL
#define MODEL               1
#endif

#define MODEL_BASE_FOLDER   "../arm_m4_core/application/sabre_npu_tester/model"
#define TENSOR_ARENA_SIZE	(300 * 1024)
#define MAX_INPUTS			(2)

#ifndef MIN
#define MIN(x,y) ((x)<(y) ? (x) : (y))
#endif

alignas(16) uint8_t tensor_arena[TENSOR_ARENA_SIZE];

namespace model_nn {
    const uint8_t* get_buffer(int index);
}

namespace ifm {
    const uint8_t* get_buffer(int index);
    uint32_t get_num_buffers(void);
}



TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestInvoke) {
	const uint8_t* model_bytes = NULL;
	const uint8_t* input_data[MAX_INPUTS] = { 0 };

	// load model bytes 
	model_bytes = model_nn::get_buffer(0);

	// load on input buffers from memory
	int inputs_cnt = MIN(MAX_INPUTS, ifm::get_num_buffers());
	for (int i = 0; i < inputs_cnt; i++) {
		input_data[i] = ifm::get_buffer(i);
	}
	   
	const tflite::Model* model = ::tflite::GetModel(model_bytes);

	if (model->version() != TFLITE_SCHEMA_VERSION) {
		MicroPrintf(
			"Model provided is schema version %d not equal "
			"to supported version %d.\n",
			model->version(), TFLITE_SCHEMA_VERSION);
		return kTfLiteError;
	}

    static tflite::MicroMutableOpResolver<6> micro_op_resolver;
	micro_op_resolver.AddFullyConnected(tflite::Register_FULLY_CONNECTED_INT8());  
	micro_op_resolver.AddTranspose();
	micro_op_resolver.AddReshape();
	micro_op_resolver.AddSplit();
	micro_op_resolver.AddPack();
	micro_op_resolver.AddSoftmax(tflite::Register_SOFTMAX_INT8());

	tflite::MicroInterpreter interpreter(model, micro_op_resolver, tensor_arena, TENSOR_ARENA_SIZE);

	TfLiteStatus allocate_status = interpreter.AllocateTensors();
	if (allocate_status != kTfLiteOk) {
		MicroPrintf("Tensor allocation failed\n");
		return kTfLiteError;
	}

	for (size_t i = 0; i < interpreter.inputs_size(); ++i) {
	  TfLiteTensor* input = interpreter.input(i);
	  memcpy(input->data.data, &input_data[i], input->bytes);
	}
	
	TfLiteStatus invoke_status = interpreter.Invoke();
	if (invoke_status != kTfLiteOk) {
	  MicroPrintf("Invoke failed\n");
	  return kTfLiteError;
	}
	
	TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);	
	MicroPrintf("Ran successfully\n");
}

TF_LITE_MICRO_TESTS_END
