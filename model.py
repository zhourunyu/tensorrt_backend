import numpy as np
import os
import gc
import math

import triton_python_backend_utils as pb_utils

import tensorrt as trt
import pycuda.driver as cuda
from pydantic import BaseModel, Field, Json

class ModelConfig(BaseModel):
    class Input(BaseModel):
        name: str
        data_type: str
        dims: list[int]
        optional: bool = False

    class Output(BaseModel):
        name: str
        data_type: str
        dims: list[int]

    name: str
    platform: str
    backend: str
    max_batch_size: int
    input: list[Input]
    output: list[Output]

class TritonArguments(BaseModel):
    config: Json[ModelConfig] = Field(alias="model_config")
    instance_kind: str = Field(alias="model_instance_kind")
    instance_name: str = Field(alias="model_instance_name")
    instance_device_id: int = Field(alias="model_instance_device_id")
    repository: str = Field(alias="model_repository")
    version: str = Field(alias="model_version")
    name: str = Field(alias="model_name")

def get_size(dtype: str):
    return np.dtype(pb_utils.triton_string_to_numpy(dtype)).itemsize

class TritonPythonModel:
    def initialize(self, args):
        # parse the arguments
        args = TritonArguments.model_validate(args)
        self.config = args.config
        model_file = os.path.join(pb_utils.get_model_dir(), f"{args.name}.engine")
        assert os.path.isfile(model_file), f"Model file {model_file} does not exist."
        self.logger = pb_utils.Logger

        cuda.init()
        device_id = int(args.instance_name.split("_")[-2])
        self.device = cuda.Device(device_id)
        self.ctx = self.device.make_context()
        self.stream = cuda.Stream()
        # init trt engine and load model
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(model_file, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.inputs = self.config.input
        self.outputs = self.config.output

        if self.config.max_batch_size > 0:
            self.logger.log_info(f"[tensorrt] dynamic batch enabled with max batch size {self.config.max_batch_size}")

        # malloc input and output buffers
        self.input_buffers = []
        for input in self.inputs:
            index = self.engine.get_binding_index(input.name)
            max_shape = self.engine.get_profile_shape(0, index)[2]
            d_ptr = cuda.mem_alloc(math.prod(max_shape) * get_size(input.data_type))
            self.context.set_tensor_address(input.name, int(d_ptr))
            self.input_buffers.append(d_ptr)
            self.context.set_input_shape(input.name, max_shape) # set max shape to get max output shape
        self.output_buffers = []
        for output in self.outputs:
            index = self.engine.get_binding_index(output.name)
            max_shape = self.context.get_tensor_shape(output.name)
            d_ptr = cuda.mem_alloc(math.prod(max_shape) * get_size(output.data_type))
            self.context.set_tensor_address(output.name, int(d_ptr))
            self.output_buffers.append(d_ptr)

    def execute(self, requests: list) -> list:
        self.logger.log_verbose(f"[tensorrt] received {len(requests)} requests")

        # dynamic batching if enabled
        if self.config.max_batch_size > 0:
            return self._execute_batch(requests)

        responses: list[pb_utils.InferenceResponse] = []
        for request in requests:
            # copy input data to device
            for input, buffer in zip(self.inputs, self.input_buffers):
                input_data = pb_utils.get_input_tensor_by_name(request, input.name).as_numpy()
                cuda.memcpy_htod_async(buffer, input_data, self.stream)
                self.context.set_input_shape(input.name, input_data.shape)

            self.context.execute_async_v3(self.stream.handle)

            # retrieve output data
            output_tensors = []
            for output, buffer in zip(self.outputs, self.output_buffers):
                output_shape = self.context.get_tensor_shape(output.name)
                output_data = np.empty(output_shape, dtype=pb_utils.triton_string_to_numpy(output.data_type))
                cuda.memcpy_dtoh_async(output_data, buffer, self.stream)
                self.stream.synchronize()
                output_tensors.append(pb_utils.Tensor(output.name, output_data))

            response = pb_utils.InferenceResponse(output_tensors=output_tensors)
            responses.append(response)
        return responses

    def _execute_batch(self, requests: list) -> list:
        responses: list[pb_utils.InferenceResponse] = []

        batch_sizes: list[int] = []
        # gather and concatenate inputs across all requests
        for input, buffer in zip(self.inputs, self.input_buffers):
            inputs = [pb_utils.get_input_tensor_by_name(request, input.name).as_numpy() for request in requests]
            input_data = np.concatenate(inputs, axis=0)
            cuda.memcpy_htod_async(buffer, input_data, self.stream)
            self.context.set_input_shape(input.name, input_data.shape)
            if not batch_sizes:
                batch_sizes = [inp.shape[0] for inp in inputs]

        # batch inference
        self.context.execute_async_v3(self.stream.handle)
        output_arraies: list[np.ndarray] = []
        for output, buffer in zip(self.outputs, self.output_buffers):
            output_shape = self.context.get_tensor_shape(output.name)
            output_data = np.empty(output_shape, dtype=pb_utils.triton_string_to_numpy(output.data_type))
            cuda.memcpy_dtoh_async(output_data, buffer, self.stream)
            output_arraies.append(output_data)
        self.stream.synchronize()

        # scatter outputs for each request
        current_batch_index = 0
        for batch in batch_sizes:
            output_tensors: list[pb_utils.Tensor] = []
            for output, output_array in zip(self.outputs, output_arraies):
                output_slice = output_array[current_batch_index:current_batch_index + batch]
                output_tensor = pb_utils.Tensor(output.name, output_slice)
                output_tensors.append(output_tensor)
            response = pb_utils.InferenceResponse(output_tensors=output_tensors)
            responses.append(response)
            current_batch_index += batch

        return responses

    def finalize(self):
        for buffer in self.input_buffers + self.output_buffers:
            buffer.free()
        self.logger.log_info("[tensorrt] Running Garbage Collector on finalize...")
        gc.collect()
        self.logger.log_info("[tensorrt] Garbage Collector on finalize... done")