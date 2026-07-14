import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time
import os
import subprocess
import signal
import re

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def get_data_for_reconstruct_model():

    data_folder_path = "/home/udayanga123/jetson_nano_code/onnx_models/model3_onnx_models_data/formatted_data/evCIVIL/"
    temp_list = []
    files = os.listdir(data_folder_path)
    for npz_file in files:
        full_path = os.path.join(data_folder_path,npz_file)
        data = np.load(full_path)
        print("ev img shape ",data.shape)
        temp_list.append(data)
    arr = np.stack(temp_list)
    print("after stacking ",arr.shape)
    repeats = 512 // arr.shape[0]
    test_data = np.repeat(arr,repeats,axis=0)
    print("after repeating ",test_data.shape)
    return test_data

def measure_idle_power():
    print(f"\n📊 Measuring idle power for {IDLE_DURATION}s...")
    proc, log = start_tegrastats(IDLE_LOG)
    time.sleep(IDLE_DURATION)
    stop_tegrastats(proc, log)
    avg_idle_power, _ = parse_power_log(IDLE_LOG)
    print(f"✅ Average Idle Power: {avg_idle_power:.2f} mW")
    return avg_idle_power, _

def start_tegrastats(logfile):
    log = open(logfile, "w")
    proc = subprocess.Popen(
        ["tegrastats", "--interval", f"{int(SAMPLE_INTERVAL*1000)}"],
        stdout=log,
        stderr=subprocess.DEVNULL
    )
    return proc, log

def stop_tegrastats(proc, log):
    os.kill(proc.pid, signal.SIGTERM)
    proc.wait()
    log.close()

def parse_power_log(log_file):
    power_samples = []
    with open(log_file, "r") as f:
        for line in f:
            # Capture VDD_IN value (mW)
            match = re.search(r'VDD_IN (\d+)', line)
            if match:
                power_samples.append(int(match.group(1)))

    if not power_samples:
        print("⚠ No power data found in log.")
        return 0, 0

    avg_power_mw = sum(power_samples) / len(power_samples)
    duration_sec = len(power_samples) * SAMPLE_INTERVAL
    energy_joules = (avg_power_mw / 1000) * duration_sec  # mW -> W -> J
    return avg_power_mw, energy_joules
    
    
def load_engine(engine_path):
    """
    Load a TensorRT engine from file (Jetson-compatible).
    """
    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"Engine file not found: {engine_path}")

    with open(engine_path, "rb") as f:
        engine_data = f.read()

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_data)

    if engine is None:
        raise RuntimeError("Failed to deserialize CUDA engine. "
                           "Make sure the engine was built on this device (Jetson Orin Nano).")

    return engine

import tensorrt as trt
import pycuda.driver as cuda
import numpy as np

def infer_batch_reconstruct(engine, input_tensors_dict):
    context = engine.create_execution_context()
    stream = cuda.Stream()

    host_outputs = {}
    device_memories = {}

    # ✅ Set input shapes (NEW API)
    for name, data in input_tensors_dict.items():
        context.set_input_shape(name, data.shape)

    # ✅ Allocate buffers
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)  # INPUT or OUTPUT
        dtype = trt.nptype(engine.get_tensor_dtype(name))

        shape = tuple(context.get_tensor_shape(name))
        size = int(np.prod(shape))

        host_mem = cuda.pagelocked_empty(size, dtype).reshape(shape)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        device_memories[name] = device_mem

        if mode == trt.TensorIOMode.INPUT:
            data = input_tensors_dict[name].astype(dtype)
            np.copyto(host_mem, data)
            cuda.memcpy_htod_async(device_mem, host_mem, stream)
        else:
            host_outputs[name] = host_mem

        # ✅ Set tensor address (NEW API)
        context.set_tensor_address(name, int(device_mem))

    # 🚀 Run inference (NEW API)
    context.execute_async_v3(stream_handle=stream.handle)

    # ✅ Copy outputs back
    for name in host_outputs:
        cuda.memcpy_dtoh_async(host_outputs[name], device_memories[name], stream)

    stream.synchronize()

    return host_outputs

def build_engine_reconstruct(
    onnx_path,
    engine_path="reconstruct_ev_gen1_fp16.engine",
    use_fp16=True
):
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(
             1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
         ) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        config = builder.create_builder_config()

        # ✅ Orin Nano: allow larger workspace (important)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

        # ✅ Enable FP16 only if supported
        if use_fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("FP16 enabled")
        else:
            print("FP16 not supported, using FP32")

        print(f"Loading ONNX file: {onnx_path}")
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                print("❌ Failed to parse ONNX")
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                return None

        profile = builder.create_optimization_profile()

        # ✅ Input shapes (NCHW)
        dynamic_shapes = {
            "event_tensor": (3, 192, 256),
            "h0_0": (64, 48, 64),
            "c0_0": (64, 48, 64),
            "h0_1": (128, 24, 32),
            "c0_1": (128, 24, 32),
            "h0_2": (256, 12, 16),
            "c0_2": (256, 12, 16),
        }

        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            name = tensor.name

            if name not in dynamic_shapes:
                raise ValueError(f"Missing shape config for input: {name}")

            c, h, w = dynamic_shapes[name]

            profile.set_shape(
                name,
                min=(1, c, h, w),
                opt=(2, c, h, w),   # ✅ safer for Orin memory
                max=(4, c, h, w)    # ✅ reduced max batch to avoid OOM
            )

        config.add_optimization_profile(profile)

        print("🚀 Building TensorRT engine for Orin Nano...")

        # ✅ Modern API (IMPORTANT for JetPack 5/6)
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            print("❌ Engine build failed")
            return None

        # Save engine
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)

        print(f"✅ Engine saved to {engine_path}")

        # Optional: deserialize to return engine object
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(serialized_engine)

        return engine
        
# Constants
IDLE_DURATION = 5           # seconds to measure idle power
SAMPLE_INTERVAL = 0.5       # seconds between tegrastats samples
IDLE_LOG = "idle_power.log"
LOOP_LOG = "inference_power.log"
BATCH_SIZE = 1

def main(tensor_rt_model):
    
    if os.path.exists(IDLE_LOG):
        os.remove(IDLE_LOG)
    if os.path.exists(LOOP_LOG):
        os.remove(LOOP_LOG)
        
    # Load TensorRT engine
    engine = load_engine(tensor_rt_model)

    # Measure idle power
    idle_power, _ = measure_idle_power()

    # Load test data
    test_data = get_data_for_reconstruct_model()
    num_samples = (test_data.shape[0] // BATCH_SIZE) * BATCH_SIZE
    print(f"[INFO] Running inference on {num_samples} samples with batch size {BATCH_SIZE}")

    # Start tegrastats logging for inference
    print("📊 Starting tegrastats logging for inference...")
    proc, log = start_tegrastats(LOOP_LOG)

    # Run inference and measure time
    start_time = time.time()
    
    for i in range(0, num_samples, BATCH_SIZE):
        batch = test_data[i:i+BATCH_SIZE]
        #print("current_batch ",batch.shape)
        #output = infer_batch(engine, batch)
        #all_outputs.append(output)
         
        
        h0_0 = np.zeros((BATCH_SIZE, 64, 48, 64), dtype=np.float32)
        c0_0 = np.zeros((BATCH_SIZE, 64, 48, 64), dtype=np.float32)
        h0_1 = np.zeros((BATCH_SIZE, 128, 24, 32), dtype=np.float32)
        c0_1 = np.zeros((BATCH_SIZE, 128, 24, 32), dtype=np.float32)
        h0_2 = np.zeros((BATCH_SIZE, 256, 12, 16), dtype=np.float32)
        c0_2 = np.zeros((BATCH_SIZE, 256, 12, 16), dtype=np.float32)
   
        for j in range(5):
            
            current_batch = batch[:,j,:,:,:]
            #print("current_batch ",current_batch.shape)
        
            input_tensors = {
	        "event_tensor": current_batch,
	        "h0_0": h0_0,
	        "c0_0": c0_0,
	        "h0_1": h0_1,
	        "c0_1": c0_1,
	        "h0_2": h0_2,
	        "c0_2": c0_2,
	    }

            output = infer_batch_reconstruct(engine, input_tensors)
            #print("output keys ",output.keys())
            output_tensor = output["detector_output"]
            h0_0 = output["h0_0_out"]
            c0_0 = output["c0_0_out"]
            h0_1 = output["h0_1_out"]
            c0_1 = output["c0_1_out"]
            h0_2 = output["h0_2_out"]
            c0_2 = output["c0_2_out"]
            #print("output tensor shape ",output_tensor.shape)

    end_time = time.time()
    elapsed = end_time - start_time

    # Stop tegrastats
    stop_tegrastats(proc, log)

    # Parse power log
    full_power, total_energy_joules = parse_power_log(LOOP_LOG)

    # Compute throughput
    throughput = num_samples / elapsed

    # Compute per-sample energy (subtract idle contribution)
    inference_energy_j = total_energy_joules - (idle_power / 1000 * elapsed)
    per_sample_energy_j = inference_energy_j / num_samples
    per_sample_energy_mj = per_sample_energy_j * 1000  # in mJ

    # Print results
    print("\n📊 Inference Benchmark Results:")
    print(f"Total samples: {num_samples}")
    print(f"Elapsed time: {elapsed:.3f} s")
    print(f"Throughput: {throughput:.2f} samples/sec")
    print(f"Total energy consumed (J): {inference_energy_j:.3f} J")
    print(f"Per-sample energy: {per_sample_energy_j:.6f} J ({per_sample_energy_mj:.2f} mJ)")
    print(f"Average inference power: {full_power:.2f} mW")
    print(f"Idle power: {idle_power:.2f} mW")
    
    del engine
    runtime = None
    
if __name__ == "__main__":
     
    onnx_path = "/home/udayanga123/jetson_nano_code/onnx_models/model3_onnx_models_data/with_recurrent/reconstruct_evCIVIL_with_recurrent_detector.onnx"
    tensorrt_eng_path = "/home/udayanga123/jetson_nano_code/onnx_models/model3_onnx_models_data/with_recurrent/reconstruct_evCIVIL_with_recurrent_detector.engine"
    #build_engine_reconstruct(onnx_path, engine_path=tensorrt_eng_path, use_fp16=True)
    main(tensorrt_eng_path)
