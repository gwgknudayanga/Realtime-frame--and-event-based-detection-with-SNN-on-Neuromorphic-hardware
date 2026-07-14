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
    
def infer_batch(engine, input_batch):
    context = engine.create_execution_context()

    # Get tensor names (new API)
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    # Set input shape
    context.set_input_shape(input_name, input_batch.shape)

    # Get output shape
    output_shape = context.get_tensor_shape(output_name)

    # Allocate host memory
    h_input = cuda.pagelocked_empty(input_batch.size, dtype=np.float32)
    np.copyto(h_input, input_batch.ravel())

    h_output = cuda.pagelocked_empty(int(np.prod(output_shape)), dtype=np.float32)

    # Allocate device memory
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    # Create stream
    stream = cuda.Stream()

    # Transfer input
    cuda.memcpy_htod_async(d_input, h_input, stream)

    # Bindings (IMPORTANT: use tensor addresses in new API)
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    # Run inference
    context.execute_async_v3(stream_handle=stream.handle)

    # Copy output
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

    return np.reshape(h_output, output_shape)
    
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
    
def get_pascalvoc_img_data():

    data_path = "/home/udayanga123/jetson_nano_code/selected_model1_and_model2_data/data_and_weights1/voc/PASCALVOC_FINAL/test/images"
    
    images = os.listdir(data_path)
    imgs_npy_lst = []
    for image in images:
        img = cv2.imread(os.path.join(data_path,image),cv2.IMREAD_COLOR)
        #print("image name ",image)
        #print("image name ",image,"  ",img.shape)
        img_resized = cv2.resize(img,(224,224))/255.0
        #print("sssssss ",image,"  ",img_resized.shape)
        img_resized = np.transpose(img_resized,(2,0,1))
        imgs_npy_lst.append(img_resized)
    imgs_npy_tensor = np.stack(imgs_npy_lst)
    return imgs_npy_tensor
    
def get_image_dataset_evCIVIL():

    data_path = "/home/udayanga123/jetson_nano_code/selected_model1_and_model2_data/data_and_weights1/evCIVIL_img/latest_dataset_images/val/images/"
    images = os.listdir(data_path)
    imgs_npy_lst = []
    for image in images:
        img = cv2.imread(os.path.join(data_path,image),cv2.IMREAD_GRAYSCALE)
        #print("image name ",image,"  ",img.shape)
        img_resized = cv2.resize(img,(256,192))/255.0 #resize API expects (width,height) convension
        #print("sssssss ",image,"  ",img_resized.shape)
        img_resized = np.expand_dims(img_resized,axis = 0)
        imgs_npy_lst.append(img_resized)
    imgs_npy_tensor = np.stack(imgs_npy_lst)
    return imgs_npy_tensor

def get_event_dataset_evCIVIL():

    path = "/home/udayanga123/jetson_nano_code/selected_model1_and_model2_data/data_and_weights1/evCIVIL_event/latest_dataset/"
    csv_file_name = "/home/udayanga123/jetson_nano_code/selected_model1_and_model2_data/data_and_weights1/evCIVIL_event/latest_dataset/test_files_event_based.txt"

    src_files = os.listdir()
    dest_files = []
    
    with open(os.path.join(path,csv_file_name),"r") as f:
        files = f.readlines()
        dest_files = []
        for filee in files:
            data = np.load(os.path.join(path,filee.rstrip()))
            print(data.keys())
            ev_img = data["ev_color_img"]
            #print("original image ",ev_img.shape)
            ev_img = np.transpose(ev_img,(2,0,1))
            resized_channels = []
            for channel in range(ev_img.shape[0]):
                resized = cv2.resize(ev_img[channel],(256,192),interpolation=cv2.INTER_LINEAR)
                resized_channels.append(resized)
            ev_img = np.stack(resized_channels,axis=0)
            #print("ev_img ",ev_img.shape)
            dest_files.append(ev_img)
        dest_arr = np.stack(dest_files,axis=0)
        #print("dest arr shape ",dest_arr.shape)
        return dest_arr
        
def get_event_dataset_GEN1():
    
    path = "/home/udayanga123/jetson_nano_code/selected_model1_and_model2_data/data_and_weights1/Prophesee_GEN1/prophesee_processed_dataset/"
    csv_file_name = "/home/udayanga123/jetson_nano_code/selected_model1_and_model2_data/data_and_weights1/Prophesee_GEN1/prophesee_processed_dataset/test_files_event_based.txt"
    
    src_files = os.listdir()
    dest_files = []
    
    with open(os.path.join(path,csv_file_name),"r") as f:
        files = f.readlines()
        dest_files = []
        for filee in files:
            data = np.load(os.path.join(path,filee.rstrip()))
            print(data.keys())
            ev_img = data["ev_color_img"]
            #print("original image ",ev_img.shape)
            ev_img = np.transpose(ev_img,(2,0,1))
            resized_channels = []
            for channel in range(ev_img.shape[0]):
                resized = cv2.resize(ev_img[channel],(256,192),interpolation=cv2.INTER_LINEAR)
                resized_channels.append(resized)
            ev_img = np.stack(resized_channels,axis=0)
            #print("ev_img ",ev_img.shape)
            dest_files.append(ev_img)
        dest_arr = np.stack(dest_files,axis=0)
        #print("dest arr shape ",dest_arr.shape)
        return dest_arr



def build_engine(onnx_path, engine_path="model_fp16.engine", fp16=True):
    # Using 'with' blocks is good practice for resource management
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()

    # FIX 1: Modern Workspace setting (256 MiB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28) 

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    print(f"Loading ONNX file: {onnx_path}")
    with open(onnx_path, "rb") as model_file:
        if not parser.parse(model_file.read()):
            print("Failed to parse the ONNX file.")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return None

    # Handling Optimization Profile (Keep your specific shapes)
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    profile = builder.create_optimization_profile()
    
    # Your specific shape: [Batch, 2, 192, 256]
    profile.set_shape(
        input_name,
        min=(1,3,192,256),
        opt=(1,3,192,256),
        max=(16,3,192,256)
    )
    config.add_optimization_profile(profile)

    print("Building TensorRT engine (FP16)...")
    
    # FIX 2: Modern Build and Serialization method
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("[ERROR] Engine build failed.")
        return None

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
        
    print(f"Engine saved to {engine_path}")
    return True # Returns True on success
    
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
    test_data = get_reconstruct_dataset()    #get_data()
    num_samples = (test_data.shape[0] // BATCH_SIZE) * BATCH_SIZE
    print(f"[INFO] Running inference on {num_samples} samples with batch size {BATCH_SIZE}")

    # Start tegrastats logging for inference
    print("📊 Starting tegrastats logging for inference...")
    proc, log = start_tegrastats(LOOP_LOG)

    # Run inference and measure time
    start_time = time.time()
    all_outputs = []
    for i in range(0, num_samples, BATCH_SIZE):
        batch = test_data[i:i+BATCH_SIZE]
        output = infer_batch(engine, batch)
        all_outputs.append(output)
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
    
# Constants
IDLE_DURATION = 5           # seconds to measure idle power
SAMPLE_INTERVAL = 0.5       # seconds between tegrastats samples
IDLE_LOG = "idle_power.log"
LOOP_LOG = "inference_power.log"
BATCH_SIZE = 1

def get_reconstruct_dataset():
    #for model3 without recurrents
    
    data_path = "/home/udayanga123/jetson_nano_code/onnx_models/model3_onnx_models_data/formatted_data/evCIVIL/"
    temp_list = []
    files = os.listdir(data_path)
    for npz_file in files:
        full_path = os.path.join(data_path,npz_file)
        data = np.load(full_path)
        print("ev img shape ",data.shape)
        temp_list.append(data)
    arr = np.stack(temp_list)
    #print("after stacking early ",arr.shape)
    arr = arr[:,2,:,:,:]
    #print("after stacking ",arr.shape)
    repeats = 2048 // arr.shape[0]
    test_data = np.repeat(arr,repeats,axis=0)
    return test_data

def get_data():
    #arr = get_pascalvoc_img_data()
    arr = get_event_dataset_GEN1()
    #arr = get_image_dataset_evCIVIL()
    #arr = get_event_dataset_evCIVIL()
    repeats = 2048 // arr.shape[0]
    test_data = np.repeat(arr,repeats,axis=0)
    print("tensor shape ",test_data.shape)
    return test_data
    
if __name__ == "__main__":
     
    onnx_path = "/home/udayanga123/jetson_nano_code/onnx_models/model3_onnx_models_data/without_recurrent/reconstruct_evCIVIL_no_recurrent_detector.onnx"
    tensorrt_eng_path = "/home/udayanga123/jetson_nano_code/updated_examples/tensorrt_models/reconstruct_evCIVIL_no_recurrent_detector.engine"
    #build_engine(onnx_path, engine_path=tensorrt_eng_path, fp16=True)
    main(tensorrt_eng_path)
