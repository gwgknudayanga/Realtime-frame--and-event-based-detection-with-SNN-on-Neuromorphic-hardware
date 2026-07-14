import tensorrt as trt
import os

# Initialize the Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, engine_file_path="model.engine"):
    # 1. Initialize Builder, Network, and Parser
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()

    # 2. Set Memory Pool Limit (Workspace) - 256MB is usually safe for Orin Nano
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 * 1024 * 1024)

    # 3. Enable FP16 (The "Speed Boost" for Jetson)
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("--- FP16 mode enabled ---")

    # 4. Parse the ONNX model
    print(f"Reading ONNX file: {onnx_file_path}")
    if not os.path.exists(onnx_file_path):
        print(f"ERROR: File {onnx_file_path} not found.")
        return None

    with open(onnx_file_path, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # 5. Handle Dynamic Shapes (The fix for your error)
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    print(f"Detected input name: '{input_name}' with shape {input_tensor.shape}")

    # Set shapes: (Min, Optimal, Max)
    # Adjust (1, 3, 224, 224) if your model uses a different resolution
    standard_shape = (-1, 2, 192, 256)
    profile.set_shape(input_name, standard_shape, standard_shape, standard_shape)
    config.add_optimization_profile(profile)

    # 6. Build the Engine
    print("Building Engine... (This takes a few minutes on Jetson)")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("ERROR: Engine build failed. Check for incompatible layers.")
        return None

    # 7. Save the Engine
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)
    
    print(f"SUCCESS: Engine saved as {engine_file_path}")
    print(f"File size: {os.path.getsize(engine_file_path) / (1024*1024):.2f} MB")
    return True

if __name__ == "__main__":
    # Specify your ONNX filename here
    ONNX_FILE = "prop_gen1_model1.onnx"
    ENGINE_FILE = "prop_gen1_model1.engine"
    
    build_engine(ONNX_FILE, ENGINE_FILE)
