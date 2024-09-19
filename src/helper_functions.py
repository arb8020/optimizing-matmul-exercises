import sys
import subprocess
import os

def load_solution_kernel(kernel_number):
    solution_path = f"./solutions/kernel_{kernel_number}_solution.cu"
    if os.path.exists(solution_path):
        with open(solution_path, 'r') as f:
            return f.read()
    else:
        print(f"Solution file not found: {solution_path}")
        return None

def compile_and_run_kernel(kernel_code, kernel_name, M, N, K):
    # Save the kernel code to a temporary .cu file
    with open(f"{kernel_name}.cpp", "w") as f:
        f.write(kernel_code)
    
    # We are assuming that the kernel source is a .cpp file for syntax highlighting purposes,
    # so we'll copy it to a .cu file before compilation
    subprocess.run(["cp", f"{kernel_name}.cpp", f"{kernel_name}.cu"])

    # Compile the kernel
    compile_result = subprocess.run(["nvcc", f"{kernel_name}.cu", "-o", kernel_name, "-lcublas"], capture_output=True, text=True)
    if compile_result.returncode != 0:
        print("Compilation failed:")
        print(compile_result.stderr)
        return None
    
    # Run the kernel
    run_result = subprocess.run([f"./{kernel_name}", str(M), str(N), str(K)], capture_output=True, text=True)
    if run_result.returncode != 0:
        print("Execution failed:")
        print(run_result.stderr)
        return None

    # Parse the output
    lines = run_result.stdout.strip().split('\n')
    is_correct = "The matrix multiplication is correct!" in lines[-1]
    # Locate the line with performance information
    performance_line = [line for line in lines if "Average Performance" in line][0]

    # Extract the performance value and remove the "GFLOPS" unit
    performance = float(performance_line.split()[-2])  # This will get the second-to-last element which is the numerical value 
    return is_correct, performance

def check_solution(kernel_number, user_code, M=4096, N=4096, K=4096):
    solution_code = load_solution_kernel(kernel_number)
    if not solution_code:
        print("Failed to load the solution. Please check if the solution file exists.")
        return

    print("Testing your implementation...")
    user_result = compile_and_run_kernel(user_code, f"kernel_{kernel_number}", M, N, K)
    
    if not user_result:
        print("Your implementation failed to compile or run.")
        return

    print("\nTesting reference implementation...")
    solution_result = compile_and_run_kernel(solution_code, f"solution_kernel_{kernel_number}", M, N, K)

    if not solution_result:
        print("The reference implementation failed to compile or run. Please contact the course administrators.")
        return

    user_correct, user_performance = user_result
    solution_correct, solution_performance = solution_result

    print("\nResults:")
    print(f"Your implementation - Correct: {user_correct}, Performance: {user_performance:.2f} GFLOPS")
    print(f"Reference implementation - Correct: {solution_correct}, Performance: {solution_performance:.2f} GFLOPS")

    if not user_correct:
        print("\nYour implementation produced incorrect results. Please check your code for errors.")
    elif user_performance < 0.9 * solution_performance:
        print("\nYour implementation is correct but significantly slower than the reference.")
        print("Consider optimizing your code further.")
    elif user_performance > 1.1 * solution_performance:
        print("\nGreat job! Your implementation is correct and faster than the reference.")
    else:
        print("\nWell done! Your implementation is correct and has similar performance to the reference.")

def check_roofline_calculation(user_function):

    solutions_path = os.path.join(os.getcwd(), "solutions", "roofline_solution.py")
    
    # Load the solution function using the absolute path
    try:
        spec = importlib.util.spec_from_file_location("roofline_solution", solutions_path)
        roofline_solution = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(roofline_solution)
        solution_function = roofline_solution.calculate_roofline
    except Exception as e:
        print(f"Error loading solution module: {e}")
        return


    # Test parameters
    m, n, k = 512, 512, 512
    memory_bandwidth = 320  # GB/s
    computational_performance = 65  # TFLOPs
    data_type_size = 2  # bytes (for float16)

    # Calculate results
    user_result = user_function(m, n, k, memory_bandwidth, computational_performance, data_type_size)
    solution_result = solution_function(m, n, k, memory_bandwidth, computational_performance, data_type_size)

    # Compare results
    all_correct = True
    for key in solution_result:
        if user_result[key] is None:
            print(f"Missing calculation for {key}")
            all_correct = False
        elif abs(user_result[key] - solution_result[key]) > 1e-6:
            print(f"Mismatch in {key}:")
            print(f"Your result: {user_result[key]}")
            print(f"Expected result: {solution_result[key]}")
            all_correct = False

    if all_correct:
        print("Congratulations! Your roofline calculation is correct.")
    else:
        print("There are some discrepancies in your calculation. Please check your implementation.")

