import multiprocessing
import time
import itertools

def brute_force_password_chunk(process_name, valid_characters, target_password, found_event, result_queue):
    """
    Fungsi untuk mencari password dalam rentang tertentu
    """
    print(f"Starting process: {process_name}")

    for password_length in range(1, len(target_password) + 1):
        for combination in itertools.product(valid_characters, repeat=password_length):
            attempt = ''.join(combination)
            if attempt == target_password:
                found_event.set()  # Menandai bahwa password telah ditemukan
                result_queue.put((attempt, process_name))  # Menyimpan hasil
                return

def brute_force_password_parallel(target_password, num_processes):
    """
    Fungsi untuk memulai pencarian password secara paralel
    """
    valid_characters = target_password
    max_password_length = len(target_password)

    found_event = multiprocessing.Event()
    result_queue = multiprocessing.Queue()

    processes = []
    for i in range(num_processes):
        process_name = f"Process#{i+1}"
        process = multiprocessing.Process(target=brute_force_password_chunk, args=(process_name, valid_characters, target_password, found_event, result_queue))
        processes.append(process)

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    if not result_queue.empty():
        found_password, process_name = result_queue.get()
        print(f"Password found: {found_password} in {process_name}")
    else:
        print("Password not found")

if __name__ == "__main__":
    target_password = input("Masukkan Password yang ingin dicari: ")
    num_processes = multiprocessing.cpu_count()

    print("Brute Force Password Finder using Multi-processing")
    print(f"CPU count: {num_processes}")

    start_time = time.perf_counter()

    brute_force_password_parallel(target_password, num_processes)

    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"Duration = {duration:.4f} seconds")
