import time
import itertools

def brute_force_password(target_password):
    """
    Fungsi brute force untuk mencari password secara sekuensial
    """
    print("Starting brute force search sequentially...")

    start_time = time.time()

    valid_characters = target_password

    for password_length in range(1, len(target_password) + 1):
        for combination in itertools.product(valid_characters, repeat=password_length):
            attempt = ''.join(combination)
            print(f"Trying: {attempt}")
            # Memeriksa apakah attempt = password yang dicari
            if attempt == target_password:
                end_time = time.time()
                duration = end_time - start_time
                print(f"Password found: {attempt}")
                print(f"Duration = {duration:.4f} seconds")
                return

    print("Password not found")

if __name__ == "__main__":
    target_password = input("Masukkan Password yang ingin dicari: ")
    brute_force_password(target_password)
