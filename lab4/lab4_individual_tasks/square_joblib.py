# 11. Паралельне піднесення до квадрату: Паралельно піднести кожен елемент великого маасиву до квадрату,
# розділивши масив на частини

import numpy as np
from joblib import Parallel, delayed
import time

def square(x):
    time.sleep(0.0001)
    return x * x

def process_chunk(chunk):
    return [square(x) for x in chunk]

if __name__ == "__main__":
    array_size = 100_000
    data = np.random.rand(array_size)
    data = np.round(data, 2)


    # --- Послідовне виконання ---
    start_seq = time.time()
    result_seq = [square(x) for x in data]
    end_seq = time.time()
    print(f"Час послідовного виконання: {end_seq - start_seq:.4f} секунд")

    # --- Паралельне виконання з joblib ---
    num_jobs = -1  # Використовує всі доступні ядра
    chunks = np.array_split(data, 8)  # або більше/менше — для тесту

    start_par = time.time()
    result_chunks = Parallel(n_jobs=num_jobs)(
        delayed(process_chunk)(chunk) for chunk in chunks
    )
    result_par = [x for chunk in result_chunks for x in chunk]
    result_par = np.round(result_par, 5)
    end_par = time.time()

    print(f"Час паралельного виконання (joblib): {end_par - start_par:.4f} секунд")
    print("Результати однакові:", np.allclose(result_seq, result_par))
    print(f'Оригінальний масив (10 ел.): {data[:10]}')
    print(f"Піднесений до квадрату (10 ел.): {result_par[:10]}")