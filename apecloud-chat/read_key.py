# read_key.py
def read_key_from_file(file_name):
    with open(file_name, 'rb') as f:
        key = f.read()
    return key.strip()

if __name__ == "__main__":
    key_file = "key.txt"
    key = read_key_from_file(key_file)
    print(f"Key from {key_file}: {key}")
