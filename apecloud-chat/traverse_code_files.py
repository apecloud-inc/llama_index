import os
import fnmatch

def list_go_files_recursive(directory):
    go_files = []

    for root, _, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, "*.go"):
            go_files.append(os.path.join(root, filename))

    return go_files

def main():
    directory = "/root/kubeblocks"

    go_files = list_go_files_recursive(directory)

    print("Golang files:")
    for go_file in go_files:
        print(go_file)

if __name__ == "__main__":
    main()

