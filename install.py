import launch
import os
import pkg_resources
import sys
from tqdm import tqdm
import urllib.request

req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")

models_dir = os.path.abspath("models/roop")
model_url = "https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx"
model_name = os.path.basename(model_url)
model_path = os.path.join(models_dir, model_name)

def download(url, path):
    request = urllib.request.urlopen(url)
    total = int(request.headers.get('Content-Length', 0))
    with tqdm(total=total, desc='Downloading', unit='B', unit_scale=True, unit_divisor=1024) as progress:
        urllib.request.urlretrieve(url, path, reporthook=lambda count, block_size, total_size: progress.update(block_size))

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(model_path):
    download(model_url, model_path)

print("Checking roop requirements")
with open(req_file) as file:
    for package in file:
        try:
            python = sys.executable
            package = package.strip()

            if not launch.is_installed(package):
                print(f"Install {package}")
                launch.run_pip(
                    f"install {package}", f"sd-webui-roop requirement: {package}"
                )
            elif "==" in package:
                package_name, package_version = package.split("==")
                installed_version = pkg_resources.get_distribution(package_name).version
                if installed_version != package_version:
                    print(
                        f"Install {package}, {installed_version} vs {package_version}"
                    )
                    launch.run_pip(
                        f"install {package}",
                        f"sd-webui-roop requirement: changing {package_name} version from {installed_version} to {package_version}",
                    )

        except Exception as e:
            print(e)
            print(f"Warning: Failed to install {package}, roop will not work.")
            raise e
