"""Settings shared by the scripts in this folder. Everything comes from environment variables, so nothing is tied to
one machine.

  CUT_ROOT        Working directory. The scripts read and write data/, hyper/, step4/, models/ and logs/ under it
                  (layout in README.md). Default: the current directory.
  THINKING_DIR    Hugging Face folder of Qwen/Qwen3-4B-Thinking-2507.   Default: $CUT_ROOT/Qwen3-4B-Thinking-2507
  INSTRUCT_DIR    Hugging Face folder of Qwen/Qwen3-4B-Instruct-2507.   Default: $CUT_ROOT/Qwen3-4B-Instruct-2507
  LLAMA_SERVER    llama.cpp server binary used by the evaluation scripts.            Default: llama-server (on PATH)
  CONVERT_SCRIPT  llama.cpp's convert_hf_to_gguf.py, used by scripts that build a GGUF.
                  Default: $CUT_ROOT/llama.cpp/convert_hf_to_gguf.py

The scripts choose no GPU and set no library paths. Use CUDA_VISIBLE_DEVICES, LD_LIBRARY_PATH and so on in your own
shell. Tested with llama.cpp build b11669 and with the Python packages listed in README.md.
"""
import os

ROOT = os.path.abspath(os.environ.get("CUT_ROOT", os.getcwd()))
THINKING_DIR = os.environ.get("THINKING_DIR", os.path.join(ROOT, "Qwen3-4B-Thinking-2507"))
INSTRUCT_DIR = os.environ.get("INSTRUCT_DIR", os.path.join(ROOT, "Qwen3-4B-Instruct-2507"))
LLAMA_SERVER = os.environ.get("LLAMA_SERVER", "llama-server")
CONVERT_SCRIPT = os.environ.get("CONVERT_SCRIPT", os.path.join(ROOT, "llama.cpp", "convert_hf_to_gguf.py"))
