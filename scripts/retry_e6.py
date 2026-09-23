"""Three models whose server died partway through a fill (WinError 10054): the context that
allocated at load could not survive the fill itself. Retry each one step lower."""
import wave10 as w
w.window_sweep("qwen3-32b-q6_k-retry", w.M + r"\unsloth\Qwen3-32B-GGUF\Qwen3-32B-Q6_K.gguf", [16384, 12288, 8192])
w.window_sweep("seed-oss-36b-q6_k-retry", w.M + r"\lmstudio-community\Seed-OSS-36B-Instruct-GGUF\Seed-OSS-36B-Instruct-Q6_K.gguf", [8192, 6144])
w.window_sweep("llama33-70b-q2_k_xl-retry", w.M + r"\unsloth\Llama-3.3-70B-Instruct-GGUF\Llama-3.3-70B-Instruct-UD-Q2_K_XL.gguf", [12288, 8192, 6144])
