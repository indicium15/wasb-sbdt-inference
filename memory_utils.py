import torch

def estimate_batch_size(model, input_shape=(9, 288, 512), max_cap=32, safety_factor=0.8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device).eval()

    dummy_input = torch.randn(1, *input_shape, device=device)
    torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode(), torch.cuda.amp.autocast():
        _ = model(dummy_input)

    mem_per_sample = torch.cuda.max_memory_allocated(device)
    free_mem, _ = torch.cuda.mem_get_info(device)
    free_mem = int(free_mem * safety_factor)

    return max(1, min(free_mem // mem_per_sample, max_cap))
