import stim
import numpy as np
import pymatching as pm
from function_calls import build_spacetime, make_cnn_input
import os

def generate_dataset(k, p=0.001, distance=5, rounds=30, shots=10000,
                     keep_t=(1, 29), out_root="data"):
# Circuit information
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
    )

    dem = circuit.detector_error_model()
    coords = dem.get_detector_coordinates()

# Collect samples of detectors and logical observables
    sampler = circuit.compile_detector_sampler()
    dets, obs = sampler.sample(shots, separate_observables=True)
    dets = dets.astype(np.uint8)
    obs  = obs.astype(np.uint8)

# Convert data into arrays
    X = np.zeros((shots, 2, k, 6, 6), dtype=np.float32)
    y = np.zeros((shots,), dtype=np.uint8)

# Build the 6x6 grid of 
    for i in range(shots):
        det_values = dets[i]
        y[i] = obs[i, 0]

        events, mask = build_spacetime(coords, det_values, keep_t=keep_t)
        x = make_cnn_input(events, mask, k=k)

        X[i] = x

        if i % 1000 == 0:
            print(f"k={k} built {i}/{shots}")
   
#MWPM
    matching = pm.Matching.from_stim_circuit(circuit)

    pred_obs = matching.decode_batch(dets)  # shape (shots, num_obs)

    mwpm_pred = pred_obs[:, 0].astype(np.uint8)
    mwpm_true = obs[:, 0].astype(np.uint8)

    mwpm_acc = (mwpm_pred == mwpm_true).mean()
    print("MWPM acc:", float(mwpm_acc))
    print("MWPM logical error rate:", float(1.0 - mwpm_acc))

#save as file
    save_dir = os.path.join(out_root, f"surface_d{distance}_p{p}_k{k}")
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "X.npy"), X)
    np.save(os.path.join(save_dir, "y.npy"), y)

    with open(os.path.join(save_dir, "meta.txt"), "w") as f:
        f.write(
            "surface code: rotated_memory_x\n"
            f"distance: {distance}\n"
            f"rounds simulated: {rounds}\n"
            f"kept rounds: {keep_t}\n"
            f"history length k: {k}\n"
            f"physical error rate p: {p}\n"
            f"shots: {shots}\n"
            "label: final logical X flip\n"
            "channels:\n"
            "  0 = detector events\n"
            "  1 = detector existence mask\n"
        )

    print("saved dataset to", save_dir)
    print("X shape:", X.shape, "y mean:", float(y.mean()))
    return save_dir

if __name__ == "__main__":
    for k in [1, 5, 10, 15, 20, 25, 29]:
        generate_dataset(k=k)