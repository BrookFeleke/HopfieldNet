# main.py

from model.Hopfield_Net import HopfieldNet
from utils.data import load_mnist_data
from utils.visualization import plot_predictions, plot_energy_transition

if __name__ == '__main__':
    fetch = load_mnist_data(size=3, error_rate=0.15)
    bias = 45
    model = HopfieldNet(inputs=fetch.original)

    # Synchronous prediction
    # All are updated at the same time
    sync_states, sync_energies = [], []
    for i in range(len(fetch.original)):
        result = model.predict(fetch.original[i], bias=bias, sync=True)
        sync_states.append([fetch.original[i], fetch.noised[i], result.states[-1]])
        sync_energies.append(result.energies)
    plot_predictions(sync_states, sync=True)
    plot_energy_transition(sync_energies, sync=True)

    # Asynchronous prediction
    # States are updated one by one
    async_states, async_energies = [], []
    for i in range(len(fetch.original)):
        result = model.predict(fetch.original[i], bias=bias, sync=False)
        async_states.append([fetch.original[i], fetch.noised[i], result.states[-1]])
        async_energies.append(result.energies)
    plot_predictions(async_states, sync=False)
    plot_energy_transition(async_energies, sync=False)
