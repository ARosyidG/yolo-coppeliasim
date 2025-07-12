from coppeliasim_zmqremoteapi_client import RemoteAPIClient

def main():
    client = RemoteAPIClient()
    sim = client.require('sim')
    target = sim.getObject('/target')
    while sim.getSimulationState() != sim.simulation_stopped:
        sim.setObjectPosition(target, -1, [0.5, 0.5, 0.5])

if __name__ == "__main__":
    main()