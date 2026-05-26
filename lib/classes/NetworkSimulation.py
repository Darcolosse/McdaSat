from dataclasses import dataclass
from .Satellite import Satellite
import copy


@dataclass
class PDU:

    def __init__(self, id, source_sat_name: str, dest_sat_name: str):
        self.id = id
        self.source_sat_name = source_sat_name
        self.dest_sat_name = dest_sat_name
        self.trace = []

    def __str__(self):
        route = " -> ".join(self.trace)

        # Add destination if not already reached
        if self.trace[-1] != self.dest_sat_name:
            route += f" -> trying to reach {self.dest_sat_name}"

        return (
            f"PDU #{self.id}\n"
            f"  From : {self.source_sat_name}\n"
            f"  To   : {self.dest_sat_name}\n"
            f"  Path : {route}"
        )


@dataclass
class RouterInterface:

    def __init__(self, satellite: Satellite, network_simulation):
        self.receiving: list[PDU] = []
        self.pdus_to_send: list[PDU] = []
        self.pdus_memory: list[PDU] = []
        self.pdus_where_this_router_is_the_destination: list[PDU] = []
        self.satellite = satellite
        self.network_simulation = network_simulation

    def simulate_reception(self):
        while self.receiving:
            self.pdus_to_send.append(self.receiving.pop(0))

    def any_messages_to_send(self) -> bool:
        return len(self.pdus_to_send) > 0

    def send(self):
        raise NotImplementedError

    def receive(self, pdu: PDU):
        # If packet already stored, drop it, else store it
        is_pdu_already_in_memory = any(pdu.id == stored_pdu.id for stored_pdu in self.pdus_to_send + self.pdus_memory + self.pdus_where_this_router_is_the_destination)
        if not is_pdu_already_in_memory:
            shallow_proofed = copy.deepcopy(pdu)
            shallow_proofed.trace.append(self.satellite.name)
            if shallow_proofed.dest_sat_name == self.satellite.name:
                self.pdus_where_this_router_is_the_destination.append(shallow_proofed)
            else:
                self.pdus_memory.append(shallow_proofed)
                self.receiving.append(shallow_proofed)


@dataclass
class NetworkSimulation:

    def __init__(self, satellites: list[Satellite], router_class, scenario: list[tuple[int, str]]):
        self.routers = {sat.name: router_class(sat, self) for sat in satellites}
        self.current_index = 0
        self.scenario = copy.deepcopy(scenario)

    def next(self):
        # Send PDUs - can't be in the same loop because of instant retransmission
        for router in self.routers.values():
            if router.any_messages_to_send():
                router.send()
                
        # Create PDUs in the routers memory like if they are receiving it
        for i in reversed(range(len(self.scenario))):
            instant = self.scenario[i][0]
            pdu = self.scenario[i][1]
            if instant == self.current_index:
                self.routers[pdu.source_sat_name].receive(pdu)
                self.scenario.pop(i)
                
        # Simulate reception of transmitting packets
        for router in self.routers.values():
            router.simulate_reception()

        self.current_index += 1


class FloodingRouter(RouterInterface):

    def send(self):
        while self.pdus_to_send:
            pdu = self.pdus_to_send.pop(0)
            for neighbor in self.satellite.list_coordinates[self.network_simulation.current_index].neighbors:
                self.network_simulation.routers[neighbor.name].receive(pdu)


class EpidemicRouter(RouterInterface):
    """Epidemic routing: store-carry-forward with summary-vector exchange."""

    def _known_pdu_ids(self) -> set:
        return {
            p.id
            for p in (
                self.pdus_to_send
                + self.pdus_memory
                + self.pdus_where_this_router_is_the_destination
                + self.receiving
            )
        }

    def any_messages_to_send(self) -> bool:
        return len(self.pdus_memory) > 0 or len(self.pdus_to_send) > 0

    def send(self):
        self.pdus_to_send.clear()
        for pdu in self.pdus_memory:
            for neighbor in self.satellite.list_coordinates[self.network_simulation.current_index].neighbors:
                neighbor_router = self.network_simulation.routers[neighbor.name]
                if pdu.id not in neighbor_router._known_pdu_ids():
                    neighbor_router.receive(pdu)
