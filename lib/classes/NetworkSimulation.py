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
    """
    Epidemic routing (store-carry-forward).

    Garantie : exactement 1 saut par pas de simulation.

    Clé du design : receive() n'ajoute le PDU QU'à self.receiving (pas à
    pdus_memory). simulate_reception() fait la migration en FIN de pas.
    Ainsi, un PDU reçu au pas T n'est dans pdus_memory qu'au pas T+1,
    et ne peut donc pas être retransmis dans le même pas (pas de chaîne
    multi-sauts instantanée).
    """

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

    def receive(self, pdu: PDU):
        if pdu.id in self._known_pdu_ids():
            return
        cloned = copy.deepcopy(pdu)
        cloned.trace.append(self.satellite.name)
        if cloned.dest_sat_name == self.satellite.name:
            self.pdus_where_this_router_is_the_destination.append(cloned)
        else:
            self.receiving.append(cloned)   # pdus_memory mis à jour en fin de pas

    def simulate_reception(self):
        while self.receiving:
            pdu = self.receiving.pop(0)
            self.pdus_memory.append(pdu)
            self.pdus_to_send.append(pdu)   # prêt à émettre dès le prochain pas

    def send(self):
        self.pdus_to_send.clear()
        for pdu in self.pdus_memory:
            for neighbor in self.satellite.list_coordinates[self.network_simulation.current_index].neighbors:
                neighbor_router = self.network_simulation.routers[neighbor.name]
                if pdu.id not in neighbor_router._known_pdu_ids():
                    neighbor_router.receive(pdu)
        self.pdus_to_send.extend(self.pdus_memory)  # reste actif tant qu'on porte des PDUs


class LinkStateRouter(RouterInterface):
    """
    Routage à état de lien — copie unique, plus court chemin (BFS).

    À chaque pas, un BFS est lancé depuis la position courante du routeur
    pour trouver le prochain saut vers la destination dans la topologie
    instantanée.  Le PDU est transmis d'un seul saut.  Si aucun chemin
    n'existe à cet instant, le PDU est conservé (store-and-wait) et la
    tentative est répétée au pas suivant.

    Différence clé avec Epidemic :
      - Une seule copie du PDU dans tout le réseau (pas de flood).
      - Plus rapide quand le chemin existe, mais ne peut pas exploiter
        une connectivité future si le réseau est partitionné maintenant.
    """

    def _known_ids(self) -> set:
        return {
            p.id
            for p in (
                self.pdus_to_send
                + self.pdus_memory
                + self.pdus_where_this_router_is_the_destination
                + self.receiving
            )
        }

    def receive(self, pdu: PDU):
        if pdu.id in self._known_ids():
            return
        cloned = copy.deepcopy(pdu)
        cloned.trace.append(self.satellite.name)
        if cloned.dest_sat_name == self.satellite.name:
            self.pdus_where_this_router_is_the_destination.append(cloned)
        else:
            self.receiving.append(cloned)

    def simulate_reception(self):
        while self.receiving:
            pdu = self.receiving.pop(0)
            self.pdus_memory.append(pdu)
            self.pdus_to_send.append(pdu)

    def _next_hop(self, dest_name: str, t: int) -> str | None:
        """BFS depuis ce satellite vers dest_name à l'instant t.

        Retourne le nom du premier saut, ou None si dest est inatteignable.
        """
        from collections import deque

        t = min(t, len(self.satellite.list_coordinates) - 1)

        visited = {self.satellite.name}
        queue: deque[tuple[str, str]] = deque()

        for nbr in self.satellite.list_coordinates[t].neighbors:
            if nbr.name == dest_name:
                return nbr.name          # voisin direct
            if nbr.name not in visited:
                visited.add(nbr.name)
                queue.append((nbr.name, nbr.name))  # (nœud courant, premier saut)

        while queue:
            current_name, first_hop = queue.popleft()
            current_sat = self.network_simulation.routers[current_name].satellite
            tc = min(t, len(current_sat.list_coordinates) - 1)

            for nbr in current_sat.list_coordinates[tc].neighbors:
                if nbr.name == dest_name:
                    return first_hop
                if nbr.name not in visited:
                    visited.add(nbr.name)
                    queue.append((nbr.name, first_hop))

        return None

    def send(self):
        t = self.network_simulation.current_index
        forwarded_ids: set[int] = set()

        for pdu in list(self.pdus_to_send):
            next_hop = self._next_hop(pdu.dest_sat_name, t)
            if next_hop is not None:
                self.network_simulation.routers[next_hop].receive(pdu)
                forwarded_ids.add(pdu.id)

        # Copie unique : l'émetteur efface le PDU dès qu'il l'a transmis
        self.pdus_to_send = [p for p in self.pdus_to_send if p.id not in forwarded_ids]
        self.pdus_memory  = [p for p in self.pdus_memory  if p.id not in forwarded_ids]
