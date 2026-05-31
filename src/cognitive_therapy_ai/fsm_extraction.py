"""
Gate 5: FSM Extraction from Trained Agents

This module extracts finite state machines (FSMs) from trained LSTM agents.
The extraction process enables mechanistic analysis of learned representations.

Pipeline:
1. Rollout: Generate trajectories with epsilon-greedy exploration
2. Clustering: Identify discrete states via geometric clustering of hidden states
3. L* Algorithm: Learn FSM from input-output traces
4. Hopcroft: Minimize FSM to canonical form
5. Validation: Compare FSM behavior to analytic best response

This is critical for the research claim: the training environment shapes an agent's
internal representation, which we can extract and analyze as a discrete automaton.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Optional
import numpy as np
import torch
from sklearn.cluster import KMeans
from collections import defaultdict, deque

from cognitive_therapy_ai.games import Action
from cognitive_therapy_ai.encoding import ObservationEncoder
from cognitive_therapy_ai.representation_agent import RepresentationAgent


@dataclass
class Trajectory:
    """Single rollout trajectory."""
    observations: List[np.ndarray]  # One-hot encoded observations
    actions: List[Action]  # Agent actions
    hidden_states: List[np.ndarray]  # LSTM hidden states (h_t)
    rewards: List[float]  # Rewards received
    opponent_actions: List[Action]  # Opponent actions


@dataclass
class StateCluster:
    """Clustered discrete state."""
    cluster_id: int  # Discrete state ID
    centroid: np.ndarray  # Cluster centroid in hidden state space
    members: List[Tuple[int, int]]  # (trajectory_idx, timestep) pairs
    size: int  # Number of members


@dataclass
class FSMTransition:
    """Single FSM transition."""
    from_state: int  # Source state ID
    input_symbol: str  # Input alphabet symbol (e.g., "CC", "CD")
    to_state: int  # Destination state ID
    output_action: Action  # Output action


@dataclass
class FSM:
    """Extracted finite state machine."""
    states: Set[int]  # Discrete state IDs
    alphabet: List[str]  # Input symbols (e.g., ["START", "CC", "CD", "DC", "DD"])
    transitions: Dict[Tuple[int, str], Tuple[int, Action]]  # (state, input) -> (next_state, action)
    initial_state: int  # Starting state
    n_states: int  # Number of states
    
    def get_action(self, state: int, input_symbol: str) -> Optional[Action]:
        """Get action for state and input."""
        key = (state, input_symbol)
        if key in self.transitions:
            return self.transitions[key][1]
        return None
    
    def get_next_state(self, state: int, input_symbol: str) -> Optional[int]:
        """Get next state for current state and input."""
        key = (state, input_symbol)
        if key in self.transitions:
            return self.transitions[key][0]
        return None
    
    def is_complete(self) -> bool:
        """Check if FSM is complete (all transitions defined)."""
        for state in self.states:
            for symbol in self.alphabet:
                if (state, symbol) not in self.transitions:
                    return False
        return True


class RolloutCollector:
    """Collect trajectories from trained agent with exploration."""
    
    def __init__(
        self,
        agent: RepresentationAgent,
        encoder: ObservationEncoder,
        device: torch.device = None
    ):
        self.agent = agent
        self.encoder = encoder
        self.device = device or torch.device('cpu')
        self.agent.eval()  # Evaluation mode
    
    def collect_trajectory(
        self,
        env,
        max_steps: int,
        epsilon_explore: float = 0.1,
        deterministic_after_explore: bool = True
    ) -> Trajectory:
        """
        Collect single trajectory with epsilon-greedy exploration.
        
        Args:
            env: SessionEnvironment instance
            max_steps: Maximum steps per trajectory
            epsilon_explore: Probability of random action
            deterministic_after_explore: If True, use deterministic policy after exploration
        
        Returns:
            Trajectory with observations, actions, hidden states, rewards
        """
        observations = []
        actions = []
        hidden_states = []
        rewards = []
        opponent_actions = []
        
        obs = env.reset()
        hidden_state = self.agent.reset_hidden_state(batch_size=1)
        
        for step in range(max_steps):
            observations.append(obs)
            
            # Epsilon-greedy exploration
            if np.random.rand() < epsilon_explore:
                # Random action (integer)
                action = np.random.randint(0, 2)
            else:
                # Policy action - convert obs to tensor
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    action, _, hidden_state = self.agent.select_action(
                        obs_tensor, hidden_state,
                        deterministic=deterministic_after_explore,
                        epsilon=0.0
                    )
            
            # Store hidden state (h component of LSTM)
            h_t = hidden_state[0].cpu().numpy().flatten()
            hidden_states.append(h_t)
            
            # Step environment (action is integer 0 or 1)
            next_obs, reward, done = env.step(action)
            
            # Convert action to Action enum for storage
            action_enum = Action.COOPERATE if action == 0 else Action.DEFECT
            actions.append(action_enum)
            rewards.append(reward)
            
            # Extract opponent action from next observation
            # Observation encodes: agent_action(2) + opp_action(2) + outcome(4)
            # For one-hot encoding, find which opponent action index is 1
            obs_vector = next_obs if isinstance(next_obs, np.ndarray) else next_obs
            if obs_vector.sum() > 0:  # Not START token
                opp_action_start = 2  # After agent_action(2)
                opp_action_bits = obs_vector[opp_action_start:opp_action_start+2]
                if opp_action_bits[0] == 1:
                    opponent_actions.append(Action.COOPERATE)
                elif opp_action_bits[1] == 1:
                    opponent_actions.append(Action.DEFECT)
                else:
                    opponent_actions.append(None)  # Unclear
            else:
                opponent_actions.append(None)  # START token
            
            if done:
                break
            
            obs = next_obs
        
        return Trajectory(
            observations=observations,
            actions=actions,
            hidden_states=hidden_states,
            rewards=rewards,
            opponent_actions=opponent_actions
        )
    
    def collect_multiple_trajectories(
        self,
        env,
        n_trajectories: int,
        max_steps: int,
        epsilon_explore: float = 0.1
    ) -> List[Trajectory]:
        """Collect multiple trajectories."""
        trajectories = []
        for _ in range(n_trajectories):
            traj = self.collect_trajectory(env, max_steps, epsilon_explore)
            trajectories.append(traj)
        return trajectories


class HiddenStateClusterer:
    """Cluster hidden states to identify discrete FSM states."""
    
    def __init__(self, n_clusters: int, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.centroids = None
    
    def fit(self, trajectories: List[Trajectory]) -> List[StateCluster]:
        """
        Cluster hidden states across all trajectories.
        
        Args:
            trajectories: List of collected trajectories
        
        Returns:
            List of StateCluster objects
        """
        # Collect all hidden states
        all_hidden_states = []
        state_to_traj_time = []  # (traj_idx, time_idx) for each hidden state
        
        for traj_idx, traj in enumerate(trajectories):
            for time_idx, h_t in enumerate(traj.hidden_states):
                all_hidden_states.append(h_t)
                state_to_traj_time.append((traj_idx, time_idx))
        
        # Convert to array (ensure float64 for sklearn)
        X = np.array(all_hidden_states, dtype=np.float64)
        
        # Fit k-means
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        labels = self.kmeans.fit_predict(X)
        self.centroids = self.kmeans.cluster_centers_
        
        # Group by cluster
        cluster_members = defaultdict(list)
        for idx, label in enumerate(labels):
            cluster_members[label].append(state_to_traj_time[idx])
        
        # Create StateCluster objects
        clusters = []
        for cluster_id in range(self.n_clusters):
            members = cluster_members[cluster_id]
            clusters.append(StateCluster(
                cluster_id=cluster_id,
                centroid=self.centroids[cluster_id],
                members=members,
                size=len(members)
            ))
        
        return clusters
    
    def predict(self, hidden_state: np.ndarray) -> int:
        """Predict cluster ID for a hidden state."""
        if self.kmeans is None:
            raise ValueError("Must call fit() before predict()")
        # Ensure float64 for sklearn
        hidden_state_64 = hidden_state.astype(np.float64)
        return self.kmeans.predict(hidden_state_64.reshape(1, -1))[0]


class LStarExtractor:
    """
    L* algorithm for FSM extraction.
    
    Learns a minimal DFA from input-output traces.
    This is a simplified version that builds the FSM directly from clustered trajectories.
    """
    
    def __init__(self, alphabet: List[str]):
        self.alphabet = alphabet
    
    def extract_fsm(
        self,
        trajectories: List[Trajectory],
        clusters: List[StateCluster],
        clusterer: HiddenStateClusterer,
        encoder: ObservationEncoder
    ) -> FSM:
        """
        Extract FSM from clustered trajectories.
        
        Args:
            trajectories: Collected trajectories
            clusters: Clustered states
            clusterer: Trained clusterer for state assignment
            encoder: Observation encoder for symbol mapping
        
        Returns:
            Extracted FSM
        """
        # Build transition table
        transitions = {}
        state_visits = set()
        
        for traj in trajectories:
            for t in range(len(traj.hidden_states)):
                # Current state
                current_cluster = clusterer.predict(traj.hidden_states[t])
                state_visits.add(current_cluster)
                
                # Input symbol from observation
                obs = traj.observations[t]
                input_symbol = self._observation_to_symbol(obs, encoder)
                
                # Output action
                output_action = traj.actions[t]
                
                # Next state (if exists)
                if t + 1 < len(traj.hidden_states):
                    next_cluster = clusterer.predict(traj.hidden_states[t + 1])
                    
                    # Record transition
                    key = (current_cluster, input_symbol)
                    transitions[key] = (next_cluster, output_action)
        
        # Create FSM
        fsm = FSM(
            states=state_visits,
            alphabet=self.alphabet,
            transitions=transitions,
            initial_state=0,  # Assume cluster 0 is initial
            n_states=len(state_visits)
        )
        
        return fsm
    
    def _observation_to_symbol(
        self,
        obs: np.ndarray,
        encoder: ObservationEncoder
    ) -> str:
        """Convert observation to alphabet symbol."""
        # Check if START token (all zeros except maybe game tag)
        history_dim = encoder.history_dim
        if obs[:history_dim].sum() == 0:
            return "START"
        
        # Extract outcome from one-hot encoding
        # agent_action(2) + opp_action(2) + outcome(4)
        outcome_start = 4  # After agent(2) and opp(2)
        outcome_bits = obs[outcome_start:outcome_start+4]
        
        outcome_map = {0: "CC", 1: "CD", 2: "DC", 3: "DD"}
        outcome_idx = np.argmax(outcome_bits)
        return outcome_map[outcome_idx]


class HopcroftMinimizer:
    """Hopcroft's algorithm for DFA minimization."""
    
    def minimize(self, fsm: FSM) -> FSM:
        """
        Minimize FSM using Hopcroft's algorithm.
        
        This is a simplified version that merges equivalent states.
        Two states are equivalent if they produce the same outputs and
        transition to equivalent states for all inputs.
        
        Args:
            fsm: Input FSM
        
        Returns:
            Minimized FSM
        """
        # Start with all states in one partition, then refine
        partitions = [set(fsm.states)]
        
        # Iteratively refine partitions
        changed = True
        max_iterations = 100
        iteration = 0
        
        while changed and iteration < max_iterations:
            changed = False
            iteration += 1
            new_partitions = []
            
            for partition in partitions:
                if len(partition) <= 1:
                    new_partitions.append(partition)
                    continue
                
                # Try to split this partition
                sub_partitions = self._split_partition(partition, fsm, partitions)
                
                if len(sub_partitions) > 1:
                    changed = True
                    new_partitions.extend(sub_partitions)
                else:
                    new_partitions.append(partition)
            
            partitions = new_partitions
        
        # Build minimized FSM
        return self._build_minimized_fsm(fsm, partitions)
    
    def _split_partition(
        self,
        partition: Set[int],
        fsm: FSM,
        all_partitions: List[Set[int]]
    ) -> List[Set[int]]:
        """Split partition based on distinguishability."""
        # Group states by their transition signatures
        signatures = defaultdict(set)
        
        for state in partition:
            sig = self._get_signature(state, fsm, all_partitions)
            signatures[sig].add(state)
        
        return list(signatures.values())
    
    def _get_signature(
        self,
        state: int,
        fsm: FSM,
        partitions: List[Set[int]]
    ) -> Tuple:
        """Get transition signature for a state."""
        sig = []
        for symbol in fsm.alphabet:
            next_state = fsm.get_next_state(state, symbol)
            action = fsm.get_action(state, symbol)
            
            # Find which partition the next state belongs to
            partition_idx = -1
            if next_state is not None:
                for idx, part in enumerate(partitions):
                    if next_state in part:
                        partition_idx = idx
                        break
            
            sig.append((partition_idx, action))
        
        return tuple(sig)
    
    def _build_minimized_fsm(
        self,
        original_fsm: FSM,
        partitions: List[Set[int]]
    ) -> FSM:
        """Build minimized FSM from partitions."""
        # Map old states to new states (partition indices)
        state_map = {}
        for part_idx, partition in enumerate(partitions):
            for state in partition:
                state_map[state] = part_idx
        
        # Build new transitions
        new_transitions = {}
        new_states = set()
        
        for (old_state, symbol), (old_next, action) in original_fsm.transitions.items():
            new_state = state_map[old_state]
            new_next = state_map[old_next]
            new_states.add(new_state)
            new_states.add(new_next)
            
            key = (new_state, symbol)
            new_transitions[key] = (new_next, action)
        
        # Find initial state
        new_initial = state_map[original_fsm.initial_state]
        
        return FSM(
            states=new_states,
            alphabet=original_fsm.alphabet,
            transitions=new_transitions,
            initial_state=new_initial,
            n_states=len(new_states)
        )


class FSMValidator:
    """Validate extracted FSM against analytic best response."""
    
    def __init__(self, encoder: ObservationEncoder):
        self.encoder = encoder
    
    def validate_against_best_response(
        self,
        fsm: FSM,
        game,
        opponent,
        n_validation_episodes: int = 100,
        max_steps: int = 100
    ) -> Dict[str, float]:
        """
        Validate FSM by comparing to best response.
        
        Args:
            fsm: Extracted FSM
            game: Game instance
            opponent: Opponent instance
            n_validation_episodes: Number of validation episodes
            max_steps: Steps per episode
        
        Returns:
            Dict with validation metrics
        """
        from cognitive_therapy_ai.best_response import analytic_best_response
        
        total_actions = 0
        correct_actions = 0
        
        for episode in range(n_validation_episodes):
            # Reset
            fsm_state = fsm.initial_state
            prev_agent_action = None
            prev_opp_action = None
            
            for step in range(max_steps):
                # Get FSM action
                if step == 0:
                    input_symbol = "START"
                else:
                    # Build symbol from previous actions
                    input_symbol = self._actions_to_symbol(
                        prev_agent_action, prev_opp_action
                    )
                
                fsm_action = fsm.get_action(fsm_state, input_symbol)
                
                if fsm_action is None:
                    break  # Incomplete FSM
                
                # Get best response action
                payoff_matrix = game.get_payoff_matrix()
                # For ProbabilisticOpponent, cooperation probability = 1 - defection_probability
                p_coop = 1.0 - opponent.defection_probability
                br_action = analytic_best_response(payoff_matrix, p_coop)
                
                # Compare
                total_actions += 1
                if fsm_action == br_action:
                    correct_actions += 1
                
                # Simulate opponent action
                opp_action = opponent.choose_action([], step)
                
                # Update for next step
                prev_agent_action = fsm_action
                prev_opp_action = opp_action
                
                # Transition FSM
                next_state = fsm.get_next_state(fsm_state, input_symbol)
                if next_state is None:
                    break
                fsm_state = next_state
        
        accuracy = correct_actions / total_actions if total_actions > 0 else 0.0
        
        return {
            'br_accuracy': accuracy,
            'total_actions': total_actions,
            'correct_actions': correct_actions,
            'fsm_complete': fsm.is_complete()
        }
    
    def _actions_to_symbol(
        self,
        agent_action: Action,
        opponent_action: Action
    ) -> str:
        """Convert action pair to symbol."""
        if agent_action == Action.COOPERATE:
            if opponent_action == Action.COOPERATE:
                return "CC"
            else:
                return "CD"
        else:
            if opponent_action == Action.COOPERATE:
                return "DC"
            else:
                return "DD"


def extract_fsm_from_agent(
    agent: RepresentationAgent,
    encoder: ObservationEncoder,
    env,
    n_clusters: int,
    n_trajectories: int = 500,
    max_steps: int = 100,
    epsilon_explore: float = 0.1,
    device: torch.device = None
) -> Tuple[FSM, List[Trajectory], List[StateCluster]]:
    """
    Complete FSM extraction pipeline.
    
    Args:
        agent: Trained agent
        encoder: Observation encoder
        env: Session environment
        n_clusters: Number of discrete states to extract
        n_trajectories: Number of rollout trajectories
        max_steps: Steps per trajectory
        epsilon_explore: Exploration probability
        device: Torch device
    
    Returns:
        (minimized_fsm, trajectories, clusters)
    """
    # 1. Collect trajectories
    collector = RolloutCollector(agent, encoder, device)
    trajectories = collector.collect_multiple_trajectories(
        env, n_trajectories, max_steps, epsilon_explore
    )
    
    # 2. Cluster hidden states
    clusterer = HiddenStateClusterer(n_clusters)
    clusters = clusterer.fit(trajectories)
    
    # 3. Extract FSM with L*
    alphabet = encoder.get_alphabet()
    extractor = LStarExtractor(alphabet)
    fsm = extractor.extract_fsm(trajectories, clusters, clusterer, encoder)
    
    # 4. Minimize with Hopcroft
    minimizer = HopcroftMinimizer()
    minimized_fsm = minimizer.minimize(fsm)
    
    return minimized_fsm, trajectories, clusters
