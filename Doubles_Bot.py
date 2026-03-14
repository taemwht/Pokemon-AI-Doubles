import asyncio
import itertools
from typing import TYPE_CHECKING

import numpy as np
from poke_env.player import Player, RandomPlayer
from poke_env.player.battle_order import BattleOrder, SingleBattleOrder
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.move import Move
from poke_env.battle.side_condition import SideCondition
from poke_env import ServerConfiguration

if TYPE_CHECKING:
    from poke_env.battle.double_battle import DoubleBattle
    from poke_env.battle.pokemon import Pokemon


def _score_from_state(
    our_alive: int,
    opp_alive: int,
    our_hp_pct: float,
    opp_hp_pct: float,
    tailwind_ours: bool,
) -> float:
    """Raw heuristic score from state counts and HP (used by _evaluate_board and simulation)."""
    return (
        100 * our_alive
        - 100 * opp_alive
        + 50 * our_hp_pct
        - 50 * opp_hp_pct
        + (30 if tailwind_ours else 0)
    )


class SmartBot(Player):
    """Bot that uses a heuristic evaluation and simulates board score for move choice."""

    def _evaluate_board(self, battle: AbstractBattle) -> float:
        """Return a numerical score for the current battle state.
        +100 per our Pokémon alive, -100 per opponent alive;
        +50 * (our total HP fraction), -50 * (opponent total HP fraction);
        +30 if Tailwind is active on our side.
        """
        our_alive = sum(1 for p in battle.team.values() if not p.fainted)
        opp_alive = sum(1 for p in battle.opponent_team.values() if not p.fainted)
        our_hp_pct = sum(
            p.current_hp_fraction for p in battle.team.values() if not p.fainted
        )
        opp_hp_pct = sum(
            p.current_hp_fraction
            for p in battle.opponent_team.values()
            if not p.fainted
        )
        tailwind_ours = SideCondition.TAILWIND in battle.side_conditions
        return _score_from_state(our_alive, opp_alive, our_hp_pct, opp_hp_pct, tailwind_ours)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        # Singles: available_moves is a list; Doubles: list of two lists per active mon
        if battle.format_is_doubles:
            return self._choose_doubles_move(battle)
        return self._choose_singles_move(battle)

    def _choose_singles_move(self, battle: AbstractBattle) -> BattleOrder:
        moves = battle.available_moves
        opponent = battle.opponent_active_pokemon

        if not moves:
            return self.choose_random_move(battle)
        if opponent is None or opponent.fainted:
            return Player.create_order(moves[0])

        # Type effectiveness of each move against the opponent
        effectiveness = np.array([opponent.damage_multiplier(move) for move in moves])
        # Rank by effectiveness (highest first); break ties by index with stable sort
        ranks = np.argsort(-effectiveness, kind="stable")
        best_idx = int(ranks[0])
        return Player.create_order(moves[best_idx])

    def _estimate_damage_fraction(
        self, move: Move, target: "Pokemon | None"
    ) -> float:
        """Rough estimate of damage as fraction of target's max HP (0..1)."""
        if target is None or target.fainted:
            return 0.0
        if move.base_power == 0:
            return 0.0
        effectiveness = target.damage_multiplier(move)
        # Heuristic: (base_power/100) * effectiveness * 0.5 caps around 0.5–1.0 for strong hits
        return min(1.0, (move.base_power / 100.0) * effectiveness * 0.5)

    # Support moves that affect the ally's damage or board score (used in doubles simulation)
    _HELPING_HAND_MULTIPLIER = 1.5  # in-game: ally's move damage 1.5x

    def _is_support_move(self, order: SingleBattleOrder) -> bool:
        """True if this order is a support move (e.g. Helping Hand, Tailwind) that benefits the pair."""
        if not isinstance(order.order, Move):
            return False
        return order.order.id in {"helpinghand", "tailwind"}

    def _simulate_doubles_order(
        self,
        battle: "DoubleBattle",
        first_order: SingleBattleOrder,
        second_order: SingleBattleOrder,
    ) -> float:
        """Estimate board score after playing the given double battle order (full combination).
        Support moves are accounted for: Helping Hand boosts the ally's damage; Tailwind adds +30 to score.
        """
        from poke_env.battle.double_battle import DoubleBattle

        battle = battle  # type: DoubleBattle
        our_alive = sum(1 for p in battle.team.values() if not p.fainted)
        opp_alive = sum(1 for p in battle.opponent_team.values() if not p.fainted)
        our_hp_pct = sum(
            p.current_hp_fraction for p in battle.team.values() if not p.fainted
        )
        opp_active = battle.opponent_active_pokemon
        opp_hp = [
            opp_active[0].current_hp_fraction if opp_active[0] and not opp_active[0].fainted else 0.0,
            opp_active[1].current_hp_fraction if opp_active[1] and not opp_active[1].fainted else 0.0,
        ]
        opp_hp_pct_total = sum(
            p.current_hp_fraction for p in battle.opponent_team.values() if not p.fainted
        )
        tailwind_ours = SideCondition.TAILWIND in battle.side_conditions

        # If the *other* slot used Helping Hand, our damaging move gets 1.5x
        ally_used_helping_hand = [False, False]  # [for slot 0, for slot 1]
        if isinstance(second_order.order, Move) and second_order.order.id == "helpinghand":
            ally_used_helping_hand[0] = True  # slot 1 used HH → boost slot 0's damage
        if isinstance(first_order.order, Move) and first_order.order.id == "helpinghand":
            ally_used_helping_hand[1] = True  # slot 0 used HH → boost slot 1's damage

        def apply_order(order: SingleBattleOrder, slot: int) -> None:
            nonlocal opp_hp, opp_alive, tailwind_ours
            if not isinstance(order.order, Move):
                return
            move = order.order
            if move.id == "tailwind":
                tailwind_ours = True
            if move.base_power == 0:
                return
            tidx = order.move_target
            idx = 0 if tidx == 1 else (1 if tidx == 2 else -1)
            if idx < 0 or idx >= len(opp_active) or not opp_active[idx] or opp_active[idx].fainted:
                return
            target = opp_active[idx]
            dmg = self._estimate_damage_fraction(move, target)
            dmg *= move.accuracy
            if ally_used_helping_hand[slot]:
                dmg *= self._HELPING_HAND_MULTIPLIER
            opp_hp[idx] = max(0.0, opp_hp[idx] - dmg)
            if opp_hp[idx] <= 0:
                opp_alive -= 1

        apply_order(first_order, 0)
        apply_order(second_order, 1)

        # Simulated opponent HP: subtract current active contributions, add simulated
        active_contrib = 0.0
        for i, p in enumerate(opp_active):
            if p is not None and not p.fainted and i < 2:
                active_contrib += p.current_hp_fraction
        opp_hp_pct_sim = opp_hp_pct_total - active_contrib + opp_hp[0] + opp_hp[1]
        opp_hp_pct_sim = max(0.0, opp_hp_pct_sim)

        return _score_from_state(our_alive, opp_alive, our_hp_pct, opp_hp_pct_sim, tailwind_ours)

    def _choose_doubles_move(self, battle: AbstractBattle) -> BattleOrder:
        from poke_env.battle.double_battle import DoubleBattle
        from poke_env.player.battle_order import (
            DoubleBattleOrder,
            PassBattleOrder,
            DefaultBattleOrder,
        )

        battle = battle  # type: DoubleBattle
        if any(battle.force_switch):
            return self.choose_random_doubles_move(battle)

        valid = battle.valid_orders
        if not valid or not valid[0] or not valid[1]:
            return self.choose_random_doubles_move(battle)

        # Evaluate every (slot0, slot1) move combination; support moves (Helping Hand, Tailwind) are modeled in simulation
        best_score = float("-inf")
        best_order = None
        for o1, o2 in itertools.product(valid[0], valid[1]):
            score = self._simulate_doubles_order(battle, o1, o2)
            if score > best_score:
                best_score = score
                best_order = (o1, o2)

        if best_order is not None:
            return DoubleBattleOrder(best_order[0], best_order[1])
        return self.choose_random_doubles_move(battle)


async def main():
    # Connect to the stadium running in your other tab
    local_server = ServerConfiguration("ws://localhost:8000/showdown/websocket", "http://localhost:8000")

    # Create two bots to test the connection
    bot_1 = RandomPlayer(battle_format="gen9vgc2024regg", server_configuration=local_server)
    bot_2 = RandomPlayer(battle_format="gen9vgc2024regg", server_configuration=local_server)

    print("Testing connection... Battle starting!")
    await bot_1.battle_against(bot_2, n_battles=1)
    print("Success! The bots played a full game.")

if __name__ == "__main__":
    asyncio.run(main())